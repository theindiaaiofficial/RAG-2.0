from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings, RESERVED_LLM_PARAMS

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelRole:
    role: str
    provider: str
    model: str
    api_key: str | None
    api_base: str | None
    litellm_params: dict[str, Any]


class LLMClient:
    """Universal provider-aware LiteLLM adapter."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def normalize_model(provider: str | None, model: str) -> str:
        model = model.strip()

        if not model:
            raise LLMConfigurationError("Model identifier is missing")

        if provider:
            provider = provider.strip().lower()

            if model.lower().startswith(provider + "/"):
                return model

            return f"{provider}/{model}"

        # Without explicit provider, require a qualified LiteLLM identifier.
        if "/" not in model:
            raise LLMConfigurationError(
                "Provider is missing. Configure PROVIDER or use a fully qualified model."
            )

        return model

    def _role(self, role: str) -> ModelRole:
        cfg = self.settings.provider_configuration(role)

        if not cfg["model"]:
            raise LLMConfigurationError(f"{role.upper()}_MODEL is not configured")

        provider = cfg["provider"]

        model = self.normalize_model(
            provider,
            cfg["model"],
        )

        if not provider:
            provider = model.split("/", 1)[0].lower()

        params = dict(cfg["litellm_params"])

        invalid = RESERVED_LLM_PARAMS.intersection(params)

        if invalid:
            raise LLMConfigurationError(
                f"Reserved LiteLLM parameters are not allowed: {', '.join(sorted(invalid))}"
            )

        return ModelRole(
            role=role,
            provider=provider,
            model=model,
            api_key=cfg["api_key"],
            api_base=cfg["api_base"],
            litellm_params=params,
        )

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        """Read provider retry hints; fall back to a short exponential delay."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None) or {}
        try:
            for key in ("retry-after", "Retry-After"):
                value = headers.get(key) if hasattr(headers, "get") else None
                if value is not None:
                    return min(30.0, max(0.25, float(value)))
        except (TypeError, ValueError):
            pass

        # Groq includes messages such as "Please try again in 2.865s".
        match = re.search(r"(?:try again in|retry in)\s*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
        if match:
            return min(30.0, max(0.25, float(match.group(1)) + 0.2))
        return min(8.0, 1.0 * (2 ** attempt))

    @staticmethod
    def _is_json_validation_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "json_validate_failed" in message or "failed to validate json" in message

    async def complete(
        self,
        role: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 1400,
    ) -> str:
        cfg = self._role(role)

        try:
            from litellm import acompletion

        except ImportError as exc:
            raise LLMProviderError("LiteLLM is not installed. Run: pip install -r requirements.txt") from exc

        kwargs: dict[str, Any] = dict(cfg.litellm_params)
        if cfg.provider == "groq" and "gpt-oss" in cfg.model.lower():
            kwargs.setdefault("reasoning_effort", "low")

        kwargs.update(
            {
                "model": cfg.model,
                "messages": messages,
                "temperature": temperature,
                "timeout": self.settings.llm_timeout_seconds,
                "max_tokens": max_tokens,
                "drop_params": self.settings.llm_drop_unsupported_params,
            }
        )

        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key

        if cfg.api_base:
            kwargs["api_base"] = cfg.api_base

        if response_format:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        json_fallback_used = False
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = await acompletion(**kwargs)
                message = response.choices[0].message
                content = message.content

                # Handle providers returning structured text blocks.
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )

                # Never use reasoning_content as the user-facing answer.
                if content is None:
                    finish_reason = response.choices[0].finish_reason
                    dump = getattr(message, "model_dump", None)
                    fields = list(dump(exclude_none=True).keys()) if callable(dump) else []
                    logger.warning(
                        "Empty LLM response role=%s provider=%s model=%s "
                        "finish_reason=%s message_fields=%s usage=%s",
                        role, cfg.provider, cfg.model, finish_reason, fields,
                        getattr(response, "usage", None),
                    )
                    raise LLMProviderError(
                        f"Provider returned no user-facing message. Finish reason: {finish_reason}"
                    )

                if not isinstance(content, str) or not content.strip():
                    raise LLMProviderError("Provider returned an empty text response")

                return content.strip()

            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None) or getattr(
                    getattr(exc, "response", None), "status_code", None
                )

                # Some Groq JSON-mode generations fail validation with an empty generated
                # payload. Retry once without response_format; the evaluator/verifier
                # system prompts still explicitly require JSON and parse it afterwards.
                if (
                    status == 400
                    and "response_format" in kwargs
                    and not json_fallback_used
                    and self._is_json_validation_error(exc)
                ):
                    kwargs.pop("response_format", None)
                    json_fallback_used = True
                    logger.warning(
                        "JSON response mode rejected by provider; retrying prompt-constrained JSON "
                        "role=%s provider=%s model=%s",
                        role, cfg.provider, cfg.model,
                    )
                    continue

                error_name = type(exc).__name__.lower()
                retryable = (
                    status == 429
                    or any(
                        value in error_name
                        for value in ("timeout", "connection", "serviceunavailable", "internalserver")
                    )
                    or (status is not None and status >= 500)
                )

                if retryable and attempt < max_attempts - 1:
                    delay = (
                        self._retry_delay_seconds(exc, attempt)
                        if status == 429
                        else min(8.0, 0.75 * (2 ** attempt) + random.random() * 0.25)
                    )
                    logger.warning(
                        "Retrying LLM request after provider transient error role=%s "
                        "provider=%s model=%s error_type=%s retry_in_seconds=%.2f",
                        role, cfg.provider, cfg.model, type(exc).__name__, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Never log credentials. Provider error text is retained only after redaction.
                safe_message = str(exc)
                if cfg.api_key:
                    safe_message = safe_message.replace(cfg.api_key, "[REDACTED]")
                safe_message = re.sub(
                    r"(?i)(authorization|api[_-]?key|bearer)\s*[:=]\s*\S+",
                    "[REDACTED]", safe_message,
                )[:700]
                logger.warning(
                    "LLM request failed role=%s provider=%s model=%s "
                    "error_type=%s http_status=%s provider_message=%s",
                    role, cfg.provider, cfg.model, type(exc).__name__, status, safe_message,
                )

        if last_error:
            error_name = type(last_error).__name__.lower()

            status = getattr(last_error, "status_code", None) or getattr(
                getattr(last_error, "response", None), "status_code", None
            )

            if "timeout" in error_name:
                raise LLMProviderError(f"{role.title()} provider request timed out") from None

            if status in (401, 403):
                raise LLMProviderError(
                    f"{role.title()} authentication failed. Check API key and provider credentials."
                ) from None

            if status == 404:
                raise LLMProviderError(
                    f"{role.title()} model or endpoint not found (404). "
                    f"Check {role.upper()}_PROVIDER, {role.upper()}_MODEL "
                    f"and {role.upper()}_API_BASE."
                ) from None

            if status == 429:
                raise LLMProviderError(f"{role.title()} provider rate limit exceeded") from None

            if status in (400, 422):
                raise LLMProviderError(
                    f"{role.title()} provider rejected the request. "
                    "Check model parameters and provider compatibility."
                ) from None

            raise LLMProviderError(
                f"{role.title()} model request failed "
                f"({type(last_error).__name__}). "
                "Check provider, model and authentication configuration."
            ) from None

        raise LLMProviderError(
            f"{role.title()} model request failed: {type(last_error).__name__}: {safe_message}"
        ) from None

    async def check_connection(self, role: str) -> tuple[bool, str]:
        try:
            result = await self.complete(
                role,
                [{"role": "user", "content": "Reply with exactly CONNECTION_OK"}],
                temperature=0,
                max_tokens=128,
            )

            if result and result.strip():
                return True, "Provider connection successful"

            return False, "Provider returned an empty response"

        except (LLMConfigurationError, LLMProviderError) as exc:
            return False, str(exc)
