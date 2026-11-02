from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
import warnings

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load provider-native environment variables without overwriting OS variables.
load_dotenv(PROJECT_ROOT / ".env", override=False)


RESERVED_LLM_PARAMS = {
    "model",
    "messages",
    "api_key",
    "api_base",
    "base_url",
    "timeout",
    "temperature",
    "max_tokens",
    "response_format",
    "drop_params",
    "stream",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Router LLM
    router_provider: str | None = Field(default=None, alias="ROUTER_PROVIDER")
    router_model: str = Field(default="", alias="ROUTER_MODEL")
    router_api_key: SecretStr | None = Field(default=None, alias="ROUTER_API_KEY", repr=False)
    router_api_base: str | None = Field(default=None, alias="ROUTER_API_BASE")
    router_litellm_params: dict[str, Any] = Field(default_factory=dict, alias="ROUTER_LITELLM_PARAMS")

    # Answer LLM
    answer_provider: str | None = Field(default=None, alias="ANSWER_PROVIDER")
    answer_model: str = Field(default="", alias="ANSWER_MODEL")
    answer_api_key: SecretStr | None = Field(default=None, alias="ANSWER_API_KEY", repr=False)
    answer_api_base: str | None = Field(default=None, alias="ANSWER_API_BASE")
    answer_litellm_params: dict[str, Any] = Field(default_factory=dict, alias="ANSWER_LITELLM_PARAMS")

    llm_timeout_seconds: Annotated[int, Field(ge=5, le=300)] = Field(default=45, alias="LLM_TIMEOUT_SECONDS")
    llm_drop_unsupported_params: bool = Field(default=True, alias="LLM_DROP_UNSUPPORTED_PARAMS")

    # Application
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: Annotated[int, Field(ge=1, le=65535)] = Field(default=8000, alias="APP_PORT")
    max_history_messages: Annotated[int, Field(ge=2, le=100)] = Field(
        default=12, alias="MAX_HISTORY_MESSAGES"
    )

    # RAG
    rag_top_k: Annotated[int, Field(ge=1, le=20)] = Field(default=5, alias="RAG_TOP_K")
    rag_chunk_size: Annotated[int, Field(ge=200, le=5000)] = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: Annotated[int, Field(ge=0, le=2000)] = Field(default=150, alias="RAG_CHUNK_OVERLAP")
    rag_min_similarity: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.22, alias="RAG_MIN_SIMILARITY"
    )
    rag_max_context_chars: Annotated[int, Field(ge=1000, le=50000)] = Field(
        default=14000, alias="RAG_MAX_CONTEXT_CHARS"
    )

    embedding_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY", repr=False)
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_output_dimensionality: Annotated[int, Field(ge=128, le=3072)] = Field(
        default=768, alias="EMBEDDING_OUTPUT_DIMENSIONALITY"
    )
    embedding_batch_size: Annotated[int, Field(ge=1, le=100)] = Field(
        default=16, alias="EMBEDDING_BATCH_SIZE"
    )

    knowledge_dir: Path = Field(default=PROJECT_ROOT / "knowledge", alias="KNOWLEDGE_DIR")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DATA_DIR")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_DIR")
    sqlite_path: Path = Field(default=PROJECT_ROOT / "data" / "assistant.db", alias="SQLITE_PATH")

    router_confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.70, alias="ROUTER_CONFIDENCE_THRESHOLD"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="after")
    def validate_settings(self):
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

        if not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL cannot be empty")
        if self.embedding_model.startswith("sentence-transformers/"):
            warnings.warn(
                "The local Sentence Transformers model setting is obsolete; switching to "
                "gemini-embedding-001. Update EMBEDDING_MODEL in your .env/Render Environment.",
                UserWarning,
                stacklevel=2,
            )
            self.embedding_model = "gemini-embedding-001"
        if self.embedding_model != "gemini-embedding-001":
            raise ValueError(
                "This remote embedding adapter currently supports only EMBEDDING_MODEL=gemini-embedding-001"
            )

        for role, params in (
            ("router", self.router_litellm_params),
            ("answer", self.answer_litellm_params),
        ):
            invalid = RESERVED_LLM_PARAMS.intersection(params)

            if invalid:
                raise ValueError(
                    f"{role.upper()}_LITELLM_PARAMS contains reserved parameters: "
                    f"{', '.join(sorted(invalid))}"
                )

        return self

    def provider_configuration(self, role: str) -> dict[str, Any]:
        if role == "router":
            provider = self.router_provider
            model = self.router_model
            api_key = self.router_api_key
            api_base = self.router_api_base
            params = self.router_litellm_params

        elif role == "answer":
            provider = self.answer_provider
            model = self.answer_model
            api_key = self.answer_api_key
            api_base = self.answer_api_base
            params = self.answer_litellm_params

        else:
            raise ValueError(f"Unknown LLM role: {role}")

        return {
            "provider": (provider.strip().lower() if provider and provider.strip() else None),
            "model": model.strip(),
            "api_key": (
                api_key.get_secret_value() if api_key and api_key.get_secret_value().strip() else None
            ),
            "api_base": api_base.strip() if api_base and api_base.strip() else None,
            "litellm_params": dict(params),
        }

    def provider_ready(self, role: str) -> bool:
        cfg = self.provider_configuration(role)

        if not cfg["model"]:
            return False

        # Explicit provider OR a fully qualified LiteLLM model identifier.
        return bool(cfg["provider"] or "/" in cfg["model"])

    def readiness_issues(self) -> list[str]:
        issues = []

        for role in ("router", "answer"):
            cfg = self.provider_configuration(role)

            if not cfg["model"]:
                issues.append(f"{role.upper()}_MODEL is not configured")

            elif not cfg["provider"] and "/" not in cfg["model"]:
                issues.append(f"{role.upper()}_PROVIDER is required for an unqualified model")

        if not self.embedding_api_key or not self.embedding_api_key.get_secret_value().strip():
            issues.append("GEMINI_API_KEY is not configured for remote embeddings")

        return issues


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
