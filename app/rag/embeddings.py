from __future__ import annotations

import logging
import math
import threading
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class RemoteEmbeddingModel:
    """Generate retrieval embeddings through Google's Gemini Embedding API.

    The class intentionally exposes the same encode_documents/encode_query interface
    used by the existing RAG pipeline, without loading a local ML model or PyTorch.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        output_dimensionality: int = 768,
        batch_size: int = 16,
    ):
        if model_name.strip() != "gemini-embedding-001":
            raise ValueError("This adapter supports only the gemini-embedding-001 model")
        if not 128 <= output_dimensionality <= 3072:
            raise ValueError("Embedding output dimensionality must be between 128 and 3072")
        if not 1 <= batch_size <= 100:
            raise ValueError("Embedding batch size must be between 1 and 100")

        self.model_name = model_name.strip()
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size
        self._client: Any | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the lightweight remote API client has been initialized."""
        return self._client is not None

    def _create_client(self):
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add it to the local .env file "
                "or the Render Environment settings."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        return genai.Client(api_key=self.api_key)

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._create_client()
                    logger.info(
                        "Remote embedding client initialized model=%s dimensions=%s",
                        self.model_name,
                        self.output_dimensionality,
                    )
        return self._client

    def initialize(self) -> None:
        """Initialize the API client only; no embedding model is loaded into RAM."""
        self._get_client()

    def _build_config(self, task_type: str):
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc

        return types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self.output_dimensionality,
        )

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:
            raise ValueError("Gemini returned a zero-length embedding vector")
        return [value / norm for value in values]

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        response = self._get_client().models.embed_content(
            model=self.model_name,
            contents=list(texts),
            config=self._build_config(task_type),
        )
        response_embeddings = getattr(response, "embeddings", None)
        if not response_embeddings:
            raise RuntimeError("Gemini returned no embeddings")
        if len(response_embeddings) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(response_embeddings)} embeddings for {len(texts)} inputs"
            )

        vectors = []
        for item in response_embeddings:
            values = getattr(item, "values", None)
            if values is None or len(values) != self.output_dimensionality:
                actual = len(values) if values is not None else 0
                raise RuntimeError(
                    f"Expected an embedding with {self.output_dimensionality} dimensions; received {actual}"
                )
            vectors.append(self._normalize(values))
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate document vectors in bounded batches for retrieval indexing."""
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vectors.extend(self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
        return all_vectors

    def encode_query(self, text: str) -> list[float]:
        """Generate a query vector compatible with RETRIEVAL_DOCUMENT vectors."""
        if not text.strip():
            raise ValueError("Embedding query cannot be empty")
        vectors = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    def dimension(self) -> int:
        return self.output_dimensionality
