"""Sentence-transformer embedding wrapper for VOC memo text."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL_PATH = "/Volumes/sandbox/z_jungryo_lee/tv_voc/bge-large-en-v1.5"


class Embedder:
    """Lazy sentence-transformer model loader.

    Databricks Volumes can be slower or unstable for direct model reads, so the
    model directory is copied to /tmp once per cluster process before loading.
    Embeddings are normalized so cosine similarity is equal to dot product.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
        *,
        local_cache_dir: str = "/tmp",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.local_cache_dir = str(local_cache_dir)
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _ensure_local_model_path(self) -> str:
        """Copy the model directory to local disk when needed."""
        source_path = Path(self.model_path)
        local_path = Path(self.local_cache_dir) / source_path.name

        if not local_path.exists():
            shutil.copytree(source_path, local_path)

        return str(local_path)

    def _get_model(self):
        """Load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - runtime dependency check
                raise ImportError(
                    "sentence-transformers is required for memo embedding. "
                    "Install it in the Databricks cluster or notebook first."
                ) from exc

            local_path = self._ensure_local_model_path()
            self._model = SentenceTransformer(local_path, device=self.device)

        return self._model

    def embedding_dim(self) -> int:
        """Return embedding dimension for the loaded model."""
        return int(self._get_model().get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into a float32 matrix."""
        if not texts:
            return np.zeros((0, self.embedding_dim()), dtype=np.float32)

        embeddings = self._get_model().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)
