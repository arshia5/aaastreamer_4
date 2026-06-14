"""Runtime embedder: loads the fitted pipeline and turns movie metadata into
390-d vectors that match the training space.
"""
from __future__ import annotations

import threading

import joblib
import numpy as np

from app.ml import config, features


class PipelineNotFitted(RuntimeError):
    pass


class MovieEmbedder:
    """Lazily-loaded singleton wrapping the fitted artifacts + MiniLM model."""

    _instance: "MovieEmbedder | None" = None
    _lock = threading.Lock()

    def __init__(self, artifacts: dict):
        self.mlb = artifacts["genre_mlb"]
        self.scaler = artifacts["year_scaler"]
        self.median_year = artifacts["median_year"]
        self.pca = artifacts["pca"]
        self.model_name = artifacts.get("model_name", config.MODEL_NAME)
        self.explained_variance = artifacts.get("explained_variance")
        self.n_components = self.pca.n_components_
        self._st_model = None
        self._model_lock = threading.Lock()

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls) -> "MovieEmbedder":
        if not config.PIPELINE_PATH.exists():
            raise PipelineNotFitted(
                f"No fitted pipeline at {config.PIPELINE_PATH}. "
                "Run: python -m scripts.fit_embeddings"
            )
        artifacts = joblib.load(config.PIPELINE_PATH)
        return cls(artifacts)

    @classmethod
    def instance(cls) -> "MovieEmbedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.load()
        return cls._instance

    @property
    def st_model(self):
        if self._st_model is None:
            with self._model_lock:
                if self._st_model is None:
                    # Imported lazily so the heavy torch import only happens
                    # when an embedding is actually requested.
                    from sentence_transformers import SentenceTransformer

                    self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    # -- inference -------------------------------------------------------- #
    def embed_records(self, records: list[dict]) -> np.ndarray:
        raw = features.build_feature_matrix(
            records, self.st_model, self.mlb, self.scaler, self.median_year
        )
        return self.pca.transform(raw).astype(np.float32)

    def embed_one(self, record: dict) -> list[float]:
        return self.embed_records([record])[0].tolist()
