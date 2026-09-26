"""MiniLM as the router's Explainer: closeness to the action descriptions, for display only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from router import DescriptionMatch
from policy import ACTION_CATALOG

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Hub download cache (safetensors live here after the first fetch).
MODEL_CACHE_DIR = Path(__file__).resolve(
).parents[1] / ".cache" / "sentence-transformers"
# Persistent embedding vectors so CLI runs can skip loading weights into RAM.
EMBEDDING_CACHE_DIR = Path(__file__).resolve(
).parents[1] / ".cache" / "embeddings"
_MODEL_CACHE: dict[str, object] = {}
# A chosen action below this cosine is shown as a description gap.
MIN_SCORE = 0.45


def cosine_scores(
    query_vector: np.ndarray, action_embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores: dict[str, float] = {}
    for action_id, action_vec in action_embeddings.items():
        action_norm = action_vec / np.linalg.norm(action_vec)
        scores[action_id] = float(np.dot(query_norm, action_norm))
    return scores


def _catalog_fingerprint(model_name: str, catalog: dict[str, str]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {"model": model_name, "catalog": catalog},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _query_fingerprint(model_name: str, text: str) -> str:
    import hashlib

    return hashlib.sha256(f"{model_name}\0{text}".encode()).hexdigest()


class ActionEmbedder:
    """MiniLM encoder with Hub weight cache and on-disk embedding cache.

    Hugging Face weights already persist under MODEL_CACHE_DIR. Each new CLI
    process still has to map those weights into RAM unless the needed vectors
    are already saved under EMBEDDING_CACHE_DIR — in that case the model is
    never loaded.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_folder: Path | str | None = MODEL_CACHE_DIR,
        embedding_cache_dir: Path | str | None = EMBEDDING_CACHE_DIR,
        use_embedding_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_folder = Path(
            cache_folder) if cache_folder else MODEL_CACHE_DIR
        self.embedding_cache_dir = (
            Path(embedding_cache_dir) if embedding_cache_dir else EMBEDDING_CACHE_DIR
        )
        self.use_embedding_cache = use_embedding_cache
        self._model = None
        self._action_embeddings: dict[str, np.ndarray] | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        cache_key = f"{self.model_name}|{self.cache_folder.resolve()}"
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model

        self.cache_folder.mkdir(parents=True, exist_ok=True)
        from sentence_transformers import SentenceTransformer

        print(
            f"Loading MiniLM weights into memory from {self.cache_folder} "
            "(one-time per process; embedding cache avoids this on repeat runs)...",
            flush=True,
        )
        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_folder),
                local_files_only=True,
            )
        except OSError:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_folder),
                local_files_only=False,
            )
        _MODEL_CACHE[cache_key] = self._model
        return self._model

    def _query_cache_path(self, text: str) -> Path:
        digest = _query_fingerprint(self.model_name, text)
        return self.embedding_cache_dir / "queries" / f"{digest}.npy"

    def _action_cache_paths(
        self, catalog: dict[str, str]
    ) -> tuple[Path, Path]:
        digest = _catalog_fingerprint(self.model_name, catalog)
        base = self.embedding_cache_dir / "actions" / digest
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def embed(self, text: str) -> np.ndarray:
        if self.use_embedding_cache:
            path = self._query_cache_path(text)
            if path.exists():
                return np.asarray(np.load(path), dtype=np.float64)

        vector = np.asarray(
            self._load_model().encode(text, normalize_embeddings=True),
            dtype=np.float64,
        )
        if self.use_embedding_cache:
            path = self._query_cache_path(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
        return vector

    def action_embeddings(
        self, catalog: dict[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        catalog = catalog or ACTION_CATALOG
        if self._action_embeddings is not None:
            return self._action_embeddings

        if self.use_embedding_cache:
            npz_path, meta_path = self._action_cache_paths(catalog)
            if npz_path.exists() and meta_path.exists():
                import json

                meta = json.loads(meta_path.read_text())
                if meta.get("model") == self.model_name and set(
                    meta.get("actions", [])
                ) == set(catalog):
                    loaded = np.load(npz_path)
                    self._action_embeddings = {
                        action_id: np.asarray(
                            loaded[action_id], dtype=np.float64)
                        for action_id in catalog
                    }
                    return self._action_embeddings

        self._action_embeddings = {
            action_id: self.embed(description)
            for action_id, description in catalog.items()
        }

        if self.use_embedding_cache:
            import json

            npz_path, meta_path = self._action_cache_paths(catalog)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_path, **self._action_embeddings)
            meta_path.write_text(
                json.dumps(
                    {
                        "model": self.model_name,
                        "actions": list(catalog.keys()),
                    },
                    indent=2,
                )
            )
        return self._action_embeddings


class MiniLMExplainer:
    """Never picks or blocks an action. A chosen action far from its description is a gap."""

    def __init__(self, embedder: ActionEmbedder | None = None) -> None:
        self.embedder = embedder or ActionEmbedder()

    def explain(self, text: str, action: str) -> DescriptionMatch:
        similarities = cosine_scores(
            self.embedder.embed(text), self.embedder.action_embeddings())
        nearest = max(similarities, key=similarities.get)
        chosen_score = float(similarities.get(action, 0.0))
        return DescriptionMatch(
            chosen=action,
            nearest=nearest,
            nearest_score=similarities[nearest],
            chosen_score=chosen_score,
            similarities=similarities,
            description_gap=nearest != action or chosen_score < MIN_SCORE,
        )
