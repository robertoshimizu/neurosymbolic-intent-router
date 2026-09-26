"""Tests for the MiniLM explainer and its embedding cache."""

from __future__ import annotations

import numpy as np
import pytest

from adapters.minilm import ActionEmbedder, MiniLMExplainer


@pytest.mark.integration
def test_minilm_agrees_with_wire_for_canonical_sentence() -> None:
    match = MiniLMExplainer().explain(
        "I want to send $5,000 to my external bank account.", "wire_transfer_funds")
    assert match.agrees
    assert not match.description_gap


def test_disk_cached_embed_does_not_load_model(tmp_path) -> None:
    text = "cached utterance"
    writer = ActionEmbedder(embedding_cache_dir=tmp_path)
    path = writer._query_cache_path(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    reader = ActionEmbedder(embedding_cache_dir=tmp_path)
    vector = reader.embed(text)
    assert reader._model is None
    np.testing.assert_array_equal(vector, [1.0, 0.0, 0.0])


def test_disk_cached_actions_do_not_load_model(tmp_path) -> None:
    catalog = {"wire_transfer_funds": "Send money", "view_public_faq": "FAQ"}
    writer = ActionEmbedder(embedding_cache_dir=tmp_path)
    npz_path, meta_path = writer._action_cache_paths(catalog)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        wire_transfer_funds=np.array([1.0, 0.0], dtype=np.float64),
        view_public_faq=np.array([0.0, 1.0], dtype=np.float64),
    )
    meta_path.write_text(
        '{"model": "%s", "actions": ["wire_transfer_funds", "view_public_faq"]}'
        % writer.model_name
    )

    reader = ActionEmbedder(embedding_cache_dir=tmp_path)
    vectors = reader.action_embeddings(catalog)
    assert reader._model is None
    assert set(vectors) == set(catalog)
