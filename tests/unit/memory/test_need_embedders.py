"""Tests for backend.memory.need_embedders: the gate embedders and how each one loads."""

from __future__ import annotations

from backend.memory.need_embedders import NEED_EMBEDDERS, NeedEmbedderSpec, need_embedder_kwargs


class TestNeedEmbedderSpec:
    def test_is_frozen_and_slotted(self):
        assert NeedEmbedderSpec.__dataclass_params__.frozen
        assert hasattr(NeedEmbedderSpec, "__slots__")


class TestEmbedders:
    def test_need_embedders_hold_the_prefixes_and_attention_settings(self):
        assert dict(NEED_EMBEDDERS) == {
            "shared": NeedEmbedderSpec("", None),
            "bge-small-en-v1.5": NeedEmbedderSpec(
                "Represent this sentence for searching relevant passages: ", None
            ),
            "granite-embedding-small-english-r2": NeedEmbedderSpec("", "sdpa"),
        }

    def test_kwargs_empty_without_an_attention_setting(self):
        assert need_embedder_kwargs(NEED_EMBEDDERS["shared"]) == {}
        assert need_embedder_kwargs(NEED_EMBEDDERS["bge-small-en-v1.5"]) == {}

    def test_kwargs_carry_the_one_keyword_with_an_attention_setting(self):
        spec = NEED_EMBEDDERS["granite-embedding-small-english-r2"]
        assert need_embedder_kwargs(spec) == {"attn_implementation": "sdpa"}
