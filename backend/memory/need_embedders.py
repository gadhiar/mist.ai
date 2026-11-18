"""The gate embedders the need gate can run, and how each one loads.

Kept apart from `need_gate` and on the standard library only, because `config` validates the
configured embedder's name against this table, and the extraction service imports `config`
through `backend.knowledge.config` in an image that installs no numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class NeedEmbedderSpec:
    """How to load one gate embedder and what to put before each text.

    Attributes:
        query_prefix: The model card's query instruction, or empty.
        attn_implementation: Passed to the model load when set. ModernBERT models pick flash
            attention by themselves when the package is installed, which a CPU forward cannot
            run, so such a model names `sdpa` here.
    """

    query_prefix: str
    attn_implementation: str | None


NEED_EMBEDDERS: Mapping[str, NeedEmbedderSpec] = MappingProxyType(
    {
        "shared": NeedEmbedderSpec("", None),
        "bge-small-en-v1.5": NeedEmbedderSpec(
            "Represent this sentence for searching relevant passages: ", None
        ),
        "granite-embedding-small-english-r2": NeedEmbedderSpec("", "sdpa"),
    }
)


def need_embedder_kwargs(spec: NeedEmbedderSpec) -> dict[str, str]:
    """The keyword arguments a gate embedder's load takes beyond its name and device.

    Empty without an attention setting, so a caller passes the keyword only when a spec needs
    it, and a loader that does not take it is never handed it.
    """
    if spec.attn_implementation is None:
        return {}
    return {"attn_implementation": spec.attn_implementation}
