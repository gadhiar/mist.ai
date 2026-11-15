"""The serving identity: one hash over every serving setting that changes extraction output.

An extraction epoch is keyed on (ontology_version, extraction_version,
model_hash), and several serving settings change extraction output without
moving any of the three: the context size, the reasoning effort, the sampling
fields each request carries, the llama.cpp build and image, and where the
model's layers and experts run. Two hosts that differ in any of them write
different graphs under one key, and nothing detects it (MIS-196). The serving
identity covers them, so an epoch can be keyed on its full setup: the triple,
this identity (computed by the extraction service), and a dispatch identity
(computed by the backend).

The rule for a setting is that it is keyed unless it provably cannot change the
bytes the model returns. A needless key costs one cutover when that setting
changes; a missed one mixes outputs under one epoch.

What it covers, and why some settings are absent:

- The request profile: the sampling and decoding fields each stage's request
  carries, recorded as sent, after the adapter has prepared it. A setting an
  adapter ignores (a reasoning effort under qwen or gemma) therefore does not
  split epochs.
- The declared llama-server settings, `EXTRACTION_SERVER_<NAME>` on the
  service, which compose builds from the same `${...}` expression as the
  llama-server flag. Flags left at their default are covered by the image
  digest, because a default changes only with the build.
- `n_ctx` as llama-server's `/props` reports it on the call, cross-checked by
  the service against the declared context size.
- Code is not covered. A code change is governed by `EXTRACTION_VERSION` and
  `CODE_REVISIONS` (`backend.knowledge.version_stamps`), which choose between
  re-extracting history and new turns only; a key here would override that
  choice. The run identity (`backend.knowledge.extraction.run_identity`)
  records code and settings together; it stays a record, never a key.

This module is a leaf: standard library only. The extraction service imports
it, and so does the backend for `diff_components`, and the service image must
not pull the backend's dependencies in with it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

SERVING_IDENTITY_PREFIX = "si1-"

# The stages that send a request to the model, in the order a job runs them.
# Every `stage=` literal the engine passes to `adapter.prepare` is one of these
# (pinned by an AST test in tests/unit/extraction_service).
REQUEST_STAGES: tuple[str, ...] = ("scope", "extract", "derive")

# The declared llama-server settings without which the service reports no
# serving identity: the digest-pinned image (the build and every default flag),
# the device (the GPU, so two hosts never share an epoch), and the context size
# (cross-checked against the observed `n_ctx`).
REQUIRED_SERVER_KEYS: tuple[str, ...] = ("image", "device", "ctx_size")

_MISSING = "<missing>"


@dataclass(frozen=True, slots=True)
class StageRequestProfile:
    """The output-changing fields of one stage's request, as the engine sends it.

    Attributes:
        stage: One of `REQUEST_STAGES`.
        temperature: The sampling temperature.
        top_p: The nucleus sampling cutoff.
        constrained: How the reply is constrained: "schema" (a JSON schema),
            "json_object", or "none". When a request carries both a schema and
            JSON mode, the schema wins, as it does in the request sent.
        thinking_effort: The reasoning-effort hint, or None when not sent.
        thinking_enabled: The thinking switch, or None when not sent.
        thinking_budget_tokens: The reasoning token budget, or None when not
            sent (unbudgeted).
    """

    stage: str
    temperature: float
    top_p: float
    constrained: str
    thinking_effort: str | None
    thinking_enabled: bool | None
    thinking_budget_tokens: int | None


@dataclass(frozen=True, slots=True)
class ServingIdentityComponents:
    """Everything the serving identity hashes.

    Attributes:
        adapter: The model-family adapter, which selects the thinking hints and
            how the final text is parsed out of a reply.
        model_file: The weights file llama-server loads. `model_hash` is a free
            string, so the file name is the only recorded link to the weights.
        llama_cpp_build: The llama.cpp build, kept for continuity with
            `ResultStamps`; the declared image is the authoritative build.
        max_attempts: Stage 2 attempts per batch: a repair retry can turn an
            unparsable output into an extraction.
        scope_enabled: Whether Stage 1.5 runs; when off every scope is
            "unknown", and Stage 2 reads the scope.
        requests: One profile per stage in `REQUEST_STAGES`.
        server: The declared `EXTRACTION_SERVER_*` settings: lower-cased key,
            stripped value, sorted by key.
        n_ctx: llama-server's context size, observed on this call.
    """

    adapter: str
    model_file: str
    llama_cpp_build: str
    max_attempts: int
    scope_enabled: bool
    requests: tuple[StageRequestProfile, ...]
    server: tuple[tuple[str, str], ...]
    n_ctx: int


# Derived from the dataclass, so a field added to the profile is hashed too.
_PROFILE_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(StageRequestProfile) if f.name != "stage"
)
_NESTED_FIELDS = frozenset({"requests", "server"})


def components_dict(c: ServingIdentityComponents) -> dict[str, str | int | float | bool | None]:
    """The components as one flat mapping of JSON scalars, the form stored and compared.

    The scalar fields keep their names; each profile field becomes
    `request.<stage>.<field>` and each declared setting `server.<key>`.

    Raises:
        ValueError: Two profiles name one stage, or one server key is declared
            twice: either would let one value hide the other.
    """
    flat: dict[str, str | int | float | bool | None] = {
        f.name: getattr(c, f.name)
        for f in dataclasses.fields(ServingIdentityComponents)
        if f.name not in _NESTED_FIELDS
    }
    seen_stages: set[str] = set()
    for profile in c.requests:
        if profile.stage in seen_stages:
            raise ValueError(f"two request profiles for stage {profile.stage!r}")
        seen_stages.add(profile.stage)
        for name in _PROFILE_FIELDS:
            flat[f"request.{profile.stage}.{name}"] = getattr(profile, name)
    seen_keys: set[str] = set()
    for key, value in c.server:
        if key in seen_keys:
            raise ValueError(f"server setting {key!r} declared twice")
        seen_keys.add(key)
        flat[f"server.{key}"] = value
    return flat


def compute_serving_identity(c: ServingIdentityComponents) -> str:
    """Return `"si1-"` and the sha256 of `components_dict(c)` as canonical JSON."""
    canonical = json.dumps(components_dict(c), sort_keys=True, separators=(",", ":"))
    return SERVING_IDENTITY_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diff_components(expected: Mapping[str, object], actual: Mapping[str, object]) -> list[str]:
    """One `"name: <expected> != <actual>"` line per component that differs, sorted by name.

    Values are shown, and compared, as JSON, so `1`, `1.0` and `true` differ
    here as they do in the hash, although Python calls them equal. A key on
    one side only is shown as `<missing>` on the other.
    """
    lines: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        left = _shown(expected, name)
        right = _shown(actual, name)
        if left != right:
            lines.append(f"{name}: {left} != {right}")
    return lines


def _shown(values: Mapping[str, object], name: str) -> str:
    if name not in values:
        return _MISSING
    return json.dumps(values[name], sort_keys=True)


# Every `ServiceSettings` field that is not a serving identity component, with
# the reason it cannot change the bytes the model returns. A completeness test
# in tests/unit/extraction_service fails on a field that is neither a
# component source nor listed here. The raw request settings (reasoning effort
# and budget, constrained mode, temperature) are not listed: they are
# components through the request profile, as sent.
SERVICE_SETTINGS_EXCLUDED: Mapping[str, str] = MappingProxyType(
    {
        "model_hash": "already in the epoch's ledger triple",
        "llm_base_url": "an address: where llama-server is reached, not what it runs",
        "port": "an address: where the service listens",
        "location_label": (
            "a display label with a default, naming a place rather than hardware; the "
            "device is keyed separately as a required declaration"
        ),
        "idempotency_cache_size": (
            "replays a stored reply for a repeated job_id and changes no output"
        ),
        "llm_timeout_seconds": (
            "a deadline: whether a call times out depends on the host's load at that "
            "moment, so one value gives both outcomes and a key cannot separate them"
        ),
    }
)
