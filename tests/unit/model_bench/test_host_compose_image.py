"""Compose image-ref parsing: successful digest extraction and the refusal
on a missing digest, plus resolve_image_ref()'s compose/snapshot/literal
dispatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import (  # noqa: E402
    ImageRefError,
    parse_compose_image,
    resolve_image_ref,
)

FIXTURES = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "host"


def test_parse_compose_image_extracts_digest_pinned_ref():
    ref = parse_compose_image(FIXTURES / "compose_with_digest.yml")
    assert ref.startswith("ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:")
    digest = ref.split("@sha256:", 1)[1]
    assert len(digest) == 64
    assert all(c in "0123456789abcdefABCDEF" for c in digest)


def test_parse_compose_image_refuses_missing_digest():
    with pytest.raises(ImageRefError):
        parse_compose_image(FIXTURES / "compose_missing_digest.yml")


def test_parse_compose_image_refuses_unknown_service():
    with pytest.raises(ImageRefError):
        parse_compose_image(FIXTURES / "compose_with_digest.yml", service="does-not-exist")


def test_resolve_image_ref_compose_token(tmp_path):
    ref = resolve_image_ref(
        "compose:mist-llm",
        compose_path=FIXTURES / "compose_with_digest.yml",
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert "@sha256:" in ref


def test_resolve_image_ref_compose_token_missing_digest_raises(tmp_path):
    with pytest.raises(ImageRefError):
        resolve_image_ref(
            "compose:mist-llm",
            compose_path=FIXTURES / "compose_missing_digest.yml",
            snapshot_path=tmp_path / "snapshot.json",
        )


def test_resolve_image_ref_snapshot_token(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"mist-llm": {"Image": "sha256:" + "b" * 64}}))
    ref = resolve_image_ref(
        "snapshot:mist-llm", compose_path=FIXTURES / "compose_with_digest.yml", snapshot_path=snapshot_path
    )
    assert ref == "sha256:" + "b" * 64


def test_resolve_image_ref_snapshot_token_missing_file_raises(tmp_path):
    with pytest.raises(ImageRefError):
        resolve_image_ref(
            "snapshot:mist-llm",
            compose_path=FIXTURES / "compose_with_digest.yml",
            snapshot_path=tmp_path / "does-not-exist.json",
        )


def test_resolve_image_ref_literal_token_passthrough(tmp_path):
    ref = resolve_image_ref(
        "docker.io/library/busybox@sha256:" + "c" * 64,
        compose_path=FIXTURES / "compose_with_digest.yml",
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert ref == "docker.io/library/busybox@sha256:" + "c" * 64
