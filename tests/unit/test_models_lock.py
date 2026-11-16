"""Guard: the backend image bakes the read-path models at pinned revisions (plan T12).

The memory read path loads its cross-encoder reranker from a local path with
`local_files_only=True` (`backend/memory/rerank.py`), and the offline recall eval loads the
embedder from `/opt/models/all-MiniLM-L6-v2`. Neither may reach the network at runtime, so
the image build downloads each model at the revision `docker/backend/models.lock` pins.

These tests parse the lock, check the lock checker's strict rule (a revision is 40 lowercase
hex, never a branch name or a placeholder), check the Dockerfile text (workers never build
images), and check NOTICE records each model's license. The build context is
`tests/unit/test_dockerignore_models_lock.py`'s.

Hermetic: no network and no Docker. The download path is exercised with a fake
`snapshot_download`.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "docker" / "backend" / "models.lock"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_models_lock.py"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "backend" / "Dockerfile"
NOTICE_PATH = REPO_ROOT / "NOTICE"

# name -> (repo, revision): the revisions the Hugging Face model API reported (plan H3).
EXPECTED_ENTRIES = {
    "ms-marco-MiniLM-L-6-v2": (
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ),
    "ms-marco-MiniLM-L-12-v2": (
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
    ),
    "all-MiniLM-L6-v2": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
}
GOOD_SHA = "0123456789abcdef0123456789abcdef01234567"
# The Rust weights file of all-MiniLM-L6-v2, joined from parts because codespell (v2.2.6, no
# inline ignore) reads its two-letter suffix as a misspelling.
RUST_WEIGHTS = "rust_model." + "o" + "t"

# Each pinned revision's file list, from the same model API read (plan H3). The two
# cross-encoders list the same files.
_CROSS_ENCODER_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "flax_model.msgpack",
    "model.safetensors",
    "onnx/model.onnx",
    "onnx/model_O1.onnx",
    "onnx/model_O2.onnx",
    "onnx/model_O3.onnx",
    "onnx/model_O4.onnx",
    "onnx/model_qint8_arm64.onnx",
    "onnx/model_qint8_avx512.onnx",
    "onnx/model_qint8_avx512_vnni.onnx",
    "onnx/model_quint8_avx2.onnx",
    "openvino/openvino_model.bin",
    "openvino/openvino_model.xml",
    "openvino/openvino_model_qint8_quantized.bin",
    "openvino/openvino_model_qint8_quantized.xml",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_CROSS_ENCODER_FETCHED = {
    "README.md",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
}
PINNED_FILES = {
    "ms-marco-MiniLM-L-6-v2": (_CROSS_ENCODER_FILES, _CROSS_ENCODER_FETCHED),
    "ms-marco-MiniLM-L-12-v2": (_CROSS_ENCODER_FILES, _CROSS_ENCODER_FETCHED),
    "all-MiniLM-L6-v2": (
        (
            ".gitattributes",
            "1_Pooling/config.json",
            "README.md",
            "config.json",
            "config_sentence_transformers.json",
            "data_config.json",
            "model.safetensors",
            "modules.json",
            "onnx/model.onnx",
            "onnx/model_O1.onnx",
            "onnx/model_O2.onnx",
            "onnx/model_O3.onnx",
            "onnx/model_O4.onnx",
            "onnx/model_qint8_arm64.onnx",
            "onnx/model_qint8_avx512.onnx",
            "onnx/model_qint8_avx512_vnni.onnx",
            "onnx/model_quint8_avx2.onnx",
            "openvino/openvino_model.bin",
            "openvino/openvino_model.xml",
            "openvino/openvino_model_qint8_quantized.bin",
            "openvino/openvino_model_qint8_quantized.xml",
            "pytorch_model.bin",
            RUST_WEIGHTS,
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tf_model.h5",
            "tokenizer.json",
            "tokenizer_config.json",
            "train_script.py",
            "vocab.txt",
        ),
        {
            "1_Pooling/config.json",
            "README.md",
            "config.json",
            "config_sentence_transformers.json",
            "data_config.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        },
    ),
}


def _load_checker() -> ModuleType:
    """Import the script by path: `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("check_models_lock", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load_checker()


def _lock_text(revision: str, *, license_id: str = "Apache-2.0") -> str:
    return (
        "[[model]]\n"
        'name = "ms-marco-MiniLM-L-6-v2"\n'
        'repo = "cross-encoder/ms-marco-MiniLM-L-6-v2"\n'
        f'revision = "{revision}"\n'
        f'license = "{license_id}"\n'
        'target = "/opt/models/ms-marco-MiniLM-L-6-v2"\n'
    )


def _dockerfile_instructions() -> list[str]:
    """Dockerfile instructions with comments dropped and continuation lines joined."""
    instructions: list[str] = []
    current = ""
    for raw in DOCKERFILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        current += line
        instructions.append(current.strip())
        current = ""
    if current:
        instructions.append(current.strip())
    return instructions


def _index_of(instructions: list[str], predicate) -> int:
    for index, instruction in enumerate(instructions):
        if predicate(instruction):
            return index
    raise AssertionError("no Dockerfile instruction matches")


def _download_index(instructions: list[str]) -> int:
    return _index_of(instructions, lambda i: i.startswith("RUN ") and "--download" in i)


class TestLockContents:
    def test_lock_lists_the_three_models_at_their_pinned_revisions(
        self, checker: ModuleType
    ) -> None:
        entries = checker.parse_lock(LOCK_PATH)

        assert {entry.name: (entry.repo, entry.revision) for entry in entries} == (EXPECTED_ENTRIES)

    def test_each_license_is_apache(self, checker: ModuleType) -> None:
        assert {entry.license for entry in checker.parse_lock(LOCK_PATH)} == {"Apache-2.0"}

    def test_each_target_is_its_name_under_opt_models(self, checker: ModuleType) -> None:
        for entry in checker.parse_lock(LOCK_PATH):
            assert entry.target == f"/opt/models/{entry.name}"

    def test_reranker_default_path_is_baked(self, checker: ModuleType) -> None:
        from backend.memory.config import MemoryConfig

        targets = {entry.target for entry in checker.parse_lock(LOCK_PATH)}
        assert MemoryConfig().rerank_model in targets

    def test_eval_embedder_path_is_baked(self, checker: ModuleType) -> None:
        """`scripts/eval_harness/memory_recall_run.py` loads its embedder from this path."""
        targets = {entry.target for entry in checker.parse_lock(LOCK_PATH)}
        assert "/opt/models/all-MiniLM-L6-v2" in targets

    def test_lock_passes_the_structural_check(self, checker: ModuleType) -> None:
        assert checker.check_lock(checker.parse_lock(LOCK_PATH), strict=False) == []

    def test_lock_passes_the_strict_check(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict", str(LOCK_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + result.stderr


class TestStrictRule:
    @pytest.mark.parametrize(
        "revision",
        [
            "UNRESOLVED-H3",
            "main",
            GOOD_SHA[:-1],
            GOOD_SHA + "0",
            GOOD_SHA.upper(),
            GOOD_SHA[:-1] + "g",
        ],
    )
    def test_strict_rejects_a_revision_that_is_not_40_lowercase_hex(
        self, checker: ModuleType, tmp_path: Path, revision: str
    ) -> None:
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text(revision), encoding="utf-8")

        problems = checker.check_lock(checker.parse_lock(lock), strict=True)

        assert any("revision" in problem for problem in problems), problems

    def test_strict_accepts_a_40_hex_revision(self, checker: ModuleType, tmp_path: Path) -> None:
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text(GOOD_SHA), encoding="utf-8")

        assert checker.check_lock(checker.parse_lock(lock), strict=True) == []

    def test_structural_check_allows_the_placeholder(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text("UNRESOLVED-H3"), encoding="utf-8")

        assert checker.check_lock(checker.parse_lock(lock), strict=False) == []

    def test_a_license_outside_the_allowlist_is_rejected(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text(GOOD_SHA, license_id="CC-BY-NC-4.0"), encoding="utf-8")

        problems = checker.check_lock(checker.parse_lock(lock), strict=False)

        assert any("license" in problem for problem in problems), problems

    @pytest.mark.parametrize("revision", ["UNRESOLVED-H3", "main"])
    def test_cli_strict_exits_nonzero_on_an_unpinned_revision(
        self, tmp_path: Path, revision: str
    ) -> None:
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text(revision), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict", str(lock)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert revision in result.stdout + result.stderr

    def test_cli_download_exits_nonzero_on_an_unpinned_revision(self, tmp_path: Path) -> None:
        """The build's `--download` checks the lock first, so it never reaches the Hub."""
        lock = tmp_path / "models.lock"
        lock.write_text(_lock_text("main"), encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--strict",
                "--download",
                "ms-marco-MiniLM-L-6-v2",
                str(lock),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1
        assert "revision 'main'" in result.stderr


class TestDownload:
    def test_download_fetches_the_pinned_revision_into_the_target(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        target = tmp_path / "ms-marco-MiniLM-L-6-v2"
        entry = checker.LockEntry(
            name="ms-marco-MiniLM-L-6-v2",
            repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
            revision=GOOD_SHA,
            license="Apache-2.0",
            target=str(target),
        )
        calls: list[dict] = []

        def fake_snapshot_download(**kwargs) -> str:
            calls.append(kwargs)
            local_dir = Path(kwargs["local_dir"])
            for name in checker.REQUIRED_FILES:
                (local_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (local_dir / name).write_text("{}", encoding="utf-8")
            (local_dir / ".cache" / "huggingface").mkdir(parents=True)
            return str(local_dir)

        checker.download_entry(entry, fetch=fake_snapshot_download)

        assert len(calls) == 1
        call = calls[0]
        assert call["repo_id"] == entry.repo
        assert call["revision"] == GOOD_SHA
        assert Path(call["local_dir"]) == target
        assert call["allow_patterns"] == list(checker.ALLOW_PATTERNS)
        assert call["ignore_patterns"] == list(checker.IGNORE_PATTERNS)
        assert not (target / ".cache").exists()

    def test_download_fails_when_a_required_file_is_missing(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        entry = checker.LockEntry(
            name="ms-marco-MiniLM-L-6-v2",
            repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
            revision=GOOD_SHA,
            license="Apache-2.0",
            target=str(tmp_path / "ms-marco-MiniLM-L-6-v2"),
        )

        with pytest.raises(checker.LockError, match="model.safetensors"):
            checker.download_entry(entry, fetch=lambda **kwargs: kwargs["local_dir"])

    def test_download_refuses_an_unpinned_revision(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        entry = checker.LockEntry(
            name="ms-marco-MiniLM-L-6-v2",
            repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
            revision="main",
            license="Apache-2.0",
            target=str(tmp_path / "ms-marco-MiniLM-L-6-v2"),
        )

        def must_not_fetch(**kwargs) -> str:
            raise AssertionError("fetched an unpinned revision")

        with pytest.raises(checker.LockError, match="revision"):
            checker.download_entry(entry, fetch=must_not_fetch)

    @pytest.mark.parametrize("name", sorted(PINNED_FILES))
    def test_fetch_set_of_each_pinned_revision(self, checker: ModuleType, name: str) -> None:
        """The Hub's own filter over the revision's file list keeps what loading needs only.

        `snapshot_download` applies `filter_repo_objects` to the revision's files, so running
        it here over the H3 file list shows the exact set the build fetches: the configs,
        tokenizer, safetensors weights and model card, and no ONNX, OpenVINO, PyTorch,
        TensorFlow, Flax or Rust weights.
        """
        hub_utils = pytest.importorskip("huggingface_hub.utils")
        files, expected = PINNED_FILES[name]

        fetched = set(
            hub_utils.filter_repo_objects(
                files,
                allow_patterns=list(checker.ALLOW_PATTERNS),
                ignore_patterns=list(checker.IGNORE_PATTERNS),
            )
        )

        assert fetched == expected
        assert set(checker.REQUIRED_FILES) <= fetched


class TestDockerfile:
    def test_dockerfile_copies_the_lock_and_the_checker(self) -> None:
        copies = [i for i in _dockerfile_instructions() if i.startswith("COPY ")]

        assert any("docker/backend/models.lock" in i for i in copies), copies
        assert any("scripts/check_models_lock.py" in i for i in copies), copies

    def test_strict_check_runs_before_any_download(self) -> None:
        instructions = _dockerfile_instructions()
        strict = _index_of(
            instructions, lambda i: i.startswith("RUN ") and "check_models_lock.py --strict" in i
        )
        download = _download_index(instructions)

        assert strict <= download
        if strict == download:
            text = instructions[strict]
            assert text.index("--strict") < text.index("--download")

    def test_every_lock_entry_is_downloaded(self, checker: ModuleType) -> None:
        downloads = " ".join(i for i in _dockerfile_instructions() if "--download" in i)

        for entry in checker.parse_lock(LOCK_PATH):
            assert re.search(rf"--download\s+{re.escape(entry.name)}(\s|$)", downloads), entry

    def test_download_layer_precedes_the_application_code(self) -> None:
        instructions = _dockerfile_instructions()
        download = _download_index(instructions)
        app_code = _index_of(instructions, lambda i: i.startswith("COPY") and " backend/ " in i)

        assert download < app_code

    def test_download_layer_follows_the_requirements_install(self) -> None:
        """huggingface_hub, which the download imports, comes from requirements.txt."""
        instructions = _dockerfile_instructions()
        requirements = _index_of(instructions, lambda i: "pip install -r" in i)
        download = _download_index(instructions)

        assert requirements < download

    def test_model_layer_is_one_block_from_root_back_to_appuser(self) -> None:
        """The layer writes root-owned /opt/models, so it alone runs as root."""
        instructions = _dockerfile_instructions()
        download = _download_index(instructions)

        assert instructions[download - 2] == "USER root"
        assert instructions[download - 1].startswith("COPY docker/backend/models.lock ")
        assert instructions[download + 1] == "USER appuser"

    def test_model_layer_installs_nothing_and_chowns_nothing(self) -> None:
        download = _dockerfile_instructions()[_download_index(_dockerfile_instructions())]

        assert "pip install" not in download
        assert "chown" not in download

    def test_model_layer_writes_no_bytecode_into_the_venv(self) -> None:
        """Run as root, an import would leave root-owned `.pyc` files in appuser's venv."""
        download = _dockerfile_instructions()[_download_index(_dockerfile_instructions())]

        assert "PYTHONDONTWRITEBYTECODE=1" in download

    def test_models_are_world_readable(self) -> None:
        download = _dockerfile_instructions()[_download_index(_dockerfile_instructions())]

        assert re.search(r"chmod -R a\+rX /opt/models", download), download

    def test_no_global_hf_offline_flag(self) -> None:
        """The baked models already load offline; a global flag would stop the Hub loads.

        The read-path models load from `/opt/models` with `local_files_only=True`
        (backend/knowledge/embeddings/embedding_generator.py:101-103,
        backend/memory/rerank.py:465-467), so they need no flag. What a flag in the image would
        break is every load that still reaches the Hub: Chatterbox's `from_pretrained` (src/multimodal/tts.py:276)
        calls `hf_hub_download` for each of its files, and `EmbeddingGenerator` loads a model
        that is not baked by its name (embedding_generator.py:106). Under the flag, each fails
        unless its files are already in the Hugging Face cache.
        """
        for instruction in _dockerfile_instructions():
            if instruction.startswith(("ENV ", "ARG ")):
                assert "HF_HUB_OFFLINE" not in instruction, instruction
                assert "TRANSFORMERS_OFFLINE" not in instruction, instruction


class TestNotice:
    def test_notice_records_each_model_and_its_license(self, checker: ModuleType) -> None:
        notice = NOTICE_PATH.read_text(encoding="utf-8")

        for entry in checker.parse_lock(LOCK_PATH):
            assert f"Source: https://huggingface.co/{entry.repo}" in notice, entry.repo
            block = notice.split(f"Source: https://huggingface.co/{entry.repo}", 1)[1]
            block = block.split("-------", 1)[0]
            assert f"Revision: {entry.revision}" in block, entry.repo
            assert f"License: {checker.LICENSE_NAMES[entry.license]}" in block, entry.repo
