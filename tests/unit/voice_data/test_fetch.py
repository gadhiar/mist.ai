"""Weight fetching through `fetch --family F` (T11).

`huggingface_hub` is a fake module put into `sys.modules`, and the URL opener a fake object, so no
test touches the network. The command's job runs the stage in this process (as `test_clean.py`
does), with `HF_HOME` and `TORCH_HOME` pointing into the test root's cache, and the tests read what
it wrote: the `docker run` argv, `job.json`, the manifest, the cache and the console output.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import sys
import types
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from scripts.voice_data import cli, jobs
from scripts.voice_data import root as root_mod
from scripts.voice_data.commands import fetch as fetch_cmd
from scripts.voice_data.errors import JobError
from scripts.voice_data.stages import fetch as fetch_stage

from .conftest import Canaries, _assert_no_canary

IMAGE = "mist-voice-fam:0123456789ab"
REV_OPEN = "1" * 40
REV_GATED = "2" * 40
PAYLOAD = b"public checkpoint bytes\n" * 64
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
CANARY = Canaries()


def families(**changes: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """A small `families` map: an open repo, a gated repo with `files`, and a url checkpoint."""
    models: dict[str, Mapping[str, object]] = {
        "open-model": {
            "kind": "hf",
            "repository": "org/open-model",
            "revision": REV_OPEN,
            "licence": "mit",
            "gated": False,
        },
        "gated-model": {
            "kind": "hf",
            "repository": "org/gated-model",
            "revision": REV_GATED,
            "licence": "cc-by-4.0",
            "gated": True,
            "files": ["weights.ckpt", "config.yaml"],
        },
        "align": {
            "kind": "url",
            "url": "https://download.example.org/models/align.pth",
            "sha256": PAYLOAD_SHA,
            "dest": "torch/hub/checkpoints/align.pth",
            "licence": "unverified",
            "gated": False,
        },
    }
    models.update(changes)
    return {
        "fam": {
            "image": "mist-voice-fam",
            "venv": "/opt/venv-fam",
            "devices": ["cpu"],
            "vram_gb": 1,
            "models": {k: dict(v) for k, v in models.items()},
        },
        "hosted-fam": {
            "image": None,
            "venv": None,
            "devices": [],
            "vram_gb": 0,
            "hosted": True,
            "models": {
                "voice-api": {
                    "kind": "hosted",
                    "repository": None,
                    "revision": None,
                    "licence": "provider terms",
                    "gated": False,
                }
            },
        },
    }


# Fakes.


class LocalEntryNotFoundError(FileNotFoundError):
    """The fake hub's "not in the cache" error; the real one is also a `FileNotFoundError`."""


@dataclass
class FakeHub:
    """A fake `huggingface_hub`: snapshots live under `$HF_HOME/hub` in the real layout."""

    calls: list[dict[str, object]] = field(default_factory=list)
    refuse: set[str] = field(default_factory=set)

    def snapshot_dir(self, repo_id: str, revision: str, repo_type: str = "model") -> Path:
        hub = Path(os.environ["HF_HOME"]) / "hub"
        return hub / f"{repo_type}s--{repo_id.replace('/', '--')}" / "snapshots" / revision

    def put(self, repo_id: str, revision: str, names: Sequence[str]) -> Path:
        folder = self.snapshot_dir(repo_id, revision)
        for name in names:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{repo_id}:{name}".encode())
        return folder

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
        local_files_only: bool = False,
        allow_patterns: list[str] | str | None = None,
        **kwargs: object,
    ) -> str:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "repo_type": repo_type,
                "local_files_only": local_files_only,
                "allow_patterns": allow_patterns,
                **kwargs,
            }
        )
        assert revision is not None
        folder = self.snapshot_dir(repo_id, revision, repo_type or "model")
        if local_files_only:
            if not folder.is_dir():
                raise LocalEntryNotFoundError("not cached")
            return str(folder)
        if repo_id in self.refuse:
            raise OSError("401 gated")
        names = list(allow_patterns or ["config.json", "model.safetensors"])
        self.put(repo_id, revision, names)
        return str(folder)

    def online(self) -> list[dict[str, object]]:
        return [c for c in self.calls if not c["local_files_only"]]

    def module(self) -> types.ModuleType:
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = self.snapshot_download  # type: ignore[attr-defined]
        return module


class FakeResponse(io.BytesIO):
    """A response body; `BytesIO` is already a context manager with `read(size)`."""


@dataclass
class FakeOpener:
    """Stands in for `urllib.request.OpenerDirector`: serves fixed bytes and records each URL."""

    body: bytes = PAYLOAD
    urls: list[str] = field(default_factory=list)
    error: OSError | None = None

    def open(self, url: str, timeout: float | None = None) -> FakeResponse:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.body)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, private_root: root_mod.Root) -> FakeHub:
    """Install the fake hub and point `HF_HOME` and `TORCH_HOME` into the root's cache."""
    fake = FakeHub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake.module())
    monkeypatch.setenv("HF_HOME", str(private_root.path / "cache" / "hf"))
    monkeypatch.setenv("TORCH_HOME", str(private_root.path / "cache" / "torch"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return fake


@pytest.fixture
def fams(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    """Serve the small families map to the command, the executor and the stage."""
    data = families()
    monkeypatch.setattr(jobs, "load_families", lambda path=None: data)
    return data


class StageDocker:
    """A `DockerOps` whose `run_logged` runs the stage module in this process."""

    def __init__(self, root: root_mod.Root, opener: FakeOpener) -> None:
        self.root = root
        self.opener = opener
        self.argvs: list[list[str]] = []

    def capture(self, argv: Sequence[str]) -> str:
        raise AssertionError("a fetch job runs no GPU preflight")

    def check(self, argv: Sequence[str]) -> None:
        raise AssertionError("fetch builds nothing")

    def run_logged(self, argv: Sequence[str], log: Path) -> int:
        self.argvs.append(list(argv))
        at = list(argv).index("-m")
        module, rest = argv[at + 1], list(argv[at + 2 :])
        assert rest[:2] == ["--data", "/data"]
        rest[1] = str(self.root.path)
        stage = importlib.import_module(module)
        with (
            pytest.MonkeyPatch.context() as mp,
            log.open("w", encoding="utf-8") as handle,
            contextlib.redirect_stdout(handle),
            contextlib.redirect_stderr(handle),
        ):
            mp.setattr(stage, "default_opener", lambda: self.opener)
            return stage.main(rest)


@dataclass
class Ran:
    root: root_mod.Root
    code: int
    out: str
    err: str
    docker: StageDocker

    @property
    def argv(self) -> list[str]:
        assert len(self.docker.argvs) == 1
        return self.docker.argvs[0]

    def job_dir(self) -> Path:
        (folder,) = (self.root.path / "runs").iterdir()
        return folder

    def job(self) -> dict[str, object]:
        return json.loads((self.job_dir() / "job.json").read_text(encoding="utf-8"))

    def manifest(self) -> dict[str, object]:
        path = self.job_dir() / fetch_cmd.OUT_DIR / fetch_cmd.MANIFEST_NAME
        return json.loads(path.read_text(encoding="utf-8"))

    def log(self) -> str:
        return (self.job_dir() / "log.txt").read_text(encoding="utf-8")


def run_fetch(root: root_mod.Root, *args: str, opener: FakeOpener | None = None) -> Ran:
    docker = StageDocker(root, opener or FakeOpener())
    out, err = io.StringIO(), io.StringIO()
    with (
        pytest.MonkeyPatch.context() as mp,
        contextlib.redirect_stdout(out),
        contextlib.redirect_stderr(err),
    ):
        mp.setattr(fetch_cmd, "make_docker", lambda: docker)
        mp.setattr(fetch_cmd, "image_tag", lambda family: IMAGE)
        code = cli.main(["--root", str(root.path), "fetch", *args])
    return Ran(root, code, out.getvalue(), err.getvalue(), docker)


# The job the command builds.


def test_job_has_network_writable_cache_and_optional_token(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    argv = ran.argv
    assert "--network" not in argv
    assert "--gpus" not in argv
    cache = f"{private_root.path / 'cache'}:/cache"
    assert argv[argv.index(cache) - 1] == "-v"
    assert "HF_HUB_OFFLINE=1" not in argv
    assert argv[argv.index("HF_TOKEN") - 1] == "-e"
    assert "HF_HOME=/cache/hf" in argv and "TORCH_HOME=/cache/torch" in argv
    tail = argv[argv.index(IMAGE) :]
    assert tail[:4] == [IMAGE, "/opt/venv-fam/bin/python", "-I", "-m"]
    assert tail[4] == fetch_cmd.MODULE == jobs.FETCH_MODULE
    assert tail[5:9] == ["--data", "/data", "--family", "fam"]
    job = ran.job()
    assert job["network"] is True and job["gpu"] is False
    assert job["optional_env_names"] == ["HF_TOKEN"] and job["env_names"] == []
    assert job["inputs"] == []


def test_token_is_never_in_argv_output_manifest_or_log(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, canaries: Canaries
) -> None:
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    _assert_no_canary(" ".join(ran.argv))
    _assert_no_canary(ran.out + ran.err)
    _assert_no_canary(json.dumps(ran.manifest()))
    _assert_no_canary(ran.log())
    _assert_no_canary((ran.job_dir() / "job.json").read_text(encoding="utf-8"))
    assert all("token" not in call for call in hub.calls)


def test_unset_token_omits_the_env_flag(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fams["fam"]["models"].pop("gated-model")
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    assert "HF_TOKEN" not in ran.argv


# Hugging Face models.


def test_revision_repository_and_files_are_passed(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    online = {c["repo_id"]: c for c in hub.online()}
    assert online["org/open-model"]["revision"] == REV_OPEN
    assert online["org/open-model"]["allow_patterns"] is None
    assert online["org/open-model"]["repo_type"] == "model"
    assert online["org/gated-model"]["revision"] == REV_GATED
    assert online["org/gated-model"]["allow_patterns"] == ["weights.ckpt", "config.yaml"]
    snapshot = hub.snapshot_dir("org/gated-model", REV_GATED)
    assert (snapshot / "weights.ckpt").is_file()


def test_unset_token_fails_only_the_gated_model(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 4
    assert [c["repo_id"] for c in hub.online()] == ["org/open-model"]
    assert hub.snapshot_dir("org/open-model", REV_OPEN).is_dir()
    assert (private_root.path / "cache" / "torch/hub/checkpoints/align.pth").is_file()
    log = ran.log()
    assert "gated-model: failed, gated and HF_TOKEN is unset" in log
    assert "open-model: fetched" in log and "align: fetched" in log


def test_gated_model_already_cached_needs_no_token(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hub.put("org/gated-model", REV_GATED, ["weights.ckpt", "config.yaml"])
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    assert "org/gated-model" not in [c["repo_id"] for c in hub.online()]
    assert ran.manifest()["models"]["gated-model"]["status"] == "present"
    assert "  gated-model: present, 2 files" in ran.out


def test_cached_snapshot_is_reported_present_without_a_download(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    first = run_fetch(private_root, "--family", "fam")
    assert first.code == 0, first.err
    hub.calls.clear()
    opener = FakeOpener()
    second = run_fetch(private_root, "--family", "fam", opener=opener)
    assert second.code == 0, second.err
    assert hub.online() == []
    assert opener.urls == []
    local = [c for c in hub.calls if c["local_files_only"]]
    assert {(c["repo_id"], c["revision"]) for c in local} == {
        ("org/open-model", REV_OPEN),
        ("org/gated-model", REV_GATED),
    }
    assert "fetch fam: 3 models, 0 fetched, 3 present, 0 skipped" in second.out


def test_partial_snapshot_missing_a_listed_file_is_fetched_again(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    hub.put("org/gated-model", REV_GATED, ["config.yaml"])
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    assert "org/gated-model" in [c["repo_id"] for c in hub.online()]
    assert ran.manifest()["models"]["gated-model"]["status"] == "fetched"


def test_manifest_lists_every_file_with_its_sha256(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    manifest = ran.manifest()
    assert manifest["schema"] == 1 and manifest["family"] == "fam"
    cache = private_root.path / "cache"
    entries = [f for m in manifest["models"].values() for f in m["files"]]
    assert len(entries) == 5
    for entry in entries:
        path = cache / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["bytes"] == path.stat().st_size
    gated = manifest["models"]["gated-model"]
    assert gated["repository"] == "org/gated-model" and gated["revision"] == REV_GATED
    paths = [f["path"] for f in gated["files"]]
    assert paths == sorted(paths)
    assert all(p.startswith("hf/hub/models--org--gated-model/snapshots/") for p in paths)


def test_unpinned_revision_is_refused_before_any_job(
    private_root: root_mod.Root, hub: FakeHub, fams: dict
) -> None:
    fams["fam"]["models"]["open-model"]["revision"] = "main"
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 2
    assert "open-model has no pinned revision" in ran.err
    assert ran.docker.argvs == []


def test_stage_refuses_an_unpinned_revision(hub: FakeHub) -> None:
    data = families()
    data["fam"]["models"]["open-model"]["revision"] = None
    with pytest.raises(JobError, match="open-model has no pinned revision"):
        fetch_stage.fetch_family("fam", data, opener=FakeOpener())
    assert hub.calls == []


def test_repo_type_is_passed(private_root: root_mod.Root, hub: FakeHub) -> None:
    data = families()
    data["fam"]["models"] = {
        "dataset-model": {
            "kind": "hf",
            "repository": "org/data",
            "revision": REV_OPEN,
            "licence": "mit",
            "gated": False,
            "repo_type": "dataset",
            "files": ["a/b.ckpt"],
        }
    }
    fetch_stage.fetch_family("fam", data, opener=FakeOpener())
    (call,) = hub.online()
    assert call["repo_type"] == "dataset" and call["allow_patterns"] == ["a/b.ckpt"]


def test_hub_errors_become_job_errors(hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    hub.refuse.add("org/gated-model")
    with pytest.raises(JobError, match="gated-model"):
        fetch_stage.fetch_family("fam", families(), opener=FakeOpener())


# URL models.


def test_url_model_is_written_once_its_sha256_matches(
    private_root: root_mod.Root, hub: FakeHub
) -> None:
    data = families()
    data["fam"]["models"] = {"align": data["fam"]["models"]["align"]}
    opener = FakeOpener()
    manifest = fetch_stage.fetch_family("fam", data, opener=opener)
    dest = private_root.path / "cache" / "torch/hub/checkpoints/align.pth"
    assert dest.read_bytes() == PAYLOAD
    assert opener.urls == ["https://download.example.org/models/align.pth"]
    entry = manifest["models"]["align"]
    assert entry["status"] == "fetched" and entry["sha256"] == PAYLOAD_SHA
    assert entry["sha256_unverified"] is False
    assert list(dest.parent.iterdir()) == [dest]


def test_url_model_with_wrong_sha256_leaves_no_file(
    private_root: root_mod.Root, hub: FakeHub
) -> None:
    data = families()
    data["fam"]["models"] = {"align": data["fam"]["models"]["align"]}
    with pytest.raises(JobError, match="align: failed, sha256 does not match"):
        fetch_stage.fetch_family("fam", data, opener=FakeOpener(body=b"tampered"))
    folder = private_root.path / "cache" / "torch/hub/checkpoints"
    assert not folder.exists() or list(folder.iterdir()) == []


def test_cached_url_file_with_wrong_sha256_is_replaced(
    private_root: root_mod.Root, hub: FakeHub
) -> None:
    data = families()
    data["fam"]["models"] = {"align": data["fam"]["models"]["align"]}
    dest = private_root.path / "cache" / "torch/hub/checkpoints/align.pth"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stale")
    with pytest.raises(JobError, match="sha256 does not match"):
        fetch_stage.fetch_family("fam", data, opener=FakeOpener(body=b"tampered"))
    assert not dest.exists()
    manifest = fetch_stage.fetch_family("fam", data, opener=FakeOpener())
    assert dest.read_bytes() == PAYLOAD
    assert manifest["models"]["align"]["status"] == "fetched"


def test_unverified_url_model_is_hashed_recorded_and_printed(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_value_never_printed")
    fams["fam"]["models"]["align"]["sha256"] = "unverified"
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 0, ran.err
    entry = ran.manifest()["models"]["align"]
    assert entry["sha256"] == PAYLOAD_SHA and entry["sha256_unverified"] is True
    assert f"align: sha256 {PAYLOAD_SHA} (unverified in families.json)" in ran.out
    assert f"align: sha256 {PAYLOAD_SHA}" in ran.log()


def test_non_https_url_is_refused_before_opening(hub: FakeHub) -> None:
    data = families()
    data["fam"]["models"] = {"align": dict(data["fam"]["models"]["align"])}
    data["fam"]["models"]["align"]["url"] = "http://download.example.org/models/align.pth"
    opener = FakeOpener()
    with pytest.raises(JobError, match="model align is not an HTTPS url"):
        fetch_stage.fetch_family("fam", data, opener=opener)
    assert opener.urls == []


def test_fetch_url_refuses_a_non_https_url_itself(private_root: root_mod.Root) -> None:
    model = dict(families()["fam"]["models"]["align"])
    model["url"] = "http://download.example.org/models/align.pth"
    opener = FakeOpener()
    with pytest.raises(JobError, match="refused a non-HTTPS url"):
        fetch_stage.fetch_url("align", model, private_root.path / "cache", opener)
    assert opener.urls == []


def test_non_https_url_is_refused_by_the_command(
    private_root: root_mod.Root, hub: FakeHub, fams: dict
) -> None:
    fams["fam"]["models"]["align"]["url"] = "ftp://download.example.org/align.pth"
    ran = run_fetch(private_root, "--family", "fam")
    assert ran.code == 2
    assert "align is not an HTTPS url" in ran.err
    assert ran.docker.argvs == []


def test_redirect_off_https_is_refused() -> None:
    handler = fetch_stage.HttpsOnlyRedirects()
    request = urllib.request.Request("https://download.example.org/a.pth")
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(request, None, 302, "Found", {}, "http://mirror.example.org/a")
    again = handler.redirect_request(
        request, None, 302, "Found", {}, "https://mirror.example.org/a"
    )
    assert again is not None and again.full_url == "https://mirror.example.org/a"


def test_default_opener_carries_the_https_only_redirect_handler() -> None:
    opener = fetch_stage.default_opener()
    assert any(isinstance(h, fetch_stage.HttpsOnlyRedirects) for h in opener.handlers)


def test_url_dest_outside_the_cache_is_refused(hub: FakeHub) -> None:
    data = families()
    data["fam"]["models"] = {"align": dict(data["fam"]["models"]["align"])}
    data["fam"]["models"]["align"]["dest"] = "../escape.pth"
    with pytest.raises(JobError, match="align"):
        fetch_stage.fetch_family("fam", data, opener=FakeOpener())


def test_download_error_becomes_a_job_error_and_leaves_no_file(
    private_root: root_mod.Root, hub: FakeHub
) -> None:
    data = families()
    data["fam"]["models"] = {"align": data["fam"]["models"]["align"]}
    opener = FakeOpener(error=urllib.error.URLError("unreachable"))
    with pytest.raises(JobError, match="align: failed, download failed"):
        fetch_stage.fetch_family("fam", data, opener=opener)
    folder = private_root.path / "cache" / "torch/hub/checkpoints"
    assert not folder.exists() or list(folder.iterdir()) == []


# Hosted models and families.


def test_hosted_model_is_skipped(hub: FakeHub) -> None:
    data = families()
    data["fam"]["models"] = {"voice-api": data["hosted-fam"]["models"]["voice-api"]}
    manifest = fetch_stage.fetch_family("fam", data, opener=FakeOpener())
    assert manifest["models"]["voice-api"] == {"kind": "hosted", "status": "skipped", "files": []}
    assert hub.calls == []


def test_hosted_and_unknown_families_are_refused(
    private_root: root_mod.Root, hub: FakeHub, fams: dict
) -> None:
    hosted = run_fetch(private_root, "--family", "hosted-fam")
    assert hosted.code == 2 and "hosted-fam is hosted; nothing to fetch" in hosted.err
    unknown = run_fetch(private_root, "--family", "nope")
    assert unknown.code == 2 and "unknown family nope" in unknown.err
    assert hosted.docker.argvs == unknown.docker.argvs == []


def test_remote_executor_is_named_through_executor_for(
    private_root: root_mod.Root, hub: FakeHub, fams: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "scripts.voice_data.remote", raising=False)
    ran = run_fetch(private_root, "--family", "fam", "--executor", "remote:pod-a")
    assert ran.code == 2 and "remote executor is not available" in ran.err


# The real families.json.


def test_every_real_family_passes_the_fetch_checks() -> None:
    real = jobs.load_families()
    fetchable = [k for k, f in real.items() if f.get("image") and not f.get("hosted")]
    assert "extract" in fetchable and "score" in fetchable
    for key in fetchable:
        assert fetch_cmd.problems(real[key]["models"]) == [], key


def test_extract_models_fetch_only_the_files_the_bindings_load() -> None:
    models = jobs.load_families()["extract"]["models"]
    assert models["enhancer-sidon"]["files"] == ["feature_extractor_cuda.pt", "decoder_cuda.pt"]
    assert models["separator-melband-vocals"]["files"] == [
        "config_vocals_becruily.yaml",
        "mel_band_roformer_vocals_becruily.ckpt",
    ]
    assert models["dereverb-melband"]["files"] == [
        "dereverb_mel_band_roformer_anvuew.yaml",
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    ]


# The offline resolver the bindings use.


def test_snapshot_dir_resolves_the_pinned_revision_offline(hub: FakeHub) -> None:
    hub.put("org/open-model", REV_OPEN, ["config.json"])
    path = fetch_stage.snapshot_dir("org/open-model", REV_OPEN)
    assert path == hub.snapshot_dir("org/open-model", REV_OPEN)
    (call,) = hub.calls
    assert call["local_files_only"] is True and call["revision"] == REV_OPEN


def test_snapshot_dir_not_cached_is_a_job_error(hub: FakeHub) -> None:
    with pytest.raises(JobError, match="org/open-model is not in the cache; run fetch"):
        fetch_stage.snapshot_dir("org/open-model", REV_OPEN)


def test_snapshot_dir_refuses_an_unpinned_revision(hub: FakeHub) -> None:
    with pytest.raises(JobError, match="no pinned revision"):
        fetch_stage.snapshot_dir("org/open-model", "main")
    assert hub.calls == []


@contextlib.contextmanager
def _no_hub() -> Iterator[None]:
    saved = sys.modules.pop("huggingface_hub", None)
    sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        del sys.modules["huggingface_hub"]
        if saved is not None:
            sys.modules["huggingface_hub"] = saved


def test_stage_imports_the_hub_lazily() -> None:
    with _no_hub():
        module = importlib.reload(fetch_stage)
        assert module.snapshot_dir is not None
    importlib.reload(fetch_stage)
