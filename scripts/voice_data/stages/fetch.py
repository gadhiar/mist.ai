"""The weight cache: the `fetch` job that fills it, and the offline resolver that reads it.

The job runs as `<venv>/bin/python -I -u -m scripts.voice_data.stages.fetch --data /data
--family F --out runs/<job_id>/fetch`, with the cache mounted read-write at the parent of `HF_HOME` and
`TORCH_HOME` and the network on. Each model of the family is fetched by its `kind`:

- `hf`: `huggingface_hub.snapshot_download` at the pinned revision into `$HF_HOME/hub`, with the
  model's `files` as `allow_patterns`. A snapshot already in the cache at that revision, with
  every listed file present, is `present` and needs neither the network nor a token. A gated
  model that is not cached fails when `HF_TOKEN` is unset; the hub reads the token from the
  environment itself, so its value never passes through this code.
- `url`: one HTTPS file, streamed to a temporary name beside `dest` under the cache and renamed
  only once its sha256 matches `families.json`. An entry whose sha256 is `"unverified"` is hashed
  at its first download and that hash is recorded in the manifest. A mismatch leaves no file.
- `hosted`: skipped.

Every model is attempted, so one gated model without a token fails alone; the job then exits 1
naming each failure. On success it writes `manifest.json` (every file with its sha256 and size,
paths relative to the cache) and `outputs.json` for the executor.

Bindings resolve a model with `snapshot_dir`, never by repository id: a snapshot fetched at a
pinned commit writes no `refs/main`, so an offline lookup by repository id alone fails, while the
commit's `snapshots/<sha>` folder resolves with `local_files_only`.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import http.client
import json
import logging
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from .. import jobs
from ..commands import fetch as contract
from ..errors import InputError, JobError

SCHEMA = 1
CHUNK_BYTES = 1 << 20
TIMEOUT_S = 60.0
HUB_LOGGER = "huggingface_hub"
EXTRACT = "extract"
SCORE = "score"
DEVICES = ("cpu", "cuda")
UPSTREAM = Path("/opt/mist-voice/upstream")
QUIET_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
}


class HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to HTTPS, so a mirror cannot downgrade a download to plain HTTP."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Refuse a redirect to any scheme but HTTPS; otherwise redirect as urllib does."""
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, "refused a redirect off HTTPS", headers, fp  # type: ignore[arg-type]
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def default_opener() -> urllib.request.OpenerDirector:
    """Return the opener url models are downloaded with; tests replace this function."""
    return urllib.request.build_opener(HttpsOnlyRedirects())


def cache_root() -> Path:
    """Return the cache folder, the common parent of `HF_HOME` and `TORCH_HOME`.

    Raises:
        JobError: Either variable is unset, or they do not share a parent.
    """
    hf_home, torch_home = os.environ.get("HF_HOME"), os.environ.get("TORCH_HOME")
    if not hf_home or not torch_home:
        raise JobError("HF_HOME and TORCH_HOME must both be set")
    root = Path(hf_home).parent
    if Path(torch_home).parent != root:
        raise JobError("HF_HOME and TORCH_HOME must share one cache folder")
    return root


def snapshot_dir(repository: str, revision: str, *, repo_type: str = "model") -> Path:
    """Return the cached snapshot folder of a repository at its pinned revision, offline.

    Raises:
        JobError: The revision is not a commit sha, or the snapshot is not in the cache.
    """
    if not contract.pinned(revision):
        raise JobError(f"{repository} has no pinned revision")
    import huggingface_hub

    try:
        path = huggingface_hub.snapshot_download(
            repository, revision=revision, repo_type=repo_type, local_files_only=True
        )
    except (OSError, ValueError) as exc:
        raise JobError(f"{repository} is not in the cache; run fetch") from exc
    return Path(path)


# Reading the cache from a binding.


def model_entry(family: str, model_id: str) -> Mapping[str, object]:
    """Return a model's `families.json` entry.

    Raises:
        JobError: The family has no such model.
    """
    models = jobs.load_families().get(family, {}).get("models", {})
    entry = models.get(model_id) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        raise JobError(f"family {family} has no model {model_id}")
    return entry


def model_dir(family: str, model_id: str) -> Path:
    """Return a Hugging Face model's snapshot folder at its pinned revision, offline.

    Raises:
        JobError: The model is unknown, not a Hugging Face model, or not in the cache.
    """
    entry = model_entry(family, model_id)
    if entry.get("kind") != "hf":
        raise JobError(f"model {model_id} is not a Hugging Face model")
    return snapshot_dir(
        str(entry["repository"]),
        str(entry["revision"]),
        repo_type=str(entry.get("repo_type") or "model"),
    )


def model_file(family: str, model_id: str, name: str) -> Path:
    """Return one file of a model's cached snapshot.

    Raises:
        JobError: The snapshot or the file is not in the cache.
    """
    path = model_dir(family, model_id) / name
    if not path.is_file():
        raise JobError(f"model {model_id} has no {name} in the cache; run fetch")
    return path


def cache_file(family: str, model_id: str) -> Path:
    """Return a `url` model's file under the cache.

    Raises:
        JobError: The model is not a `url` model, or its file is not in the cache.
    """
    entry = model_entry(family, model_id)
    if entry.get("kind") != "url":
        raise JobError(f"model {model_id} is not a url model")
    path = cache_root() / str(entry["dest"])
    if not path.is_file():
        raise JobError(f"model {model_id} is not in the cache; run fetch")
    return path


def prepare(device: str, *loggers: str) -> None:
    """Set offline mode and errors-only output before a binding imports its library.

    Offline mode and the progress-bar switches are environment variables the libraries read
    when they are imported, so this runs first in every `load`. A library's log can carry a
    transcript, and its progress bars fill the job log, so both are kept to errors.

    Raises:
        JobError: `device` is neither `cpu` nor `cuda`.
    """
    if device not in DEVICES:
        raise JobError(f"device {device} is not one of {', '.join(DEVICES)}")
    os.environ.update(QUIET_ENV)
    for name in (HUB_LOGGER, *loggers):
        logging.getLogger(name).setLevel(logging.ERROR)


@contextlib.contextmanager
def library_errors(model_id: str) -> Iterator[None]:
    """Turn a model library's failure into `JobError`, naming the model id and error class only.

    Library messages can carry paths or text, so they stay in the chained traceback in the job
    log. Errors outside these classes are bugs in the binding and surface as themselves.
    """
    try:
        yield
    except JobError:
        raise
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        raise JobError(f"{model_id}: {type(exc).__name__}") from exc


def on_path(folder: Path) -> None:
    """Put an upstream clone at the front of `sys.path`, once; `-I` keeps it off otherwise."""
    entry = str(folder)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def sha256_file(path: Path) -> str:
    """Return a file's sha256, read in chunks so a large checkpoint never sits in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _patterns(model: Mapping[str, object]) -> list[str] | None:
    files = model.get("files")
    return [str(p) for p in files] if isinstance(files, list) and files else None


def _listing(folder: Path, patterns: Sequence[str] | None) -> list[Path]:
    # The hub's allow_patterns are fnmatch patterns over posix paths, where `*` also matches `/`.
    found = sorted(p for p in folder.rglob("*") if p.is_file())
    if patterns is None:
        return found
    return [
        p
        for p in found
        if any(fnmatch.fnmatchcase(p.relative_to(folder).as_posix(), pat) for pat in patterns)
    ]


def _complete(folder: Path, patterns: Sequence[str] | None) -> bool:
    files = _listing(folder, None)
    if not files:
        return False
    if patterns is None:
        return True
    rels = [p.relative_to(folder).as_posix() for p in files]
    return all(any(fnmatch.fnmatchcase(rel, pat) for rel in rels) for pat in patterns)


def _entries(paths: Sequence[Path], cache: Path) -> list[dict[str, object]]:
    entries = []
    for path in paths:
        try:
            rel = path.relative_to(cache).as_posix()
        except ValueError as exc:
            raise JobError("a fetched file lies outside the cache") from exc
        entries.append({"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return entries


def fetch_hf(model_id: str, model: Mapping[str, object], cache: Path) -> dict[str, object]:
    """Fetch one Hugging Face model at its pinned revision, or find it already cached.

    Raises:
        JobError: The model is gated and not cached while `HF_TOKEN` is unset, or the hub
            refused or failed the download.
    """
    repository, revision = str(model["repository"]), str(model["revision"])
    repo_type = str(model.get("repo_type") or "model")
    patterns = _patterns(model)
    status = "present"
    try:
        folder: Path | None = snapshot_dir(repository, revision, repo_type=repo_type)
    except JobError:
        folder = None
    if folder is None or not _complete(folder, patterns):
        if model.get("gated") and not os.environ.get(contract.TOKEN_ENV):
            raise JobError(f"gated and {contract.TOKEN_ENV} is unset")
        import huggingface_hub

        try:
            folder = Path(
                huggingface_hub.snapshot_download(
                    repository, revision=revision, repo_type=repo_type, allow_patterns=patterns
                )
            )
        except (OSError, ValueError) as exc:
            raise JobError(f"the hub refused or failed ({type(exc).__name__})") from exc
        status = "fetched"
    return {
        "kind": "hf",
        "status": status,
        "repository": repository,
        "revision": revision,
        "files": _entries(_listing(folder, patterns), cache),
    }


def _download(url: str, dest: Path, expected: str, opener: urllib.request.OpenerDirector) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as out, opener.open(url, timeout=TIMEOUT_S) as response:
                while chunk := response.read(CHUNK_BYTES):
                    digest.update(chunk)
                    out.write(chunk)
        except (OSError, http.client.HTTPException) as exc:
            raise JobError(f"download failed ({type(exc).__name__})") from exc
        actual = digest.hexdigest()
        if expected != contract.UNVERIFIED and actual != expected:
            raise JobError("sha256 does not match families.json")
        os.replace(temp, dest)
        return actual
    finally:
        if os.path.lexists(temp):
            os.unlink(temp)


def fetch_url(
    model_id: str,
    model: Mapping[str, object],
    cache: Path,
    opener: urllib.request.OpenerDirector,
) -> dict[str, object]:
    """Fetch one HTTPS file to `dest` under the cache, or find it there with a matching sha256.

    Raises:
        JobError: The URL is not HTTPS, `dest` leaves the cache, the download failed, or the
            file's sha256 does not match; no file is left at `dest` in the last two cases.
    """
    url = str(model["url"])
    if urllib.parse.urlparse(url).scheme != "https":
        raise JobError("refused a non-HTTPS url")
    dest = (cache / str(model["dest"])).resolve()
    if not dest.is_relative_to(cache.resolve()) or dest == cache.resolve():
        raise JobError("dest resolves outside the cache")
    expected = str(model.get("sha256") or contract.UNVERIFIED)
    status = "present"
    actual = sha256_file(dest) if dest.is_file() else None
    if actual is None or expected not in (contract.UNVERIFIED, actual):
        if actual is not None:
            dest.unlink()
        actual = _download(url, dest, expected, opener)
        status = "fetched"
    return {
        "kind": "url",
        "status": status,
        "url": url,
        "sha256": actual,
        "sha256_unverified": expected == contract.UNVERIFIED,
        "files": _entries([dest], cache.resolve()),
    }


def fetch_family(
    family: str,
    families: Mapping[str, Mapping[str, object]] | None = None,
    *,
    opener: urllib.request.OpenerDirector | None = None,
) -> dict[str, object]:
    """Fetch every model of `family` into the cache and return the manifest.

    Every model is attempted before any failure is raised, so the log names each one that
    failed and the rest are cached for the next run.

    Raises:
        JobError: The family fails the command's checks, or any model failed.
    """
    try:
        models = contract.family_models(
            jobs.load_families() if families is None else families, family
        )
    except InputError as exc:
        raise JobError(str(exc)) from exc
    logging.getLogger(HUB_LOGGER).setLevel(logging.ERROR)
    cache = cache_root()
    opener = default_opener() if opener is None else opener
    done: dict[str, object] = {}
    failures: list[str] = []
    for model_id, model in sorted(models.items()):
        kind = model["kind"]
        try:
            if kind == "hf":
                entry = fetch_hf(model_id, model, cache)
            elif kind == "url":
                entry = fetch_url(model_id, model, cache, opener)
            else:
                entry = {"kind": kind, "status": "skipped", "files": []}
        except JobError as exc:
            failures.append(f"{model_id}: failed, {exc}")
            print(f"  {failures[-1]}")
            continue
        done[model_id] = entry
        print(f"  {model_id}: {entry['status']}, {len(entry['files'])} files")  # type: ignore[arg-type]
        if entry.get("sha256_unverified"):
            print(f"  {model_id}: sha256 {entry['sha256']}")
    if failures:
        raise JobError(f"fetch {family}: {len(failures)} failed; {'; '.join(failures)}")
    return {"schema": SCHEMA, "family": family, "models": done}


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job; write the manifest and `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.fetch")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.is_absolute() or ".." in out.parts:
        print("error: --out is a folder relative to --data", file=sys.stderr)
        return 1
    try:
        manifest = fetch_family(args.family)
    except JobError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rel = f"{out.as_posix()}/{contract.MANIFEST_NAME}"
    _write_json(args.data / rel, manifest)
    files = [{"rel": rel, "sha256": sha256_file(args.data / rel)}]
    _write_json(args.data / out / "outputs.json", {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
