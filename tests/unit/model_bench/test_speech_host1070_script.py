"""scripts/model_bench/speech/host1070_script.py: the H6 bash script and its game variant.

The script is checked twice: as text (ports, aliases, no `--rm`, logs before `docker rm`, the
`--memory` cap, the filtered inspect and the restore trap), and by running it under bash with fake
`docker`, `nvidia-smi`, `free`, `sleep` and `powershell.exe` defined as shell functions through
BASH_ENV. The fake GPU is an 8,192 MiB card on which extraction's model takes 6,996 MiB, and a
bench server takes 300 MiB plus 20 MiB per layer (30 layers), so the layer search keeping 512 MiB
free lands on 9.
Nothing here touches a GPU, the network or docker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.speech import core, host1070_script, lowtier_suite  # noqa: E402

PINS = _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "host" / "pins.json"
MODELS = ("gemma4-e2b", "minicpm5-2b", "qwen35-4b", "e4b-ref")
REPEATS = {"gemma4-e2b": 64, "minicpm5-2b": 48, "qwen35-4b": 80, "e4b-ref": 96}
LLM, SVC, TS = "mist-extraction-llm-host", "mist-extraction-host", "mist-extraction-ts"
BENCH, SEARCH, LOAD = "mist-bench-small", "mist-bench-search", "mist-bench-extload"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


def fake_bytes(model_id: str) -> bytes:
    return f"GGUF fake {model_id} weights\n".encode() * REPEATS[model_id]


@pytest.fixture()
def written(tmp_path):
    out = tmp_path / "speech" / "host1070" / "h6.sh"
    out.parent.mkdir(parents=True)
    paths = host1070_script.write_script(out, pins_path=PINS, repo_root=_REPO_ROOT)
    return {p.name: p for p in paths}


@pytest.fixture()
def h6_text(written):
    return written["h6.sh"].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# write_script
# ---------------------------------------------------------------------------


def test_write_script_writes_the_script_its_game_variant_and_the_load_body(written):
    assert sorted(written) == ["extraction_load.json", "h6.sh", "h6_game.sh"]
    assert written["h6.sh"].parent == written["h6_game.sh"].parent


def test_the_load_body_is_the_filler_at_about_36000_characters_and_2048_reply_tokens(written):
    body = json.loads(written["extraction_load.json"].read_text(encoding="utf-8"))
    text = "".join(m["content"] for m in body["messages"])
    filler = core.CACHE_FILLER_PATH.read_text(encoding="utf-8").replace("\r\n", "\n").strip()

    assert 35_000 <= len(text) <= 37_500
    assert filler[:200] in text
    assert body["max_tokens"] == 2048
    assert body["cache_prompt"] is False
    assert "stream" not in body or body["stream"] is False


def test_write_script_refuses_to_overwrite_any_of_its_files(tmp_path):
    out = tmp_path / "h6.sh"
    (tmp_path / "h6_game.sh").write_text("keep me", encoding="utf-8")

    with pytest.raises(core.OutputExistsError):
        host1070_script.write_script(out, pins_path=PINS, repo_root=_REPO_ROOT)

    assert not out.exists()
    assert (tmp_path / "h6_game.sh").read_text(encoding="utf-8") == "keep me"


def test_write_script_defaults_the_pins_to_the_runs_host_pin_file(tmp_path):
    speech = tmp_path / "mh1" / "speech"
    (speech / "host").mkdir(parents=True)
    shutil.copy(PINS, speech / "host" / "pins.json")
    (speech / "host1070").mkdir()

    paths = host1070_script.write_script(
        speech / "host1070" / "h6.sh", pins_path=None, repo_root=_REPO_ROOT
    )

    assert len(paths) == 3


def test_write_script_refuses_without_a_pin_file(tmp_path):
    with pytest.raises(core.PinError, match="--pins"):
        host1070_script.write_script(tmp_path / "h6.sh", pins_path=None, repo_root=_REPO_ROOT)
    assert list(tmp_path.iterdir()) == []


def test_plan_writes_the_host_1070_script_through_speech_bench(tmp_path, capsys):
    from scripts.model_bench import speech_bench

    out = tmp_path / "h6.sh"

    code = speech_bench.main(["plan", "--host-1070-script", str(out), "--pins", str(PINS)])

    printed = capsys.readouterr().out
    assert code == 0, printed
    assert printed.strip().splitlines()[-1] == "plan ok"
    for name in ("h6.sh", "h6_game.sh", "extraction_load.json"):
        assert (tmp_path / name).is_file()
        assert f"[ok] wrote {tmp_path / name}" in printed


def test_expected_aliases_name_each_model_and_pass_in_the_h6_order():
    aliases = host1070_script.expected_aliases()

    assert aliases == [
        "gemma4-e2b@gtx1070-shared-idle",
        "gemma4-e2b@gtx1070-shared-busy",
        "gemma4-e2b@host1070-cpu",
        "minicpm5-2b@gtx1070-shared-idle",
        "minicpm5-2b@gtx1070-shared-busy",
        "minicpm5-2b@host1070-cpu",
        "qwen35-4b@gtx1070-shared-idle",
        "qwen35-4b@gtx1070-shared-busy",
        "qwen35-4b@host1070-cpu",
        "e4b-ref@gtx1070-shared-idle",
        "e4b-ref@host1070-cpu",
        "gemma4-e2b@gtx1070",
        "minicpm5-2b@gtx1070",
        "qwen35-4b@gtx1070",
        "e4b-ref@gtx1070",
    ]
    assert host1070_script.game_alias("gemma4-e2b") == "gemma4-e2b@gtx1070-shared-game"


# ---------------------------------------------------------------------------
# The script as text
# ---------------------------------------------------------------------------


def test_the_script_searches_on_8099_and_serves_on_8090(h6_text):
    assert "SEARCH_PORT=8099" in h6_text
    assert "SERVE_PORT=8090" in h6_text
    assert '--port "$port"' in h6_text


def test_the_script_runs_no_container_with_rm(written):
    for name in ("h6.sh", "h6_game.sh"):
        text = written[name].read_text(encoding="utf-8")
        assert "--rm" not in text


def test_the_script_caps_the_bench_container_and_saves_logs_before_rm(h6_text):
    assert '--memory "${cap}m"' in h6_text
    remove = h6_text[h6_text.index("remove_own() {") :]
    remove = remove[: remove.index("\n}\n")]
    assert remove.index("docker logs") < remove.index("docker stop") < remove.index("docker rm")


def test_the_script_records_the_filtered_inspect_and_diffs_it(h6_text):
    for field in (
        ".Id",
        ".Image",
        ".Config.Cmd",
        ".Config.Env",
        ".Config.Entrypoint",
        ".HostConfig",
    ):
        assert "{{json " + field + "}}" in h6_text
    assert "diff -u" in h6_text
    assert "trap restore EXIT" in h6_text


def test_the_script_names_the_pinned_files_and_images(h6_text):
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    for entry in pins["models"].values():
        assert entry["file"] in h6_text
        assert entry["sha256"] in h6_text
    assert pins["images"]["cpu-llama"] in h6_text


def test_the_script_downloads_nothing_and_holds_no_token(written):
    for name in ("h6.sh", "h6_game.sh"):
        text = written[name].read_text(encoding="utf-8").lower()
        for word in ("hf_token", "huggingface-cli", "docker pull", "wget ", "hf download"):
            assert word not in text, word


@needs_bash
def test_the_scripts_parse_under_bash(written):
    for name in ("h6.sh", "h6_game.sh"):
        proc = subprocess.run(["bash", "-n", str(written[name])], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# The script under fake docker, nvidia-smi, free and sleep
# ---------------------------------------------------------------------------

FAKE_DOCKER = r"""
import json, os, sys
state = os.environ["FAKE_STATE"]
cdir = os.path.join(state, "containers")
argv = sys.argv[1:]
with open(os.path.join(state, "calls.jsonl"), "a") as fh:
    fh.write(json.dumps(argv) + "\n")

def path(name):
    return os.path.join(cdir, name + ".json")

def load(name):
    try:
        with open(path(name)) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

def save(name, doc):
    with open(path(name), "w") as fh:
        json.dump(doc, fh)

def opt(args, flag):
    return args[args.index(flag) + 1] if flag in args else None

cmd = argv[0]
if cmd == "inspect":
    fmt, name = argv[argv.index("--format") + 1], argv[-1]
    doc = load(name)
    if doc is None:
        print("Error: No such object: " + name, file=sys.stderr)
        sys.exit(1)
    if "Health" in fmt:
        print("healthy" if doc["status"] == "running" else doc["status"])
    elif "State.Status" in fmt:
        print(doc["status"])
    else:
        drift = os.environ.get("FAKE_INSPECT_DRIFT") == name and doc.get("restarted")
        print(json.dumps({"Id": name + "-id", "Image": "sha256:" + ("b" if drift else "a"),
                          "Cmd": ["serve"], "Env": ["A=1"], "Entrypoint": None,
                          "HostConfig": {"NetworkMode": "default"}}))
elif cmd == "run":
    name = opt(argv, "--name")
    if load(name) is not None:
        print("Conflict: " + name, file=sys.stderr)
        sys.exit(125)
    alias = opt(argv, "--alias")
    if alias and alias == os.environ.get("FAKE_RUN_FAIL_ALIAS"):
        print("docker: error response from daemon", file=sys.stderr)
        sys.exit(125)
    gpus = "--gpus" in argv
    status = "running"
    if gpus and opt(argv, "-fa") == "on" and os.environ.get("FAKE_FA_FAIL"):
        status = "exited"
    if alias and alias == os.environ.get("FAKE_EXIT_ALIAS"):
        status = "exited"
    ngl = opt(argv, "-ngl")
    save(name, {"status": status, "gpus": gpus, "ngl": None if ngl is None else int(ngl),
                "file": opt(argv, "-m") or "", "alias": alias})
    print(name + "-cid")
elif cmd == "exec":
    doc = load(argv[1])
    if doc is None or doc["status"] != "running":
        sys.exit(1)
    print("{}")
elif cmd == "logs":
    doc = load(argv[1])
    if doc is None:
        sys.exit(1)
    if doc.get("ngl") is not None:
        print("load_tensors: offloaded %d/30 layers to GPU" % min(doc["ngl"], 30))
    print("main: server is listening")
elif cmd in ("stop", "start"):
    for name in argv[1:]:
        doc = load(name)
        if doc is None:
            sys.exit(1)
        doc["status"] = "exited" if cmd == "stop" else "running"
        doc["restarted"] = doc.get("restarted") or cmd == "start"
        save(name, doc)
        print(name)
elif cmd == "rm":
    for name in argv[1:]:
        doc = load(name)
        if doc is None or doc["status"] == "running":
            print("cannot remove " + name, file=sys.stderr)
            sys.exit(1)
        os.remove(path(name))
else:
    print("fake docker: unhandled " + cmd, file=sys.stderr)
    sys.exit(2)
"""

FAKE_NVIDIA_SMI = r"""
import json, os, sys
state = os.environ["FAKE_STATE"]
cdir = os.path.join(state, "containers")
args = " ".join(sys.argv[1:])

def used():
    total = 200
    for entry in os.listdir(cdir):
        with open(os.path.join(cdir, entry)) as fh:
            doc = json.load(fh)
        if doc["status"] != "running":
            continue
        name = entry[:-5]
        if name == "mist-extraction-llm-host":
            total += 6996
        elif name in ("mist-bench-small", "mist-bench-search") and doc["gpus"]:
            heavy = os.environ.get("FAKE_HEAVY")
            total += 600 if heavy and heavy in doc["file"] else 300
            total += 20 * min(doc["ngl"] or 0, 30)
    return total

if "memory.total" in args:
    print(8192)
elif "memory.used" in args:
    for _ in range(3 if "-lms" in args else 1):
        print(used())
else:
    print("FAKE NVIDIA-SMI  used %d MiB / 8192 MiB" % used())
"""

FAKE_SLEEP = """#!/bin/bash
echo "$1" >> "$FAKE_STATE/sleeps.txt"
"""

FAKE_FREE = """#!/bin/bash
echo "               total        used        free      shared  buff/cache   available"
echo "Mem:           15000        8000        2000          10        5000        ${FAKE_AVAILABLE:-6000}"
echo "Swap:           4096           0        4096"
"""

# Windows' FreePhysicalMemory in KiB, with the CRLF a real powershell.exe prints through WSL
# interop. The default, 15,625 MiB, is above the fake `free -m` figure, so the headroom stays 6,000.
FAKE_POWERSHELL = """#!/bin/bash
made=0
if [[ -f $FAKE_STATE/powershell.txt ]]; then made=$(wc -l < "$FAKE_STATE/powershell.txt"); fi
echo "$*" >> "$FAKE_STATE/powershell.txt"
if [[ -n ${FAKE_POWERSHELL_FAIL:-} ]]; then exit 1; fi
if [[ -n ${FAKE_POWERSHELL_FAIL_AFTER:-} ]] && ((made >= FAKE_POWERSHELL_FAIL_AFTER)); then exit 1; fi
printf '%s\\r\\n' "${FAKE_WINDOWS_FREE_KIB:-16000000}"
"""


class Host:
    """A fake host-1070: a home with `~/mist-bench-h6/`, a models directory and fake tools."""

    def __init__(self, tmp_path: Path, written: dict[str, Path]):
        self.root = tmp_path
        self.home = tmp_path / "home"
        self.h6 = self.home / "mist-bench-h6"
        self.h6.mkdir(parents=True)
        for name, path in written.items():
            shutil.copy(path, self.h6 / name)
        self.state = tmp_path / "state"
        (self.state / "containers").mkdir(parents=True)
        for name in (LLM, SVC, TS):
            self.put(name, {"status": "running", "gpus": name == LLM, "ngl": None, "file": ""})
        self.models = tmp_path / "models"
        pins = json.loads(PINS.read_text(encoding="utf-8"))
        for model_id in MODELS:
            target = self.models / pins["models"][model_id]["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fake_bytes(model_id))
        # The fakes are shell functions that hand each call to an interpreter, so they run where
        # the temp directory is mounted noexec (a read-only review runner's /tmp). BASH_ENV
        # defines them before the script's first line, and every bash the script starts reads it.
        self.fakes = tmp_path / "fakes"
        self.fakes.mkdir()
        python = shlex.quote(sys.executable)
        lines = []
        for name, body, interpreter, suffix in (
            ("docker", FAKE_DOCKER, python, ".py"),
            ("nvidia-smi", FAKE_NVIDIA_SMI, python, ".py"),
            ("free", FAKE_FREE, "bash", ".sh"),
            ("sleep", FAKE_SLEEP, "bash", ".sh"),
            ("powershell.exe", FAKE_POWERSHELL, "bash", ".sh"),
        ):
            fake = self.fakes / f"{name}{suffix}"
            fake.write_text(body, encoding="utf-8")
            lines.append(f'{name}() {{ {interpreter} {shlex.quote(str(fake))} "$@"; }}\n')
        self.prelude = self.fakes / "prelude.sh"
        self.prelude.write_text("".join(lines), encoding="utf-8")

    def put(self, name: str, doc: dict) -> None:
        (self.state / "containers" / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")

    def run(self, script: str = "h6.sh", *args: str, **env: str) -> subprocess.CompletedProcess:
        environment = {
            "PATH": os.environ["PATH"],
            "BASH_ENV": str(self.prelude),
            "HOME": str(self.home),
            "FAKE_STATE": str(self.state),
            "MODELS_DIR": str(self.models),
            **env,
        }
        return subprocess.run(
            ["bash", str(self.h6 / script), *args],
            capture_output=True,
            text=True,
            env=environment,
            timeout=240,
        )

    def calls(self) -> list[list[str]]:
        path = self.state / "calls.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return [json.loads(line) for line in lines]

    def runs(self) -> list[list[str]]:
        return [c for c in self.calls() if c[0] == "run"]

    def served(self) -> list[str]:
        return [
            c[c.index("--alias") + 1]
            for c in self.runs()
            if "--port" in c and c[c.index("--port") + 1] == "8090"
        ]

    def sleeps(self) -> list[str]:
        path = self.state / "sleeps.txt"
        return path.read_text(encoding="utf-8").split() if path.exists() else []

    def status(self, name: str) -> str | None:
        path = self.state / "containers" / f"{name}.json"
        return json.loads(path.read_text())["status"] if path.exists() else None

    def summary(self, game: bool = False) -> dict:
        if game:
            (run_dir,) = self.h6.glob("game-*")
            return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        return json.loads((self.h6 / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture()
def host(tmp_path, written):
    return Host(tmp_path / "host", written)


@pytest.fixture(scope="module")
def ran_h6(tmp_path_factory):
    """One plain H6 run, shared by the tests that only read what it did."""
    root = tmp_path_factory.mktemp("h6")
    out = root / "speech" / "host1070" / "h6.sh"
    out.parent.mkdir(parents=True)
    paths = host1070_script.write_script(out, pins_path=PINS, repo_root=_REPO_ROOT)
    host = Host(root / "host", {p.name: p for p in paths})
    return host, host.run()


def index_of(calls, predicate, start=0):
    for i in range(start, len(calls)):
        if predicate(calls[i]):
            return i
    raise AssertionError("no matching call")


H6_SERVED = [
    "gemma4-e2b@gtx1070-shared-idle",
    "gemma4-e2b@gtx1070-shared-busy",
    "minicpm5-2b@gtx1070-shared-idle",
    "minicpm5-2b@gtx1070-shared-busy",
    "qwen35-4b@gtx1070-shared-idle",
    "qwen35-4b@gtx1070-shared-busy",
    "e4b-ref@gtx1070-shared-idle",
    "gemma4-e2b@gtx1070",
    "minicpm5-2b@gtx1070",
    "qwen35-4b@gtx1070",
    "e4b-ref@gtx1070",
]


@needs_bash
def test_h6_runs_its_steps_in_order_and_restores_extraction(ran_h6):
    host, proc = ran_h6

    assert proc.returncode == 0, proc.stderr
    calls = host.calls()
    first_stop = index_of(calls, lambda c: c[0] == "stop")
    inspects = [
        i for i, c in enumerate(calls) if c[0] == "inspect" and "{{json .Id}}" in " ".join(c)
    ]
    assert calls[first_stop] == ["stop", SVC]
    assert min(inspects) < first_stop
    assert index_of(calls, lambda c: c[0] == "run") > first_stop
    assert host.served() == H6_SERVED
    stop_llm = index_of(calls, lambda c: c == ["stop", LLM])
    first_alone = index_of(
        calls,
        lambda c: c[0] == "run"
        and "e4b-ref@gtx1070-shared-idle" not in c
        and "gemma4-e2b@gtx1070" in c,
    )
    assert stop_llm < first_alone
    start_llm = index_of(calls, lambda c: c == ["start", LLM])
    start_svc = index_of(calls, lambda c: c == ["start", SVC])
    assert start_llm < start_svc
    assert max(inspects) > start_svc
    assert (host.status(LLM), host.status(SVC), host.status(TS)) == ("running",) * 3
    for name in (BENCH, SEARCH, LOAD):
        assert host.status(name) is None


@needs_bash
def test_h6_holds_each_window_for_its_length(ran_h6):
    host, proc = ran_h6
    assert proc.returncode == 0

    sleeps = host.sleeps()
    assert sleeps.count("1500") == 4
    assert sleeps.count("720") == 3 + 4
    windows = host.summary()["windows"]
    assert [w["window_s"] for w in windows] == [1500, 720] * 3 + [1500] + [720] * 4
    assert [w["pass"] for w in windows] == ["idle", "busy"] * 3 + ["idle"] + ["alone"] * 4


@needs_bash
def test_h6_searches_on_8099_with_the_load_loop_and_serves_the_largest_count_keeping_512_free(
    ran_h6,
):
    host, proc = ran_h6
    assert proc.returncode == 0

    calls = host.calls()
    searches = [
        c for c in calls if c[0] == "run" and "--name" in c and c[c.index("--name") + 1] == SEARCH
    ]
    assert searches and all(c[c.index("--port") + 1] == "8099" for c in searches)
    assert int(searches[0][searches[0].index("-ngl") + 1]) == 0
    summary = host.summary()
    shared = [w for w in summary["windows"] if w["placement"] == "gtx1070-shared"]
    assert {w["ngl"] for w in shared} == {9}
    fitting = [s for s in summary["searches"] if s["model"] == "gemma4-e2b" and s["fits"]]
    assert max(s["ngl"] for s in fitting) == 9
    assert all(s["gpu_free_mib"] >= 512 for s in fitting)
    # the load loop runs through each search and each busy window, and nowhere else
    load_runs = [i for i, c in enumerate(calls) if c[0] == "run" and LOAD in c]
    assert len(load_runs) == 4 + 3
    loop = calls[load_runs[0]]
    assert loop[loop.index("--network") + 1] == f"container:{TS}"
    assert "/load/extraction_load.json" in loop[-1]
    assert "http://mist-extraction-llm-host:8080/v1/chat/completions" in loop[-1]


@needs_bash
def test_h6_serves_with_the_memory_cap_on_the_sidecar_network_and_never_with_rm(ran_h6):
    host, proc = ran_h6
    assert proc.returncode == 0

    for call in host.runs():
        assert "--rm" not in call
        name = call[call.index("--name") + 1]
        if name in (BENCH, SEARCH):
            assert call[call.index("--memory") + 1] == f"{6000 - 1024}m"
            assert call[call.index("--network") + 1] == f"container:{TS}"
            assert f"{host.models}:/models:ro" in call
            assert ["--reasoning", "off"] == call[
                call.index("--reasoning") : call.index("--reasoning") + 2
            ]
            assert call[call.index("-c") + 1] == "8192"


@needs_bash
def test_h6_saves_logs_before_every_rm(ran_h6):
    host, proc = ran_h6
    assert proc.returncode == 0

    calls = host.calls()
    removed = [i for i, c in enumerate(calls) if c[0] == "rm"]
    assert removed
    for i in removed:
        name = calls[i][1]
        last_run = max(j for j in range(i) if calls[j][0] == "run" and name in calls[j])
        assert any(calls[j] == ["logs", name] for j in range(last_run, i)), name
        assert any(calls[j] == ["stop", name] for j in range(last_run, i)), name
    logs = sorted(p.name for p in (host.h6 / "logs").iterdir())
    assert any("gemma4-e2b@gtx1070-shared-idle" in n for n in logs)


@needs_bash
def test_h6_writes_a_summary_remote_memory_can_read(ran_h6, tmp_path):
    host, proc = ran_h6
    assert proc.returncode == 0

    summary = host.summary()
    assert summary["schema"] == 1 and summary["variant"] == "h6"
    assert summary["card_total_mib"] == 8192
    assert summary["restore"]["inspect_diff"]["empty"] is True
    assert summary["restore"]["health"] == {LLM: "healthy", SVC: "healthy"}
    for name in (LLM, SVC):
        before = (host.h6 / "before" / f"{name}.inspect.json").read_text()
        assert before == (host.h6 / "after" / f"{name}.inspect.json").read_text()
    idle = summary["windows"][0]
    assert idle["alias"] == "gemma4-e2b@gtx1070-shared-idle"
    assert idle["gpu_peak_mib"] == 200 + 6996 + 300 + 20 * 9
    assert idle["status"] == "served" and idle["fallback"] == "none"
    assert idle["load_s"] is not None and idle["unload_s"] is not None

    path = lowtier_suite.run_remote_memory(
        tmp_path / "results", "mh1", summary_path=host.h6 / "summary.json", git_sha="abc"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in rows} >= {"window", "layers", "restore"}


@needs_bash
def test_h6_skips_a_model_whose_file_does_not_match_its_pin(host):
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    (host.models / pins["models"]["qwen35-4b"]["file"]).write_bytes(b"not the pinned weights\n")

    assert host.run().returncode == 0

    assert not any(a.startswith("qwen35-4b@") for a in host.served())
    skips = {(s["model"], s["placement"]): s["reason"] for s in host.summary()["skips"]}
    assert "differs from the pin file" in skips[("qwen35-4b", "gtx1070-shared")]
    assert "differs from the pin file" in skips[("qwen35-4b", "gtx1070")]


@needs_bash
def test_h6_skips_a_model_whose_file_does_not_fit_the_available_memory(host, tmp_path):
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    big = b"GGUF large fake weights\n" * 70_000
    (host.models / pins["models"]["e4b-ref"]["file"]).write_bytes(big)
    pins["models"]["e4b-ref"].update(sha256=hashlib.sha256(big).hexdigest(), size=len(big))
    pins_path = tmp_path / "big_pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    out = tmp_path / "again" / "h6.sh"
    out.parent.mkdir()
    for path in host1070_script.write_script(out, pins_path=pins_path, repo_root=_REPO_ROOT):
        shutil.copy(path, host.h6 / path.name)

    assert host.run(FAKE_AVAILABLE="1025").returncode == 0

    served = host.served()
    assert "e4b-ref@gtx1070-shared-idle" not in served
    assert "e4b-ref@gtx1070" in served
    skips = {(s["model"], s["placement"]): s["reason"] for s in host.summary()["skips"]}
    assert skips[("e4b-ref", "gtx1070-shared")] == "does not fit beside extraction in host memory"


@needs_bash
def test_h6_falls_back_to_host1070_cpu_when_no_layer_count_keeps_512_free(host):
    assert host.run(FAKE_HEAVY="MiniCPM").returncode == 0

    served = host.served()
    assert "minicpm5-2b@host1070-cpu" in served
    assert "minicpm5-2b@gtx1070-shared-idle" not in served
    cpu_run = next(c for c in host.runs() if "minicpm5-2b@host1070-cpu" in c)
    assert "--gpus" not in cpu_run
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["images"]["cpu-llama"] in cpu_run
    skips = {(s["model"], s["placement"]): s["reason"] for s in host.summary()["skips"]}
    assert skips[("minicpm5-2b", "gtx1070-shared")] == "does not fit beside extraction"


@needs_bash
def test_h6_retries_a_window_with_flash_attention_off_and_an_f16_cache(host):
    assert host.run(FAKE_FA_FAIL="1").returncode == 0

    windows = [w for w in host.summary()["windows"] if w["image"] == "cuda"]
    assert windows and {w["fallback"] for w in windows} == {"fa-off-f16"}
    assert {(w["fa"], w["cache"]) for w in windows} == {("off", "f16")}
    retried = [c for c in host.runs() if "gemma4-e2b@gtx1070-shared-idle" in c]
    assert [c[c.index("-fa") + 1] for c in retried] == ["on", "off"]
    assert retried[1][retried[1].index("-ctk") + 1] == "f16"


@needs_bash
def test_h6_records_a_failed_window_and_carries_on(host):
    proc = host.run(FAKE_EXIT_ALIAS="minicpm5-2b@gtx1070")

    assert proc.returncode == 0, proc.stderr
    windows = {w["alias"]: w for w in host.summary()["windows"]}
    assert windows["minicpm5-2b@gtx1070"]["status"] == "failed"
    assert windows["qwen35-4b@gtx1070"]["status"] == "served"


@needs_bash
def test_h6_restarts_what_it_stopped_when_a_step_fails(host):
    proc = host.run(FAKE_RUN_FAIL_ALIAS="qwen35-4b@gtx1070-shared-idle")

    assert proc.returncode != 0
    calls = host.calls()
    assert ["start", SVC] in calls
    assert ["start", LLM] not in calls
    assert (host.status(LLM), host.status(SVC)) == ("running", "running")
    for name in (BENCH, SEARCH, LOAD):
        assert host.status(name) is None
    summary = host.summary()
    assert summary["restore"]["started"] == [SVC]
    assert summary["exit_code"] != 0


@needs_bash
def test_h6_reports_an_inspect_diff_that_is_not_empty(host):
    assert host.run(FAKE_INSPECT_DRIFT=SVC).returncode != 0

    diff = host.summary()["restore"]["inspect_diff"]
    assert diff["empty"] is False
    assert diff["containers"] == {LLM: True, SVC: False}


@needs_bash
def test_h6_refuses_without_a_models_directory_before_stopping_anything(host):
    proc = host.run(MODELS_DIR="")

    assert proc.returncode != 0
    assert "MODELS_DIR" in proc.stderr
    assert not any(c[0] in ("stop", "run") for c in host.calls())


@needs_bash
def test_h6_refuses_when_a_bench_container_already_exists(host):
    host.put(BENCH, {"status": "running", "gpus": True, "ngl": 1, "file": ""})

    proc = host.run()

    assert proc.returncode != 0
    assert BENCH in proc.stderr
    assert not any(c[0] in ("stop", "run") for c in host.calls())


@needs_bash
def test_h6_refuses_to_run_twice_into_the_same_folder(host):
    assert host.run().returncode == 0
    count = len(host.calls())

    proc = host.run()

    assert proc.returncode != 0
    assert not any(c[0] in ("stop", "run") for c in host.calls()[count:])


# ---------------------------------------------------------------------------
# The game variant
# ---------------------------------------------------------------------------


@needs_bash
def test_the_game_variant_serves_one_model_for_40_minutes_and_restores_on_its_own(host):
    proc = host.run("h6_game.sh", "gemma4-e2b", "9")

    assert proc.returncode == 0, proc.stderr
    calls = host.calls()
    assert ["stop", SVC] in calls
    assert ["stop", LLM] not in calls
    assert host.served() == ["gemma4-e2b@gtx1070-shared-game"]
    run = host.runs()[0]
    assert run[run.index("-ngl") + 1] == "9"
    assert "2400" in host.sleeps()
    assert ["start", SVC] in calls
    assert host.status(SVC) == "running"
    summary = host.summary(game=True)
    assert summary["variant"] == "game"
    assert summary["windows"][0]["window_s"] == 2400
    assert summary["restore"]["inspect_diff"]["empty"] is True
    assert not (host.h6 / "summary.json").exists()


@needs_bash
def test_the_game_variant_searches_the_layer_count_when_none_is_given(host):
    assert host.run("h6_game.sh", "gemma4-e2b").returncode == 0

    assert any(SEARCH in c for c in host.runs())
    assert host.summary(game=True)["windows"][0]["ngl"] == 9


@needs_bash
def test_the_game_variant_refuses_an_unknown_model_before_stopping_anything(host):
    proc = host.run("h6_game.sh", "not-a-model")

    assert proc.returncode != 0
    assert not any(c[0] in ("stop", "run") for c in host.calls())


# ---------------------------------------------------------------------------
# Windows free memory (plan v3, T11): the shared fit check reads the smaller of `free -m` and
# Windows' FreePhysicalMemory; the `--memory` cap and the sampler stay on `free -m`.
# ---------------------------------------------------------------------------

POWERSHELL_ARGS = (
    "-NoProfile -NonInteractive -Command (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"
)
WINDOW_KEYS = ("windows_free_before_mib", "headroom_before_mib", "windows_free_end_mib")


def powershell_calls(host: Host) -> list[str]:
    path = host.state / "powershell.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def install_large_e4b(host: Host, tmp_path: Path) -> None:
    """Regenerate the scripts with a 1.6 MiB `e4b-ref` file: rounded up, its fit check needs 1,026."""
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    big = b"GGUF large fake weights\n" * 70_000
    (host.models / pins["models"]["e4b-ref"]["file"]).write_bytes(big)
    pins["models"]["e4b-ref"].update(sha256=hashlib.sha256(big).hexdigest(), size=len(big))
    pins_path = tmp_path / "big_pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    out = tmp_path / "again" / "h6.sh"
    out.parent.mkdir()
    for path in host1070_script.write_script(out, pins_path=pins_path, repo_root=_REPO_ROOT):
        shutil.copy(path, host.h6 / path.name)


def function_body(text: str, name: str) -> str:
    body = text[text.index(f"\n{name}() {{") :]
    return body[: body.index("\n}\n")]


def expected_powershell_calls(summary: dict) -> int:
    """1 in preflight, 1 per shared fit check, 2 per served window and 1 per failed window."""
    shared = {"gtx1070-shared", "host1070-cpu"}
    checked = {w["model"] for w in summary["windows"] if w["placement"] in shared}
    checked |= {s["model"] for s in summary["skips"] if "need_mib" in s}
    statuses = [w["status"] for w in summary["windows"]]
    return 1 + len(checked) + 2 * statuses.count("served") + statuses.count("failed")


def test_the_script_reads_windows_free_memory_but_its_sampler_does_not(h6_text):
    reader = function_body(h6_text, "windows_free_mib")
    assert (
        'powershell.exe -NoProfile -NonInteractive -Command "(Get-CimInstance '
        'Win32_OperatingSystem).FreePhysicalMemory"' in reader
    )
    assert "\nheadroom_mib() {" in h6_text
    sampler = function_body(h6_text, "start_samplers")
    assert "available_mib" in sampler
    for word in ("windows_free_mib", "headroom_mib", "powershell.exe"):
        assert word not in sampler, word


@needs_bash
def test_h6_skips_a_shared_placement_that_exceeds_the_windows_headroom(host, tmp_path):
    install_large_e4b(host, tmp_path)

    proc = host.run(FAKE_WINDOWS_FREE_KIB="1049600")

    assert proc.returncode == 0, proc.stderr
    served = host.served()
    assert "e4b-ref@gtx1070-shared-idle" not in served
    assert "e4b-ref@gtx1070" in served
    (skip,) = (
        s
        for s in host.summary()["skips"]
        if (s["model"], s["placement"]) == ("e4b-ref", "gtx1070-shared")
    )
    assert skip["reason"] == "does not fit beside extraction in host memory"
    assert (skip["need_mib"], skip["headroom_mib"]) == (1026, 1025)


@needs_bash
def test_h6_keeps_the_memory_cap_on_free_when_windows_has_less(host):
    proc = host.run(FAKE_WINDOWS_FREE_KIB="3072000")

    assert proc.returncode == 0, proc.stderr
    assert host.served() == H6_SERVED
    caps = [c[c.index("--memory") + 1] for c in host.runs() if c[c.index("--name") + 1] != LOAD]
    assert caps and set(caps) == {f"{6000 - 1024}m"}
    windows = host.summary()["windows"]
    assert {w["memory_cap_mib"] for w in windows} == {6000 - 1024}
    assert {w["windows_free_before_mib"] for w in windows} == {3000}
    assert {w["headroom_before_mib"] for w in windows} == {3000}


@needs_bash
def test_h6_records_windows_free_memory_in_each_window_and_the_summary(ran_h6):
    host, proc = ran_h6
    assert proc.returncode == 0, proc.stderr

    summary = host.summary()
    for window in summary["windows"]:
        for key in WINDOW_KEYS:
            assert isinstance(window[key], int), (window["alias"], key)
        assert window["windows_free_before_mib"] == window["windows_free_end_mib"] == 15625
        assert window["headroom_before_mib"] == 6000
    before = summary["before"]
    assert (before["windows_free_mib"], before["headroom_mib"]) == (15625, 6000)
    assert before["free_available_mib"] == 6000


@needs_bash
@pytest.mark.parametrize(
    "fault",
    [{"FAKE_POWERSHELL_FAIL": "1"}, {"FAKE_WINDOWS_FREE_KIB": "junk"}],
    ids=["powershell-fails", "not-a-number"],
)
def test_h6_stops_before_changing_anything_when_the_windows_read_fails(host, fault):
    proc = host.run(**fault)

    assert proc.returncode != 0
    assert "powershell.exe" in proc.stderr
    assert not any(c[0] in ("stop", "run") for c in host.calls())
    assert not (host.h6 / "before").exists()
    assert (host.status(LLM), host.status(SVC)) == ("running", "running")

    again = host.run()

    assert again.returncode == 0, again.stderr
    assert host.served() == H6_SERVED


@needs_bash
def test_h6_restores_extraction_when_a_windows_read_fails_after_preflight(host):
    proc = host.run(FAKE_POWERSHELL_FAIL_AFTER="1")

    assert proc.returncode != 0
    assert "powershell.exe" in proc.stderr
    assert len(powershell_calls(host)) == 2
    assert not any(c[0] == "run" for c in host.calls())
    assert ["stop", SVC] in host.calls()
    assert (host.status(LLM), host.status(SVC)) == ("running", "running")
    assert host.summary()["exit_code"] != 0


@needs_bash
def test_h6_reads_windows_once_in_preflight_per_fit_check_and_twice_per_window(ran_h6):
    host, proc = ran_h6
    assert proc.returncode == 0, proc.stderr

    calls = powershell_calls(host)
    assert set(calls) == {POWERSHELL_ARGS}
    assert len(calls) == expected_powershell_calls(host.summary()) == 1 + 4 + 2 * 11


@needs_bash
def test_h6_reads_windows_once_for_a_window_that_failed_to_start(host):
    proc = host.run(FAKE_EXIT_ALIAS="minicpm5-2b@gtx1070")

    assert proc.returncode == 0, proc.stderr
    summary = host.summary()
    failed = {w["alias"]: w for w in summary["windows"]}["minicpm5-2b@gtx1070"]
    assert failed["status"] == "failed"
    assert failed["windows_free_before_mib"] == 15625
    assert failed["windows_free_end_mib"] is None
    assert len(powershell_calls(host)) == expected_powershell_calls(summary) == 1 + 4 + 2 * 10 + 1


@needs_bash
def test_the_game_variant_refuses_a_model_beyond_the_windows_headroom_before_stopping(host):
    proc = host.run("h6_game.sh", "gemma4-e2b", "9", FAKE_WINDOWS_FREE_KIB="1048576")

    assert proc.returncode != 0
    assert "does not fit beside extraction in host memory" in proc.stderr
    assert "1025" in proc.stderr and "1024" in proc.stderr
    assert not any(c[0] in ("stop", "run") for c in host.calls())
    assert not list(host.h6.glob("game-*"))
