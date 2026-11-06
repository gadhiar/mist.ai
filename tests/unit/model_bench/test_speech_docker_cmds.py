"""scripts/model_bench/speech/docker_cmds.py and core's small-model serving.

The argv tests check every flag of every local placement against the real b11151
`llama-server --help` capture (tests/unit/model_bench/fixtures/host/llama_server_help_b11151.txt)
with `probes/help_flags.py`. The serve tests drive `serve_small`/`unserve_small` through a fake
docker and a fake chat client with a scripted clock: no docker, no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench.bench_host import parse_compose_image  # noqa: E402
from scripts.model_bench.probes.help_flags import parse_help_flags, unknown_flags  # noqa: E402
from scripts.model_bench.speech import core, docker_cmds  # noqa: E402

HELP_TEXT = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "model_bench"
    / "fixtures"
    / "host"
    / "llama_server_help_b11151.txt"
).read_text(encoding="utf-8")
PINS_PATH = (
    _REPO_ROOT / "tests" / "unit" / "model_bench" / "fixtures" / "speech" / "host" / "pins.json"
)
CUDA_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda-b11151@sha256:" + "a" * 64

FAKE_MODEL_BYTES = {
    "gemma4-e2b": b"GGUF fake gemma4-e2b weights\n" * 64,
    "minicpm5-2b": b"GGUF fake minicpm5-2b weights\n" * 48,
    "qwen35-4b": b"GGUF fake qwen35-4b weights\n" * 80,
    "e4b-ref": b"GGUF fake e4b-ref weights\n" * 96,
}


# ---------------------------------------------------------------------------
# Shared argv helpers
# ---------------------------------------------------------------------------


def test_ro_mount_is_read_only():
    assert docker_cmds.ro_mount("D:/models", "/models") == ["-v", "D:/models:/models:ro"]


def test_publish_local_binds_127_0_0_1_only():
    assert docker_cmds.publish_local(8081, 8080) == ["-p", "127.0.0.1:8081:8080"]


def test_gpus_all():
    assert docker_cmds.gpus_all() == ["--gpus", "all"]


def test_env_args_are_one_e_per_pair_in_order():
    assert docker_cmds.env_args({"A": "1", "B": "x"}) == ["-e", "A=1", "-e", "B=x"]


def test_engine_env_sets_hf_hub_offline():
    args = docker_cmds.engine_env({"OMP_NUM_THREADS": "4"})

    assert args[:2] == ["-e", "HF_HUB_OFFLINE=1"]
    assert "OMP_NUM_THREADS=4" in args


def test_cuda_llama_image_is_the_compose_mist_llm_image():
    compose = _REPO_ROOT / "docker-compose.yml"

    assert docker_cmds.cuda_llama_image(compose) == parse_compose_image(compose, "mist-llm")


def test_cpu_llama_image_is_pinned_by_digest():
    assert docker_cmds.CPU_LLAMA_IMAGE == (
        "ghcr.io/ggml-org/llama.cpp:server-b11151"
        "@sha256:fffcc1af1105bafc713a47b6680fb7cf5856ed43ea2e9e8bca5a62bbd6365edf"
    )


# ---------------------------------------------------------------------------
# small-serve argv per placement
# ---------------------------------------------------------------------------


def _argv(placement: str, ngl: int | None = None, model: str = "gemma4-e2b") -> list[str]:
    return core.small_serve_argv(
        core.load_small_arms(),
        model,
        placement,
        models_dir_host="D:/models/small",
        model_file="gemma-4-E2B-it-Q4_K_M.gguf",
        cuda_image=CUDA_IMAGE,
        ngl=ngl,
    )


def _server_tail(argv: list[str]) -> list[str]:
    for image in (CUDA_IMAGE, docker_cmds.CPU_LLAMA_IMAGE):
        if image in argv:
            return argv[argv.index(image) + 1 :]
    raise AssertionError("no llama.cpp image in argv")


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("placement", "ngl"),
    [
        pytest.param("gpu-full", None, id="gpu-full"),
        pytest.param("gpu-4g", 20, id="gpu-4g"),
        pytest.param("gpu-2g", 9, id="gpu-2g"),
        pytest.param("cpu", None, id="cpu"),
    ],
)
def test_small_serve_argv_uses_only_flags_the_b11151_help_lists(placement, ngl):
    tail = _server_tail(_argv(placement, ngl))

    assert unknown_flags(tail, parse_help_flags(HELP_TEXT)) == []


@pytest.mark.parametrize(
    ("placement", "ngl", "expected_ngl"),
    [
        pytest.param("gpu-full", None, "999", id="gpu-full"),
        pytest.param("gpu-4g", 20, "20", id="gpu-4g"),
        pytest.param("gpu-2g", 9, "9", id="gpu-2g"),
    ],
)
def test_gpu_placements_use_the_cuda_image_with_gpus_and_ngl(placement, ngl, expected_ngl):
    argv = _argv(placement, ngl)
    tail = _server_tail(argv)

    assert CUDA_IMAGE in argv
    assert argv[argv.index("--gpus") + 1] == "all"
    assert _flag_value(tail, "-ngl") == expected_ngl


def test_cpu_placement_uses_the_cpu_image_with_neither_gpus_nor_ngl():
    argv = _argv("cpu")

    assert docker_cmds.CPU_LLAMA_IMAGE in argv
    assert CUDA_IMAGE not in argv
    assert "--gpus" not in argv
    assert "-ngl" not in argv


@pytest.mark.parametrize("placement", ["gpu-full", "gpu-4g", "gpu-2g", "cpu"])
def test_small_serve_argv_carries_the_common_server_settings(placement):
    argv = _argv(placement, 12 if placement in ("gpu-4g", "gpu-2g") else None)
    tail = _server_tail(argv)

    assert argv[:5] == ["docker", "run", "-d", "--name", "mist-bench-small"]
    assert "--rm" not in argv
    assert _flag_value(argv, "-p") == "127.0.0.1:8081:8080"
    assert _flag_value(argv, "-v") == "D:/models/small:/models:ro"
    assert _flag_value(tail, "-m") == "/models/gemma-4-E2B-it-Q4_K_M.gguf"
    expected = {
        "-c": "8192",
        "-ctk": "q8_0",
        "-ctv": "q8_0",
        "-fa": "on",
        "-np": "1",
        "--reasoning": "off",
        "--fit": "off",
        "-t": "8",
        "--temp": "0.7",
        "--top-p": "0.9",
        "--top-k": "40",
        "--min-p": "0",
        "--alias": f"gemma4-e2b@{placement}",
    }
    assert {flag: _flag_value(tail, flag) for flag in expected} == expected
    assert "--jinja" in tail


@pytest.mark.parametrize("placement", ["gpu-4g", "gpu-2g"])
def test_slice_placements_refuse_without_an_explicit_ngl(placement):
    with pytest.raises(core.PlacementError):
        _argv(placement, None)


@pytest.mark.parametrize("placement", ["gpu-full", "cpu"])
def test_full_and_cpu_placements_refuse_an_explicit_ngl(placement):
    with pytest.raises(core.PlacementError):
        _argv(placement, 10)


@pytest.mark.parametrize("placement", ["gtx1070", "gtx1070-shared", "host1070-cpu"])
def test_small_serve_argv_refuses_the_host_1070_placements(placement):
    with pytest.raises(core.PlacementError, match="host-1070"):
        _argv(placement)


def test_small_serve_argv_refuses_an_unknown_model():
    with pytest.raises(core.PlacementError):
        _argv("gpu-full", model="no-such-model")


# ---------------------------------------------------------------------------
# serve_small / unserve_small with fakes
# ---------------------------------------------------------------------------


class FakeDocker:
    def __init__(self, state: str = "absent"):
        self.state = state
        self.calls: list[tuple] = []

    def container_state(self, name: str) -> str:
        self.calls.append(("state", name))
        return self.state

    def run_detached(self, argv: list[str]) -> str:
        self.calls.append(("run", argv))
        self.state = "running"
        return "cid"

    def wait_healthy(self, base_url: str, container: str, timeout: float) -> None:
        self.calls.append(("health", base_url, container))

    def logs(self, name: str) -> tuple[str, str]:
        self.calls.append(("logs", name))
        return ("server out", "server err")

    def stop(self, name: str) -> None:
        self.calls.append(("stop", name))
        self.state = "exited"

    def rm(self, name: str) -> None:
        self.calls.append(("rm", name))
        self.state = "absent"


def scripted_clock(*values: float):
    it = iter(values)
    return lambda: next(it)


def fake_chat(first_delta_at: float | None):
    def chat(base_url, body, *, clock):
        t_request = clock()
        return core.ChatStreamResult(
            content="hi" if first_delta_at is not None else "",
            reasoning_content="",
            tool_calls=(),
            t_request=t_request,
            t_first_content=first_delta_at,
            t_first_tool_call=None,
            t_end=first_delta_at or t_request,
            finish_reason="stop",
            timings=None,
        )

    return chat


@pytest.fixture()
def models_dirs(tmp_path):
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    dirs = {"small": tmp_path / "small", "main": tmp_path / "main"}
    arms = core.load_small_arms()
    for model_id, data in FAKE_MODEL_BYTES.items():
        target = dirs[arms["models"][model_id]["models_dir"]] / pins["models"][model_id]["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return dirs


def _serve(tmp_path, models_dirs, docker, *, model="gemma4-e2b", placement="gpu-full", ngl=None):
    return core.serve_small(
        model,
        placement,
        out_dir=tmp_path / "out" / model / placement,
        pins_path=PINS_PATH,
        models_dirs=models_dirs,
        ngl=ngl,
        cuda_image=CUDA_IMAGE,
        docker=docker,
        chat=fake_chat(first_delta_at=112.5),
        clock=scripted_clock(100.0, 110.0),
    )


def test_serve_small_times_load_from_container_start_to_the_first_warmup_token(
    tmp_path, models_dirs
):
    docker = FakeDocker()

    result = _serve(tmp_path, models_dirs, docker)

    assert result.load_s == pytest.approx(12.5)
    assert result.alias == "gemma4-e2b@gpu-full"
    record = json.loads((result.out_dir / "serve.json").read_text())
    assert record["load_s"] == pytest.approx(12.5)
    assert record["argv"] == list(result.argv)
    run_argv = next(c[1] for c in docker.calls if c[0] == "run")
    assert run_argv == list(result.argv)


def test_serve_small_mounts_e4b_ref_from_the_main_models_dir(tmp_path, models_dirs):
    result = _serve(tmp_path, models_dirs, FakeDocker(), model="e4b-ref")

    assert f"{models_dirs['main'].resolve().as_posix()}:/models:ro" in result.argv


def test_serve_small_refuses_a_model_file_whose_sha256_differs_before_any_docker_call(
    tmp_path, models_dirs
):
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    path = models_dirs["small"] / pins["models"]["qwen35-4b"]["file"]
    data = bytearray(path.read_bytes())
    data[3] ^= 0x01
    path.write_bytes(bytes(data))
    docker = FakeDocker()

    with pytest.raises(core.PinError):
        _serve(tmp_path, models_dirs, docker, model="qwen35-4b")

    assert not any(c[0] == "run" for c in docker.calls)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("placement", ["gtx1070", "gtx1070-shared", "host1070-cpu"])
def test_serve_small_refuses_the_host_1070_placements(tmp_path, models_dirs, placement):
    docker = FakeDocker()

    with pytest.raises(core.PlacementError):
        _serve(tmp_path, models_dirs, docker, placement=placement)

    assert docker.calls == []


def test_serve_small_refuses_when_the_container_already_exists(tmp_path, models_dirs):
    docker = FakeDocker(state="exited")

    with pytest.raises(core.SpeechBenchError, match="mist-bench-small"):
        _serve(tmp_path, models_dirs, docker)

    assert not any(c[0] == "run" for c in docker.calls)


def test_serve_small_refuses_an_existing_output_directory(tmp_path, models_dirs):
    (tmp_path / "out" / "gemma4-e2b" / "gpu-full").mkdir(parents=True)

    with pytest.raises(core.OutputExistsError):
        _serve(tmp_path, models_dirs, FakeDocker())


def test_serve_small_saves_logs_and_fails_when_the_warmup_returns_nothing(tmp_path, models_dirs):
    docker = FakeDocker()

    with pytest.raises(core.ServeFailedError):
        core.serve_small(
            "gemma4-e2b",
            "gpu-full",
            out_dir=tmp_path / "out",
            pins_path=PINS_PATH,
            models_dirs=models_dirs,
            cuda_image=CUDA_IMAGE,
            docker=docker,
            chat=fake_chat(first_delta_at=None),
            clock=scripted_clock(100.0, 110.0),
        )

    assert (tmp_path / "out" / "serve_failed.log").read_text().count("server out") == 1


def test_unserve_small_times_the_stop_and_saves_logs_before_rm(tmp_path, models_dirs):
    docker = FakeDocker()
    result = _serve(tmp_path, models_dirs, docker)
    docker.calls.clear()

    unload_s = core.unserve_small(result.out_dir, docker=docker, clock=scripted_clock(200.0, 203.0))

    assert unload_s == pytest.approx(3.0)
    order = [c[0] for c in docker.calls if c[0] in ("stop", "logs", "rm")]
    assert order == ["stop", "logs", "rm"]
    assert "server err" in (result.out_dir / "server.log").read_text()
    assert json.loads((result.out_dir / "unserve.json").read_text())["unload_s"] == 3.0


def test_unserve_small_refuses_when_no_container_exists(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(core.SpeechBenchError):
        core.unserve_small(out, docker=FakeDocker(state="absent"), clock=scripted_clock(1.0))
