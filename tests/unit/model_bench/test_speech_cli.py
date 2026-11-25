"""scripts/model_bench/speech_bench.py: the parser, the suite registry, `plan` and `selftest`.

Registry behaviour is tested on fake suite modules patched into the registry, never by asserting
that a real later suite is unbuilt: tasks T2 to T8 build those modules, and these tests must keep
passing when they land.
"""

from __future__ import annotations

import math
import sys
import types
import wave
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import speech_bench  # noqa: E402

FAKE_PKG = "zz_fake_speech_pkg"


def make_fake_suite(name: str, subcommands: tuple[str, ...], *, problems=(), selftest_error=None):
    module = types.ModuleType(f"{FAKE_PKG}.{name}")

    def register(subparsers):
        for sub in subcommands:
            parser = subparsers.add_parser(sub)
            parser.add_argument("--flag")

            def run(args, sub=sub):
                print(f"ran {sub} flag={args.flag}")
                return 0

            parser.set_defaults(func=run)

    def selftest():
        if selftest_error:
            raise AssertionError(selftest_error)

    module.register = register
    module.container_argv = lambda ctx: [(f"{name} probe", ["docker", "run", ctx.cuda_llama_image])]
    module.validate_inputs = lambda repo_root: list(problems)
    module.selftest = selftest
    return module


@pytest.fixture()
def fake_registry(monkeypatch):
    package = types.ModuleType(FAKE_PKG)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, FAKE_PKG, package)

    def install(*suites: tuple[str, tuple[str, ...], types.ModuleType | None]):
        registry = []
        for name, subcommands, module in suites:
            if module is not None:
                monkeypatch.setitem(sys.modules, f"{FAKE_PKG}.{name}", module)
            registry.append(speech_bench.SuiteSpec(name, subcommands, package=FAKE_PKG))
        monkeypatch.setattr(speech_bench, "SUITE_REGISTRY", tuple(registry))

    return install


# ---------------------------------------------------------------------------
# The registry contract
# ---------------------------------------------------------------------------


def test_registry_lists_the_suite_modules_and_their_subcommands_in_order():
    got = [(spec.module, spec.subcommands) for spec in speech_bench.SUITE_REGISTRY]

    assert got == [
        ("tts_suite", ("tts",)),
        ("stt_suite", ("stt", "prepare")),
        ("turn_suite", ("turn",)),
        ("cache_suite", ("cache",)),
        ("fit_suite", ("fit",)),
        ("listening", ("listen-pack", "reply-ab")),
        ("lowtier_suite", ("lowtier", "remote-memory")),
        ("chain", ("chain",)),
        ("game_suite", ("game-session",)),
        ("voice_design", ("voice-design",)),
        ("report", ("report",)),
    ]
    assert all(spec.package == "scripts.model_bench.speech" for spec in speech_bench.SUITE_REGISTRY)


def test_every_registered_subcommand_is_on_the_parser():
    parsers = speech_bench.subcommand_parsers(speech_bench.build_parser())

    expected = {"plan", "selftest", "small-serve", "small-unserve"}
    for spec in speech_bench.SUITE_REGISTRY:
        expected |= set(spec.subcommands)
    assert expected <= set(parsers)


def test_unbuilt_suite_subcommand_prints_not_built_and_exits_nonzero(fake_registry, capsys):
    fake_registry(("fake_missing", ("fake-b",), None))

    code = speech_bench.main(["fake-b", "--results-root", "x", "--run", "y"])

    assert code != 0
    assert "[FAIL] fake-b: fake_missing not built" in capsys.readouterr().out


def test_built_suite_subcommand_runs_the_function_its_module_registered(fake_registry, capsys):
    fake_registry(("fake_built", ("fake-a",), make_fake_suite("fake_built", ("fake-a",))))

    code = speech_bench.main(["fake-a", "--flag", "v"])

    assert code == 0
    assert "ran fake-a flag=v" in capsys.readouterr().out


def test_a_missing_dependency_inside_a_built_module_still_raises(
    fake_registry, monkeypatch, tmp_path
):
    pkg = tmp_path / "zz_dep_speech_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken_suite.py").write_text("import zz_no_such_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        speech_bench,
        "SUITE_REGISTRY",
        (speech_bench.SuiteSpec("broken_suite", ("fake-c",), package="zz_dep_speech_pkg"),),
    )

    with pytest.raises(ModuleNotFoundError) as info:
        speech_bench.build_parser()

    assert info.value.name == "zz_no_such_dependency_xyz"


def test_unknown_options_are_still_refused_for_a_built_subcommand(fake_registry):
    fake_registry(("fake_built", ("fake-a",), make_fake_suite("fake_built", ("fake-a",))))

    with pytest.raises(SystemExit) as info:
        speech_bench.main(["fake-a", "--no-such-option"])

    assert info.value.code == 2


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_skips_unbuilt_suites_prints_built_argv_and_ends_plan_ok(fake_registry, capsys):
    fake_registry(
        ("fake_built", ("fake-a",), make_fake_suite("fake_built", ("fake-a",))),
        ("fake_missing", ("fake-b",), None),
    )

    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[skip] fake_missing: not built" in out
    assert "fake_built probe" in out
    assert "small-serve gemma4-e2b@gpu-full" in out
    assert "small-serve e4b-ref@cpu" in out
    assert out.strip().splitlines()[-1] == "plan ok"


def test_plan_fails_when_a_built_suite_reports_an_input_problem(fake_registry, capsys):
    module = make_fake_suite("fake_built", ("fake-a",), problems=["lines.json: L99 missing"])
    fake_registry(("fake_built", ("fake-a",), module))

    code = speech_bench.main(["plan"])

    out = capsys.readouterr().out
    assert code != 0
    assert "L99 missing" in out
    assert "plan ok" not in out


@pytest.mark.parametrize("option", ["--results-root", "--run"])
@pytest.mark.parametrize("subcommand", ["plan", "selftest"])
def test_plan_and_selftest_take_neither_results_root_nor_run(subcommand, option):
    with pytest.raises(SystemExit) as info:
        speech_bench.main([subcommand, option, "x"])

    assert info.value.code == 2


def test_plan_refuses_a_host_1070_script_path_inside_the_repository(capsys):
    target = _REPO_ROOT / "h6_script.sh"

    code = speech_bench.main(["plan", "--host-1070-script", str(target)])

    assert code != 0
    assert "inside the repository" in capsys.readouterr().out
    assert not target.exists()


def test_plan_reports_the_host_1070_script_module_not_built(
    fake_registry, monkeypatch, tmp_path, capsys
):
    fake_registry()
    monkeypatch.setattr(speech_bench, "HOST1070_SCRIPT_MODULE", f"{FAKE_PKG}.host1070_script")

    code = speech_bench.main(["plan", "--host-1070-script", str(tmp_path / "h6.sh")])

    assert code != 0
    assert "host1070_script not built" in capsys.readouterr().out


def test_plan_calls_write_script_with_the_out_path_pins_and_repo_root(
    fake_registry, monkeypatch, tmp_path, capsys
):
    fake_registry()
    calls = {}
    module = types.ModuleType(f"{FAKE_PKG}.host1070_script")

    def write_script(out_path, *, pins_path, repo_root):
        calls.update(out_path=out_path, pins_path=pins_path, repo_root=repo_root)
        return [out_path, out_path.with_suffix(".json")]

    module.write_script = write_script
    monkeypatch.setitem(sys.modules, f"{FAKE_PKG}.host1070_script", module)
    monkeypatch.setattr(speech_bench, "HOST1070_SCRIPT_MODULE", f"{FAKE_PKG}.host1070_script")
    out_path = tmp_path / "h6.sh"
    pins = tmp_path / "pins.json"

    code = speech_bench.main(["plan", "--host-1070-script", str(out_path), "--pins", str(pins)])

    assert code == 0
    assert calls["out_path"] == out_path.resolve()
    assert calls["pins_path"] == pins.resolve()
    assert calls["repo_root"] == speech_bench.REPO_ROOT
    assert capsys.readouterr().out.strip().splitlines()[-1] == "plan ok"


def test_real_plan_ends_plan_ok(capsys):
    code = speech_bench.main(["plan"])

    assert code == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "plan ok"


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------


def test_selftest_runs_built_suites_skips_unbuilt_and_ends_selftest_ok(fake_registry, capsys):
    fake_registry(
        ("fake_built", ("fake-a",), make_fake_suite("fake_built", ("fake-a",))),
        ("fake_missing", ("fake-b",), None),
    )

    code = speech_bench.main(["selftest"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[ok] core" in out
    assert "[ok] fake_built" in out
    assert "[skip] fake_missing: not built" in out
    assert out.strip().splitlines()[-1] == "selftest ok"


def test_selftest_fails_when_a_built_suite_selftest_raises(fake_registry, capsys):
    module = make_fake_suite("fake_built", ("fake-a",), selftest_error="argv wrong")
    fake_registry(("fake_built", ("fake-a",), module))

    code = speech_bench.main(["selftest"])

    out = capsys.readouterr().out
    assert code != 0
    assert "argv wrong" in out
    assert "selftest ok" not in out


# ---------------------------------------------------------------------------
# small-serve and small-unserve
# ---------------------------------------------------------------------------


def test_small_serve_requires_results_root_and_run(monkeypatch):
    monkeypatch.delenv("MODEL_BENCH_RESULTS_ROOT", raising=False)

    with pytest.raises(SystemExit) as info:
        speech_bench.main(["small-serve", "--model", "gemma4-e2b", "--placement", "cpu"])

    assert info.value.code == 2


def test_small_serve_refuses_a_results_root_inside_the_repository(capsys):
    argv = ["small-serve", "--results-root", str(_REPO_ROOT / "out"), "--run", "mh1"]

    code = speech_bench.main([*argv, "--model", "gemma4-e2b", "--placement", "cpu"])

    assert code != 0
    assert "[FAIL]" in capsys.readouterr().out


@pytest.mark.parametrize("placement", ["gtx1070", "gtx1070-shared", "host1070-cpu"])
def test_small_serve_refuses_the_host_1070_placements(tmp_path, capsys, placement):
    argv = ["small-serve", "--results-root", str(tmp_path), "--run", "mh1", "--model", "gemma4-e2b"]

    code = speech_bench.main([*argv, "--placement", placement])

    assert code != 0
    assert "host-1070" in capsys.readouterr().out
    assert not (tmp_path / "mh1").exists()


# ---------------------------------------------------------------------------
# No option takes a voice sample
# ---------------------------------------------------------------------------

FORBIDDEN_OPTION_PARTS = (
    "reference",
    "ref-audio",
    "audio-prompt",
    "prompt-audio",
    "voice-file",
    "voice-path",
    "voice-sample",
    "voice-clip",
    "voice-dir",
    "clone",
    "speaker-wav",
)


# The owner's ruling of 12 Nov allows one cloned arm, `pocket-clone`, which clones a private
# identity clip named at run time and kept outside the checkout. So one option that takes a voice
# sample exists on the three subcommands that can run that arm: `--voice-clip`, on `tts`, `chain`
# and `reply-ab` (which voices both replies of an item with it). Plan v5 (M17) adds a second,
# `--voice-dir`, a whole identity directory with its trained weights, on `tts` alone. Any other
# option holding a listed part, or either option anywhere else, still fails.
ALLOWED_VOICE_SAMPLE_OPTIONS = {
    "tts": {"voice-clip", "voice-dir"},
    "chain": {"voice-clip"},
    "reply-ab": {"voice-clip"},
}


def _normalised(options):
    return {opt.lstrip("-").replace("_", "-").lower() for opt in options}


def _hits(options, allowed=frozenset()):
    return sorted(
        o
        for o in _normalised(options) - set(allowed)
        if any(part in o for part in FORBIDDEN_OPTION_PARTS)
    )


def test_no_option_takes_a_voice_sample_but_voice_clip_on_tts_chain_and_reply_ab():
    parser = speech_bench.build_parser()
    top = {opt for action in parser._actions for opt in action.option_strings}
    hits = [f"(top) {o}" for o in _hits(top)]
    subcommands = speech_bench.subcommand_parsers(parser)
    for name, sub in subcommands.items():
        allowed = ALLOWED_VOICE_SAMPLE_OPTIONS.get(name, set())
        hits += [f"{name} {o}" for o in _hits(speech_bench.all_option_strings(sub), allowed)]

    assert hits == []
    for name in ALLOWED_VOICE_SAMPLE_OPTIONS:
        assert "voice-clip" in _normalised(speech_bench.all_option_strings(subcommands[name]))
    with_voice_dir = [
        name
        for name, sub in subcommands.items()
        if "voice-dir" in _normalised(speech_bench.all_option_strings(sub))
    ]
    assert with_voice_dir == ["tts"]
    assert "host-1070-script" in _normalised(speech_bench.all_option_strings(parser))


def test_the_narrowed_check_still_fails_on_any_other_voice_sample_option():
    assert _hits({"--voice-clip"}) == ["voice-clip"]
    assert _hits({"--voice-dir"}) == ["voice-dir"]
    assert _hits({"--voice-dir", "--voice-clip"}, {"voice-clip"}) == ["voice-dir"]
    assert _hits({"--voice-clip", "--clone-from"}, {"voice-clip"}) == ["clone-from"]
    assert _hits({"--speaker-wav", "--voice-clip"}, {"voice-clip"}) == ["speaker-wav"]


# ---------------------------------------------------------------------------
# --voice-clip misuse exits 2 before any container starts
# ---------------------------------------------------------------------------

TRANSCRIPT = "the quartz heron counted nine amber lanterns"


def write_identity_clip(root: Path) -> Path:
    """A sine tone standing in for the identity clip, with a transcript beside it."""
    folder = root / "private-identity"
    folder.mkdir(parents=True)
    clip = folder / "identity.wav"
    samples = (round(8000 * math.sin(2 * math.pi * 220 * i / 24000)) for i in range(2400))
    with wave.open(str(clip), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"".join(s.to_bytes(2, "little", signed=True) for s in samples))
    (folder / "clip.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return clip


def _explode(*args, **kwargs):
    raise AssertionError("a container would have started")


class ExplodingRunner:
    """A container runner that may be built but fails the test if it runs anything."""

    run = staticmethod(_explode)


MISUSES = {
    "clone-without-clip": ("pocket-clone", None),
    "missing-clip": ("pocket-clone", "absent.wav"),
    "clip-with-another-engine": ("pocket", "identity.wav"),
}


def _misuse_argv(tmp_path, subcommand, case):
    clip = write_identity_clip(tmp_path)
    engine, clip_name = MISUSES[case]
    common = ["--results-root", str(tmp_path / "results"), "--run", "mh1"]
    common += ["--speech-models-dir", str(tmp_path)]
    if subcommand == "tts":
        argv = ["tts", *common, "--engine", engine]
    else:
        argv = ["chain", *common, "--model", "gemma4-e2b", "--placement", "gpu-full"]
        argv += ["--material-dir", str(tmp_path), "--tts-engine", engine]
    if clip_name is not None:
        argv += ["--voice-clip", str(clip.parent / clip_name)]
    return argv, clip


@pytest.mark.parametrize("case", sorted(MISUSES))
@pytest.mark.parametrize("subcommand", ["tts", "chain"])
def test_voice_clip_misuse_exits_2_before_any_container_or_write(
    tmp_path, monkeypatch, capsys, subcommand, case
):
    from scripts.model_bench.speech import chain, tts_suite

    monkeypatch.setattr(tts_suite, "DockerRunner", ExplodingRunner)
    monkeypatch.setattr(tts_suite.bench_host, "probe_container_state", _explode)
    monkeypatch.setattr(chain.ChainServices, "default", _explode)
    monkeypatch.setattr(chain.core, "HostDocker", _explode)
    argv, clip = _misuse_argv(tmp_path, subcommand, case)

    code = speech_bench.main(argv)

    captured = capsys.readouterr()
    assert code == 2
    assert "[REFUSED]" in captured.out
    for text in (captured.out, captured.err):
        for secret in (str(clip), clip.as_posix(), clip.parent.name, TRANSCRIPT):
            assert secret not in text
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("with_clip", [False, True])
def test_tts_without_an_engine_runs_pocket_clone_only_with_a_clip(
    tmp_path, monkeypatch, capsys, with_clip
):
    from scripts.model_bench.speech import core, tts_suite

    seen = {}

    def run_tts(request, **kwargs):
        seen["request"] = request
        return tmp_path / "tts"

    monkeypatch.setattr(tts_suite, "run_tts", run_tts)
    clip = write_identity_clip(tmp_path)
    argv = ["tts", "--results-root", str(tmp_path / "results"), "--run", "mh1"]
    argv += ["--speech-models-dir", str(tmp_path)]
    if with_clip:
        argv += ["--voice-clip", str(clip)]

    code = speech_bench.main(argv)

    assert code == 0
    request = seen["request"]
    arms = list(core.load_json(core.SPEECH_ARMS_PATH)["tts"])
    if with_clip:
        assert request.engines == tuple(arms)
        assert request.voice_clip == clip
    else:
        assert request.engines == tuple(a for a in arms if a != "pocket-clone")
        assert request.voice_clip is None
    assert str(clip.parent) not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --voice-dir: an identity directory on `tts` only, exclusive with --voice-clip
# ---------------------------------------------------------------------------


def write_voice_dir(root: Path) -> Path:
    """An identity directory under a private host folder name, with its manifest."""
    folder = root / "private-voice-folder"
    folder.mkdir(parents=True)
    (folder / "voice.json").write_text('{"schema": 1, "voice_id": "mist"}', encoding="utf-8")
    (folder / "neutral.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return folder


def _tts_argv(tmp_path: Path, *extra: str) -> list[str]:
    argv = ["tts", "--results-root", str(tmp_path / "results"), "--run", "mh1"]
    return [*argv, "--speech-models-dir", str(tmp_path), *extra]


def test_tts_refuses_voice_clip_and_voice_dir_together(tmp_path, monkeypatch, capsys):
    from scripts.model_bench.speech import tts_suite

    monkeypatch.setattr(tts_suite, "run_tts", _explode)
    clip = write_identity_clip(tmp_path)
    folder = write_voice_dir(tmp_path)

    with pytest.raises(SystemExit) as info:
        speech_bench.main(
            _tts_argv(tmp_path, "--voice-clip", str(clip), "--voice-dir", str(folder))
        )

    assert info.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("case", ["another-engine", "missing-dir"])
def test_voice_dir_misuse_exits_2_before_any_container_or_write(
    tmp_path, monkeypatch, capsys, case
):
    from scripts.model_bench.speech import tts_suite

    monkeypatch.setattr(tts_suite, "DockerRunner", ExplodingRunner)
    monkeypatch.setattr(tts_suite.bench_host, "probe_container_state", _explode)
    folder = write_voice_dir(tmp_path)
    engine = "pocket" if case == "another-engine" else "pocket-clone"
    target = folder if case == "another-engine" else folder.parent / "absent-folder"

    code = speech_bench.main(_tts_argv(tmp_path, "--engine", engine, "--voice-dir", str(target)))

    captured = capsys.readouterr()
    assert code == 2
    assert "[REFUSED]" in captured.out
    for text in (captured.out, captured.err):
        for secret in (str(target), target.as_posix(), folder.name, "absent-folder", TRANSCRIPT):
            assert secret not in text
    assert not (tmp_path / "results").exists()


def test_tts_without_an_engine_runs_pocket_clone_with_an_identity(tmp_path, monkeypatch):
    from scripts.model_bench.speech import core, tts_suite

    seen = {}

    def run_tts(request, **kwargs):
        seen["request"] = request
        return tmp_path / "tts"

    monkeypatch.setattr(tts_suite, "run_tts", run_tts)
    folder = write_voice_dir(tmp_path)

    code = speech_bench.main(_tts_argv(tmp_path, "--voice-dir", str(folder)))

    assert code == 0
    request = seen["request"]
    assert request.engines == tuple(core.load_json(core.SPEECH_ARMS_PATH)["tts"])
    assert (request.voice_dir, request.voice_clip) == (folder, None)
    assert str(folder) not in repr(request)
