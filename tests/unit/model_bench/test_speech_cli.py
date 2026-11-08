"""scripts/model_bench/speech_bench.py: the parser, the suite registry, `plan` and `selftest`.

Registry behaviour is tested on fake suite modules patched into the registry, never by asserting
that a real later suite is unbuilt: tasks T2 to T8 build those modules, and these tests must keep
passing when they land.
"""

from __future__ import annotations

import sys
import types
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
        ("listening", ("listen-pack",)),
        ("lowtier_suite", ("lowtier", "remote-memory")),
        ("chain", ("chain",)),
        ("game_suite", ("game-session",)),
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
    "clone",
    "speaker-wav",
)


def test_no_registered_subcommand_has_an_option_that_takes_a_voice_sample():
    options = speech_bench.all_option_strings(speech_bench.build_parser())

    normalised = {opt.lstrip("-").replace("_", "-").lower() for opt in options}
    hits = sorted(o for o in normalised for part in FORBIDDEN_OPTION_PARTS if part in o)
    assert hits == []
    assert "host-1070-script" in normalised
