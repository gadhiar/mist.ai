"""Speech bench driver for the mist-model-host goal (plan v2).

Invocation, from the repository root:

    python -m scripts.model_bench.speech_bench <subcommand> [--results-root <dir> --run <id>]
        [--bench-run <id>] [options]

This module holds the parser and the suite registry. Each suite is its own module under
`scripts.model_bench.speech`, exposing `register(subparsers)`, `container_argv(ctx)`,
`validate_inputs(repo_root)` and `selftest()`. The registry imports each lazily; a suite whose
module does not exist yet is "not built": its subcommands are stubs that refuse, `plan` skips its
inputs and argv, and `selftest` skips its checks. Only a `ModuleNotFoundError` for the suite
module itself counts as not built, so a missing dependency inside a built suite still fails loudly.

`plan` and `selftest` write nothing and take neither `--results-root` nor `--run`; every
subcommand that writes takes both. Stdlib only, like `bench_host.py`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.model_bench import bench_host
    from scripts.model_bench.speech import core
else:
    from . import bench_host
    from .speech import core

REPO_ROOT = core.REPO_ROOT
SPEECH_PACKAGE = "scripts.model_bench.speech"
HOST1070_SCRIPT_MODULE = f"{SPEECH_PACKAGE}.host1070_script"
SMALL_MODELS_DIR_ENV = "MODEL_BENCH_SMALL_MODELS_DIR"


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """A suite module and the subcommands it registers."""

    module: str
    subcommands: tuple[str, ...]
    package: str = SPEECH_PACKAGE

    @property
    def qualname(self) -> str:
        """The module's full import name."""
        return f"{self.package}.{self.module}"


SUITE_REGISTRY: tuple[SuiteSpec, ...] = (
    SuiteSpec("tts_suite", ("tts",)),
    SuiteSpec("stt_suite", ("stt", "prepare")),
    SuiteSpec("turn_suite", ("turn",)),
    SuiteSpec("cache_suite", ("cache",)),
    SuiteSpec("fit_suite", ("fit",)),
    SuiteSpec("listening", ("listen-pack",)),
    SuiteSpec("lowtier_suite", ("lowtier", "remote-memory")),
    SuiteSpec("chain", ("chain",)),
    SuiteSpec("game_suite", ("game-session",)),
    SuiteSpec("report", ("report",)),
)


@dataclass(frozen=True, slots=True)
class LoadedSuite:
    """A registry entry with its module, or `None` when the module is not built yet."""

    spec: SuiteSpec
    module: ModuleType | None


def import_optional(name: str) -> ModuleType | None:
    """Import `name`, or return `None` when that very module does not exist."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


def load_suites(registry: tuple[SuiteSpec, ...] | None = None) -> list[LoadedSuite]:
    specs = SUITE_REGISTRY if registry is None else registry
    return [LoadedSuite(spec, import_optional(spec.qualname)) for spec in specs]


# ---------------------------------------------------------------------------
# Subcommands owned by the core
# ---------------------------------------------------------------------------


def _not_built(subcommand: str, module: str):
    def run(args: argparse.Namespace) -> int:
        print(f"[FAIL] {subcommand}: {module} not built")
        return 1

    return run


def cmd_plan(args: argparse.Namespace) -> int:
    script_path = None
    if args.host_1070_script is not None:
        try:
            script_path = core.refuse_path_inside_repo(
                args.host_1070_script, what="--host-1070-script", repo_root=REPO_ROOT
            )
        except core.RepoPathError as exc:
            print(f"[FAIL] {exc}")
            return 1

    problems = [f"core: {p}" for p in core.validate_inputs(REPO_ROOT)]
    ctx = core.build_plan_context(REPO_ROOT)
    blocks = list(core.container_argv(ctx))
    for suite in args.loaded_suites:
        if suite.module is None:
            print(f"[skip] {suite.spec.module}: not built")
            continue
        problems += [f"{suite.spec.module}: {p}" for p in suite.module.validate_inputs(REPO_ROOT)]
        blocks += suite.module.container_argv(ctx)

    for label, argv in blocks:
        print(f"[argv] {label}: {shlex.join(argv)}")
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    if script_path is not None:
        module = import_optional(HOST1070_SCRIPT_MODULE)
        if module is None:
            print(f"[FAIL] host-1070 script: {HOST1070_SCRIPT_MODULE.rsplit('.', 1)[-1]} not built")
            return 1
        pins = args.pins.resolve() if args.pins is not None else None
        for path in module.write_script(script_path, pins_path=pins, repo_root=REPO_ROOT):
            print(f"[ok] wrote {path}")

    print("plan ok")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    checks = [("core", core.selftest)]
    for suite in args.loaded_suites:
        if suite.module is None:
            checks.append((suite.spec.module, None))
        else:
            checks.append((suite.spec.module, suite.module.selftest))

    failures = 0
    for name, check in checks:
        if check is None:
            print(f"[skip] {name}: not built")
            continue
        try:
            check()
        except (AssertionError, bench_host.BenchHostError) as exc:
            print(f"[FAIL] {name}: {exc}")
            failures += 1
        else:
            print(f"[ok] {name}")
    if failures:
        return 1
    print("selftest ok")
    return 0


def _small_out_dir(args: argparse.Namespace) -> tuple[Path, Path]:
    speech = core.speech_root(args.results_root, args.run, repo_root=REPO_ROOT)
    return speech, core.small_serve_dir(speech, args.model, args.placement, args.tag)


def _dir_arg(value: Path | None, env: str) -> Path | None:
    if value is not None:
        return value
    raw = os.environ.get(env)
    return Path(raw) if raw else None


def cmd_small_serve(args: argparse.Namespace) -> int:
    speech, out_dir = _small_out_dir(args)
    result = core.serve_small(
        args.model,
        args.placement,
        out_dir=out_dir,
        pins_path=args.pins if args.pins is not None else speech / "host" / "pins.json",
        models_dirs={
            "main": _dir_arg(args.models_dir, "MODELS_DIR"),
            "small": _dir_arg(args.small_models_dir, SMALL_MODELS_DIR_ENV),
        },
        ngl=args.ngl,
        health_timeout=args.timeout,
    )
    print(
        f"serving {result.alias} as {core.SMALL_CONTAINER} on {result.base_url}; "
        f"load {result.load_s:.2f} s; record at {out_dir / 'serve.json'}"
    )
    return 0


def cmd_small_unserve(args: argparse.Namespace) -> int:
    _, out_dir = _small_out_dir(args)
    unload_s = core.unserve_small(out_dir)
    print(f"unserved {core.SMALL_CONTAINER} in {unload_s:.2f} s; log at {out_dir / 'server.log'}")
    return 0


def _add_small_target_args(parser: argparse.ArgumentParser) -> None:
    core.add_results_args(parser)
    parser.add_argument("--model", required=True, help="a model id from small_arms.json")
    parser.add_argument(
        "--placement",
        required=True,
        help=f"one of {', '.join(core.LOCAL_PLACEMENTS)} (host-1070 placements are refused)",
    )
    parser.add_argument("--tag", help="suffix for the serve's directory, to serve a pair again")


def _register_core(subparsers: argparse._SubParsersAction) -> None:
    plan = subparsers.add_parser("plan", help="validate inputs and print every container argv")
    plan.add_argument(
        "--host-1070-script",
        type=Path,
        help="also write the host-1070 bash script here (outside the repository)",
    )
    plan.add_argument("--pins", type=Path, help="pin file the host-1070 script reads")
    plan.set_defaults(func=cmd_plan)

    selftest = subparsers.add_parser("selftest", help="run every built suite's self-checks")
    selftest.set_defaults(func=cmd_selftest)

    serve = subparsers.add_parser("small-serve", help="serve a small model as mist-bench-small")
    _add_small_target_args(serve)
    serve.add_argument("--ngl", type=int, help="layer count for gpu-4g and gpu-2g")
    serve.add_argument("--pins", type=Path, help="pin file (default <speech>/host/pins.json)")
    serve.add_argument("--models-dir", type=Path, help="main models directory (env MODELS_DIR)")
    serve.add_argument(
        "--small-models-dir",
        type=Path,
        help=f"small models directory (env {SMALL_MODELS_DIR_ENV})",
    )
    serve.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for /health")
    serve.set_defaults(func=cmd_small_serve)

    unserve = subparsers.add_parser("small-unserve", help="stop and remove mist-bench-small")
    _add_small_target_args(unserve)
    unserve.set_defaults(func=cmd_small_unserve)


def build_parser(suites: list[LoadedSuite] | None = None) -> argparse.ArgumentParser:
    loaded = load_suites() if suites is None else suites
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.speech_bench",
        description="Speech bench for the mist-model-host goal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<subcommand>")
    _register_core(subparsers)
    for suite in loaded:
        if suite.module is not None:
            suite.module.register(subparsers)
            continue
        for name in suite.spec.subcommands:
            stub = subparsers.add_parser(name, help=f"{suite.spec.module} not built")
            stub.set_defaults(func=_not_built(name, suite.spec.module), accepts_any=True)
    return parser


def subcommand_parsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """The parser's subcommands by name (argparse keeps them on a private action)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def all_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Every option string of `parser` and of its subcommands, recursively."""
    options = {opt for action in parser._actions for opt in action.option_strings}
    for child in subcommand_parsers(parser).values():
        options |= all_option_strings(child)
    return options


def main(argv: list[str] | None = None) -> int:
    suites = load_suites()
    parser = build_parser(suites)
    args, extra = parser.parse_known_args(argv)
    if extra and not getattr(args, "accepts_any", False):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.loaded_suites = suites
    try:
        return int(args.func(args))
    except bench_host.BenchHostError as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
