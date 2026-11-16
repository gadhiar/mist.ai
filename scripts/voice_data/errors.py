"""Refusals the voice-data package raises, one class per CLI exit code.

The base derives from `RuntimeError`, as `BenchHostError` does in
`scripts/model_bench/bench_host.py`, so the CLI catches the package's own refusals by class and
lets any other exception surface as a real bug with its traceback.
"""


class VoiceDataError(RuntimeError):
    """Base class for every refusal this package raises."""


class RootError(VoiceDataError):
    """The root is unset, unmarked, inside a git work tree, or a path escapes it (exit 2)."""


class InputError(VoiceDataError):
    """An input is missing or malformed: a record, the gate overlay, a file (exit 2)."""


class GateError(VoiceDataError):
    """A gate failed; its report is written before this is raised (exit 3)."""


class JobError(VoiceDataError):
    """A job failed: a non-zero container or remote exit, or a manifest mismatch (exit 4)."""


class BudgetError(VoiceDataError):
    """The budget ceiling would be exceeded (exit 5)."""
