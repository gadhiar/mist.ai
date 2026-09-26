"""Shared nvidia-smi CSV query and parser.

Used by bench_host.py's continuous 5 Hz sampler (vram.csv, `run`), the
bounded-duration sampler (`vram-step`, `voice`), and by the unit tests
against `tests/unit/model_bench/fixtures/host/nvidia_smi_sample.csv`.

The query field order here is exactly the order of vram.csv's columns after
`t_unix` (bench_host.py adds `t_unix` itself; it is not an nvidia-smi field).
"""

from __future__ import annotations

# Order matches vram.csv's header minus the leading `t_unix` column, which
# bench_host.py stamps itself at sample time (nvidia-smi has no equivalent
# --query-gpu field for "when did I run").
QUERY_FIELDS: tuple[str, ...] = (
    "memory.used",
    "memory.total",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "clocks_throttle_reasons.active",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)

# The row keys these fields parse into, in the same order, matching
# vram.csv's header (`t_unix` prepended by the caller).
ROW_KEYS: tuple[str, ...] = (
    "memory_used_mib",
    "memory_total_mib",
    "power_w",
    "sm_mhz",
    "mem_mhz",
    "temp_c",
    "throttle_reasons",
    "pcie_gen",
    "pcie_width",
)

VRAM_CSV_HEADER: tuple[str, ...] = ("t_unix",) + ROW_KEYS

# Columns that parse as int, float, or stay str (throttle_reasons is a hex
# bitmask string, e.g. "0x0000000000000000", not a number).
_INT_FIELDS = frozenset(
    {"memory_used_mib", "memory_total_mib", "sm_mhz", "mem_mhz", "temp_c", "pcie_gen", "pcie_width"}
)
_FLOAT_FIELDS = frozenset({"power_w"})


class NvidiaSmiParseError(ValueError):
    """A CSV line from nvidia-smi did not have the expected field count."""


def build_query_args(*, nvidia_smi_bin: str = "nvidia-smi", interval_ms: int = 200) -> list[str]:
    """Build the argv for a long-running, self-repeating nvidia-smi query.

    `-lms <interval_ms>` makes nvidia-smi print one CSV row every
    `interval_ms` milliseconds until the process is terminated -- the 5 Hz
    (200 ms) sampler bench_host.py's `run` subcommand keeps running for the
    duration of a suite.
    """
    return [
        nvidia_smi_bin,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-lms",
        str(interval_ms),
    ]


def _parse_field(key: str, raw: str) -> int | float | str | None:
    value = raw.strip()
    if value in ("[N/A]", "N/A", ""):
        return None
    if key in _INT_FIELDS:
        try:
            return int(float(value))
        except ValueError:
            return None
    if key in _FLOAT_FIELDS:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def parse_csv_line(line: str) -> dict[str, int | float | str | None]:
    """Parse one `--format=csv,noheader,nounits` line into a row dict.

    Keys match `ROW_KEYS` (i.e. vram.csv's columns minus `t_unix`).
    `[N/A]` values (nvidia-smi's marker for an unsupported/unavailable
    field, e.g. `clocks_throttle_reasons.active` on some driver/GPU
    combinations) parse to `None` rather than raising.

    Raises:
        NvidiaSmiParseError: If the line does not split into exactly
            `len(ROW_KEYS)` comma-separated fields.
    """
    stripped = line.strip()
    parts = [p.strip() for p in stripped.split(",")]
    if len(parts) != len(ROW_KEYS):
        raise NvidiaSmiParseError(
            f"expected {len(ROW_KEYS)} fields, got {len(parts)} from line: {line!r}"
        )
    return {key: _parse_field(key, raw) for key, raw in zip(ROW_KEYS, parts, strict=True)}


def summarize_rows(rows: list[dict]) -> dict:
    """Reduce sampled rows to the `session/vram_steps.json` step summary shape.

    Returns `{"median_mib", "max_mib", "total_mib", "samples"}`. Rows whose
    `memory_used_mib` parsed to `None` are excluded from the median/max but
    still counted in `samples` (the raw row count actually collected).
    `total_mib` is the most recent non-None `memory_total_mib`, or `None` if
    every row lacked one.
    """
    used = [r["memory_used_mib"] for r in rows if r.get("memory_used_mib") is not None]
    totals = [r["memory_total_mib"] for r in rows if r.get("memory_total_mib") is not None]
    used_sorted = sorted(used)
    n = len(used_sorted)
    if n == 0:
        median = None
    elif n % 2 == 1:
        median = used_sorted[n // 2]
    else:
        median = (used_sorted[n // 2 - 1] + used_sorted[n // 2]) / 2
    return {
        "median_mib": median,
        "max_mib": max(used_sorted) if used_sorted else None,
        "total_mib": totals[-1] if totals else None,
        "samples": len(rows),
    }
