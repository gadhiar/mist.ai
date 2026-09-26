"""Reasoning flags on the mist-llm service in docker-compose.yml (MIS-171 decision 6).

Hermetic: parses the committed compose file only. Tool-dispatch turns send a
per-request `reasoning_budget_tokens` (`backend/llm/adaptive_thinking.py`). On the
pinned b11151 build a request value of -1 falls back to the server's
`--reasoning-budget`, so the server must run uncapped, and reasoning must be
extracted into `reasoning_content` rather than left in `content`
(`tests/unit/model_bench/fixtures/host/llama_server_help_b11151.txt:628-646`).

Kept separate from `test_compose_pins.py`, which pins the image digest and the
prompt-cache cap.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"


def _mist_llm() -> dict:
    with COMPOSE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["services"]["mist-llm"]


def _command() -> list[str]:
    return [str(arg) for arg in _mist_llm()["command"]]


def _env_keys() -> set[str]:
    return {str(entry).partition("=")[0] for entry in _mist_llm().get("environment", [])}


def test_mist_llm_extracts_reasoning_in_deepseek_format() -> None:
    command = _command()
    assert "--reasoning-format" in command
    assert command[command.index("--reasoning-format") + 1] == "deepseek"


def test_mist_llm_has_no_server_reasoning_budget_cap() -> None:
    assert "--reasoning-budget" not in _command()
    assert "LLAMA_ARG_THINK_BUDGET" not in _env_keys()
