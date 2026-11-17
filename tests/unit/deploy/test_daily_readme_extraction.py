"""The daily instance README's account of night extraction, `deploy/daily/README.md`.

Hermetic: reads the README, the compose file and the source it describes; no docker, no network.
Raj operates the instance from this README, so each test pins one statement about night
extraction to the code that makes it true, or pins that a statement from before night extraction
ran in the daily project (an "extraction host") is gone.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_DIR = REPO_ROOT / "deploy" / "daily"
README_PATH = DAILY_DIR / "README.md"
COMPOSE_PATH = DAILY_DIR / "compose.yml"
INSTANCE_PATH = REPO_ROOT / "scripts" / "daily" / "instance.py"
HEALTH_PATH = REPO_ROOT / "backend" / "health.py"
HOLD_PATH = REPO_ROOT / "backend" / "extraction_hold.py"
REGISTER_PATH = DAILY_DIR / "windows" / "register-signin-task.ps1"
UNREGISTER_PATH = DAILY_DIR / "windows" / "unregister-signin-task.ps1"
CUTOVER_PATH = REPO_ROOT / "backend" / "extraction_backlog" / "CUTOVER.md"

EXTRACTION_TASK = "MIST extraction window"


def _text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _section(title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", _text(), re.M | re.S)
    assert match is not None, f"README.md has no '## {title}' section"
    return match.group(1)


def _flat(title: str) -> str:
    """The section with its whitespace collapsed, so a phrase may wrap across lines."""
    return " ".join(_section(title).split())


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _health_row(check: str, reason: str) -> list[str]:
    """The cells of the one "Reading /health" row naming `check` and `reason`."""
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in _section("Reading /health").splitlines()
        if line.startswith("| `")
    ]
    matches = [row for row in rows if f"`{check}`" in row[0] and row[1] == f"`{reason}`"]
    assert len(matches) == 1, (check, reason, matches)
    return matches[0]


# --- Night extraction section -----------------------------------------------------------------


class TestNightExtractionSection:
    def test_window_variable_format_default_and_clock(self):
        section = _flat("Night extraction")
        source = _source(INSTANCE_PATH)
        default = re.search(r'^DEFAULT_EXTRACTION_WINDOW = "([^"]+)"', source, re.M)
        assert default is not None
        assert "`MIST_EXTRACTION_WINDOW`" in section
        assert "`HH:MM-HH:MM`" in section
        assert f"`{default.group(1)}`" in section
        assert "host's local time" in section
        assert "crosses midnight" in section
        assert "equal ends" in section

    def test_modes_where_they_are_kept_and_what_init_writes(self):
        section = _flat("Night extraction")
        source = _source(INSTANCE_PATH)
        assert 'EXTRACTION_MODES = ("on", "off", "auto")' in source
        assert 'EXTRACTION_MODE_FILE = "extraction-mode"' in source
        assert 'INIT_EXTRACTION_MODE = "off"' in source
        assert 'DEFAULT_EXTRACTION_MODE = "auto"' in source
        for mode in ("on", "off", "auto"):
            assert f"`{mode}`" in section, mode
        assert "`ops/extraction-mode`" in section
        assert "`init` writes `off`" in section
        assert "no file reads as `auto`" in section

    def test_identity_variables_and_the_unconfigured_refusal(self):
        section = _flat("Night extraction")
        assert (
            'EXTRACTION_IDENTITY_VARIABLES = ("EXTRACTION_MODEL_HASH", "EXTRACTION_LLM_DEVICE")'
        ) in _source(INSTANCE_PATH)
        assert "`EXTRACTION_MODEL_HASH`" in section
        assert "`EXTRACTION_LLM_DEVICE`" in section
        assert "`unconfigured`" in section
        assert "exits 2" in section

    def test_apply_starts_nothing_while_the_backend_is_not_running(self):
        section = _flat("Night extraction")
        assert "is not running; " in _source(INSTANCE_PATH)
        assert "While `mist-backend` is not running it starts nothing" in section
        assert "writes no marker and exits 0" in section

    def test_paused_marker_and_logs(self):
        section = _flat("Night extraction")
        assert '"extraction-paused"' in _source(HOLD_PATH)
        for reason in ("window", "mode", "gaming", "starting", "unconfigured"):
            assert f'"{reason}"' in _source(HOLD_PATH), reason
            assert f"`{reason}`" in section, reason
        assert "`ops/extraction-paused`" in section
        assert "`logs/extraction-window.log`" in section
        assert "`logs/extraction-last-<service>.log`" in section

    def test_the_task_its_switch_and_its_tick(self):
        section = _flat("Night extraction")
        register = _source(REGISTER_PATH)
        assert f"-TaskName '{EXTRACTION_TASK}'" in register
        assert "-RepetitionInterval (New-TimeSpan -Minutes 5)" in register
        assert f'"{EXTRACTION_TASK}"' in section
        assert "`register-signin-task.ps1`" in section and "`-NightExtraction`" in section
        assert "every 5 minutes" in section
        assert "at sign-in" in section

    def test_gaming_stops_extraction_and_off_applies_again(self):
        section = _flat("Night extraction")
        assert "`gaming-mode on` stops extraction too" in section
        assert "`gaming-mode off`" in section and "runs `extraction apply`" in section

    def test_status_and_turning_extraction_off(self):
        section = _flat("Night extraction")
        assert "`extraction status`" in section
        assert "`extraction off`" in section
        assert "### Turning extraction off" in _section("Night extraction")

    def test_dc_is_the_compose_prefix_with_the_extraction_profile(self):
        section = _flat("Night extraction")
        assert "`<dc>`" in section
        assert "--profile extraction" in section


# --- Reading /health --------------------------------------------------------------------------


class TestReadingHealth:
    @pytest.mark.parametrize(
        ("reason", "status"),
        [("extraction_paused", "paused"), ("gpu_yielded", "yielded")],
    )
    def test_extraction_service_rows(self, reason, status):
        assert f'"{reason}"' in _source(HEALTH_PATH)
        row = _health_row("extraction_service", reason)
        assert row[2] == status

    def test_disabled_by_config_points_at_extraction_status(self):
        remedy = _health_row("extraction_service", "disabled_by_config")[4]
        assert "`extraction status`" in remedy
        assert "Compose sets" in remedy

    def test_unreachable_reads_status_and_apply(self):
        remedy = _health_row("extraction_service", "unreachable")[4]
        assert "`extraction status`" in remedy
        assert "`extraction apply`" in remedy

    def test_service_degraded_reads_the_llm_logs(self):
        remedy = _health_row("extraction_service", "service_degraded")[4]
        assert "`<dc> logs mist-extraction-llm`" in remedy
        assert "`<root>/logs/extraction-last-mist-extraction-llm.log`" in remedy

    def test_epoch_mismatch_names_the_night_setting_and_cutover_section_9(self):
        assert "## 9. Settings changes" in _source(CUTOVER_PATH)
        remedy = _health_row("extraction_service", "epoch_mismatch")[4]
        assert "night setting" in remedy
        assert "`backend/extraction_backlog/CUTOVER.md` section 9" in remedy

    def test_backlog_overdue_gives_the_daily_default(self):
        assert "${MIST_HEALTH_BACKLOG_MAX_AGE_S:-86400}" in _source(COMPOSE_PATH)
        row = _health_row("extraction_backlog", "backlog_overdue")
        assert "86,400 s (24 h)" in row[3]
        assert "`extraction status`" in row[4]


# --- Commands, tasks and rollback -------------------------------------------------------------


def test_commands_section_lists_the_extraction_commands():
    section = _section("Commands").replace("\\|", "|")
    assert 'choices=(*EXTRACTION_MODES, "apply", "status")' in _source(INSTANCE_PATH)
    assert "`extraction on|off|auto`" in section
    assert "`extraction apply`" in section
    assert "`extraction status`" in section


def test_gaming_watch_unregister_removes_all_three_tasks():
    assert f"'{EXTRACTION_TASK}'" in _source(UNREGISTER_PATH)
    assert "`unregister-signin-task.ps1` removes all three tasks when present" in _flat(
        "Gaming mode"
    )


def test_registration_usage_names_night_extraction_beside_lock_at_sign_in():
    usage = [
        line for line in _section("Start after power-on").splitlines() if "-Root <root>" in line
    ]
    assert len(usage) == 1, usage
    assert "[-LockAtSignIn]" in usage[0] and "[-NightExtraction]" in usage[0]


def test_start_after_power_on_unregister_and_reregister_after_rollback():
    section = _flat("Start after power-on")
    assert "`unregister-signin-task.ps1` removes all three tasks when present" in section
    assert (
        "after a `switch` to an older release, register again with that release's "
        "`register-signin-task.ps1`, without `-NightExtraction`"
    ) in section


def test_release_lifecycle_rollback_order_for_a_release_without_night_extraction():
    section = _flat("Release lifecycle")
    order = section[section.index("To roll back to a release without night extraction") :]
    at = 0
    for step in (
        "`extraction off`",
        "`unregister-signin-task.ps1`",
        "`switch <previous tag>`",
        "that release's `register-signin-task.ps1`",
        "without `-NightExtraction`",
    ):
        found = order.find(step, at)
        assert found >= 0, f"{step} missing or out of order"
        at = found + len(step)


# --- Memory limits ----------------------------------------------------------------------------


def test_memory_limits_include_the_extraction_services_as_estimates():
    services = yaml.safe_load(_source(COMPOSE_PATH))["services"]
    section = _flat("Memory limits")
    for service in ("mist-extraction-llm", "mist-extraction"):
        default = re.fullmatch(r"\$\{[A-Z_]+:-(\w+)\}", services[service]["mem_limit"])
        assert default is not None, service
        assert f"{default.group(1)} `{service}`" in section, service
    assert "The six limits" in section
    assert "The four limits" not in section
    assert "The extraction services' limits are estimates" in section
    assert "no measured peak" in section


# --- What is gone ----------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["extraction host", "default 6 h", "removes both tasks"])
def test_no_longer_says(phrase):
    assert phrase not in _text().lower()


def test_gaming_mode_says_extraction_runs_on_this_gpu_at_night():
    assert "runs on this GPU at night, in the daily project" in _flat("Gaming mode")
