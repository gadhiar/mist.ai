"""Tests for the "MIST extraction window" task, which `register-signin-task.ps1` registers and
`unregister-signin-task.ps1` removes.

The task runs `instance.py extraction apply` every 5 minutes, so the night extraction window is
enforced by level-triggered ticks rather than by one start and one stop. The runner has no
PowerShell, so the script is checked by parsing its text, with the helpers
`tests/unit/daily/test_gamewatch.py` uses for the watch task.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WINDOWS_DIR = REPO_ROOT / "deploy" / "daily" / "windows"
REGISTER = WINDOWS_DIR / "register-signin-task.ps1"
UNREGISTER = WINDOWS_DIR / "unregister-signin-task.ps1"
NIGHT_TASK = "MIST extraction window"
INSTANCE_TASK = "MIST daily instance"
WATCH_TASK = "MIST gaming watch"


def ps_code(path: Path) -> str:
    """The script without comments, with each backtick line continuation joined."""
    text = re.sub(r"<#.*?#>", "", path.read_text(encoding="utf-8"), flags=re.S)
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return re.sub(r"`\n\s*", " ", "\n".join(lines))


def ps_header(path: Path) -> str:
    match = re.search(r"<#(.*?)#>", path.read_text(encoding="utf-8"), re.S)
    assert match is not None
    return match.group(1)


def ps_statements(code: str, command: str) -> list[str]:
    return [line for line in code.splitlines() if re.search(rf"(?<![\w-]){command}\b", line)]


def balanced_group(text: str, start: int) -> str:
    """The text from the bracket at `start` to its matching close, outside quoted strings."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    stack: list[str] = []
    quote: str | None = None
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            assert stack and stack.pop() == char
            if not stack:
                return text[start : index + 1]
    raise AssertionError(f"no close for the bracket at {start}")


def if_blocks(code: str, condition: str) -> list[tuple[int, str]]:
    """Where each `if (<condition>) { ... }` in `code` starts, and its body."""
    blocks = []
    for match in re.finditer(rf"if \({re.escape(condition)}\) ", code):
        blocks.append((match.start(), balanced_group(code, code.index("{", match.end() - 1))))
    return blocks


def registration(code: str, name: str) -> str:
    calls = [
        call
        for call in ps_statements(code, "Register-ScheduledTask")
        if f"-TaskName '{name}'" in call
    ]
    assert len(calls) == 1, calls
    return calls[0]


def night_block(code: str) -> str:
    """The one `if ($NightExtraction)` block that registers the task."""
    call = registration(code, NIGHT_TASK)
    blocks = [body for _at, body in if_blocks(code, "$NightExtraction") if call in body]
    assert len(blocks) == 1
    return blocks[0]


def assignment(code: str, variable: str) -> str:
    lines = [
        line.strip() for line in code.splitlines() if line.lstrip().startswith(f"{variable} =")
    ]
    assert len(lines) == 1, (variable, lines)
    return lines[0]


def test_register_script_takes_the_night_extraction_switch():
    code = ps_code(REGISTER)
    param = balanced_group(code, code.index("param("))
    assert re.search(r"\[switch\]\s*\$NightExtraction\b", param)


def test_the_night_task_is_registered_only_inside_the_switch_block():
    code = ps_code(REGISTER)
    block = night_block(code)
    assert len(ps_statements(block, "Register-ScheduledTask")) == 1
    assert f"schtasks.exe /query /tn '{NIGHT_TASK}'" in block


def test_the_night_task_runs_pythonw_extraction_apply_from_the_roots_bin():
    code = ps_code(REGISTER)
    block = night_block(code)
    assert (
        '$ApplyArgument = \'"{0}/bin/instance.py" extraction apply --root "{0}"\' -f $RootPath'
        in block
    )
    action = assignment(block, "$ApplyAction")
    assert action == (
        "$ApplyAction = New-ScheduledTaskAction -Execute $PythonwPath "
        "-Argument $ApplyArgument -WorkingDirectory $RootPath"
    )
    assert "-Action $ApplyAction" in registration(code, NIGHT_TASK)


def test_the_night_task_ticks_every_5_minutes_from_midnight_and_at_sign_in():
    code = ps_code(REGISTER)
    block = night_block(code)
    assert assignment(block, "$Tick") == "$Tick = New-ScheduledTaskTrigger -Daily -At '00:00'"
    # New-ScheduledTaskTrigger -Daily takes no repetition, so a -Once trigger's is copied.
    assert assignment(block, "$Tick.Repetition") == (
        "$Tick.Repetition = (New-ScheduledTaskTrigger -Once -At '00:00' "
        "-RepetitionInterval (New-TimeSpan -Minutes 5) "
        "-RepetitionDuration (New-TimeSpan -Days 1)).Repetition"
    )
    assert "-Trigger @($Tick, $Trigger)" in registration(code, NIGHT_TASK)
    # $Trigger is the sign-in trigger the other two tasks use.
    assert (
        assignment(code, "$Trigger") == "$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User"
    )


def test_the_night_task_settings_and_principal():
    code = ps_code(REGISTER)
    call = registration(code, NIGHT_TASK)
    assert "-Principal $Principal" in call
    assert assignment(code, "$Principal") == (
        "$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive "
        "-RunLevel Limited"
    )
    settings_at = call.index("-Settings (New-ScheduledTaskSettingsSet")
    settings = balanced_group(call, call.index("(", settings_at))
    for setting in (
        "-StartWhenAvailable",
        "-MultipleInstances IgnoreNew",
        "-ExecutionTimeLimit (New-TimeSpan -Minutes 15)",
        "-AllowStartIfOnBatteries",
        "-DontStopIfGoingOnBatteries",
    ):
        assert setting in settings, setting
    assert "-Force" in call


def test_pythonw_absence_refuses_the_night_task_before_any_task():
    code = ps_code(REGISTER)
    checks = [
        (at, body)
        for at, body in if_blocks(code, "$NightExtraction")
        if "Test-Path -LiteralPath $PythonwPath -PathType Leaf" in body
    ]
    assert len(checks) == 1
    at, body = checks[0]
    assert "throw" in body
    assert at < code.index("Register-ScheduledTask")


def test_without_the_switch_nothing_about_the_other_tasks_changes():
    code = ps_code(REGISTER)
    param = balanced_group(code, code.index("param("))
    outside = code.replace(param, "")
    blocks = if_blocks(code, "$NightExtraction")
    assert len(blocks) == 2
    for _at, body in blocks:
        outside = outside.replace(f"if ($NightExtraction) {body}", "")
    # Outside the switch's two blocks the script never names the new task or its parts.
    for name in ("NightExtraction", NIGHT_TASK, "$Apply", "$Tick", "extraction"):
        assert name not in outside, name
    for name in (INSTANCE_TASK, WATCH_TASK):
        call = registration(code, name)
        assert "-Trigger $Trigger" in call
        assert (
            "-Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) "
            "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries)"
        ) in call


def test_unregister_script_removes_the_night_task_only_if_present():
    code = ps_code(UNREGISTER)
    removal = f"Unregister-ScheduledTask -TaskName '{NIGHT_TASK}' -Confirm:$false"
    assert sum(removal in line for line in ps_statements(code, "Unregister-ScheduledTask")) == 1
    lookup = f"Get-ScheduledTask -TaskName '{NIGHT_TASK}' -ErrorAction SilentlyContinue"
    variable = re.search(rf"(\$\w+) = {re.escape(lookup)}", code)
    assert variable is not None
    # The removal is the else branch of the check that the lookup found nothing.
    check_at = code.index(f"if ($null -eq {variable.group(1)}) ", variable.end())
    then_block = balanced_group(code, code.index("{", check_at))
    rest = code[code.index(then_block, check_at) + len(then_block) :]
    assert rest.lstrip().startswith("else")
    else_block = balanced_group(rest, rest.index("{"))
    assert removal in else_block
    assert removal not in then_block
    assert NIGHT_TASK in ps_header(UNREGISTER)


def test_register_script_header_documents_the_switch():
    header = ps_header(REGISTER)
    assert "-NightExtraction" in header
    assert NIGHT_TASK in header
    assert "extraction apply" in header
    assert "every 5 minutes" in header
