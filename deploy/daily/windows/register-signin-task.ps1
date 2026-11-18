<#
.SYNOPSIS
    Registers the Windows sign-in task "MIST daily instance", which starts the daily MIST.AI
    instance when the current user signs in.

.DESCRIPTION
    Run it as:

        powershell -ExecutionPolicy Bypass -File <script> -Python <python.exe> -Root <root> [-LockAtSignIn] [-GamingWatch] [-NightExtraction]

    The task runs `<Python> <Root>/bin/instance.py up --root <Root>` at the current user's
    sign-in. `up` waits up to 10 minutes for Docker to answer, so the task depends on Docker
    Desktop's own "Start when you sign in" setting: without it nothing starts Docker, and `up`
    exits 1 after the wait.

    -LockAtSignIn is for a machine Raj has set to sign in automatically. The task's first action
    then locks the workstation (rundll32.exe user32.dll,LockWorkStation), ahead of `up`.

    -GamingWatch also registers a second sign-in task, "MIST gaming watch", which runs
    `<pythonw.exe> <Root>/bin/instance.py gaming-watch --root <Root>`. pythonw.exe is taken
    from the directory of -Python, so the watcher opens no console window; the script refuses,
    before registering anything, when it is not there. Running the script again without
    -GamingWatch leaves an existing watch task in place.

    -NightExtraction also registers "MIST extraction window", which runs
    `<pythonw.exe> <Root>/bin/instance.py extraction apply --root <Root>` every 5 minutes from
    00:00, and at sign-in. Each run reads the extraction mode, the window and the gaming
    marker, and starts or stops the extraction services to match, so a missed run delays the
    change by one interval. It takes pythonw.exe as -GamingWatch does, and refuses in the same
    way. Running the script again without -NightExtraction leaves an existing
    extraction window task in place.

    This script stores no credential and does not turn automatic sign-in on. The task runs
    only while the user is signed in, as that user, without elevation. Running the script again
    replaces the task; unregister-signin-task.ps1 removes it and the other two tasks. deploy/daily/README.md, "Start
    after power-on", describes both ways to start the instance.

.PARAMETER Python
    The Python interpreter that runs bin/instance.py: a path, or a command name on PATH, which
    is resolved to its full path now.

.PARAMETER Root
    The instance root, which must already hold bin/instance.py (instance.py switch puts it
    there).

.PARAMETER LockAtSignIn
    Lock the workstation as the task's first action, for a machine that signs in
    automatically.

.PARAMETER GamingWatch
    Also register "MIST gaming watch", which runs instance.py gaming-watch under pythonw.exe
    at sign-in, so a running game takes the GPU with no command typed; a launcher, client or
    anti-cheat never does. The watcher refuses to run on a Python that is not 64-bit.

.PARAMETER NightExtraction
    Also register "MIST extraction window", which runs instance.py extraction apply under
    pythonw.exe every 5 minutes and at sign-in, so the night extraction window is kept with no
    command typed.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [switch]$LockAtSignIn,

    [switch]$GamingWatch,

    [switch]$NightExtraction
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RootPath = (Resolve-Path -LiteralPath $Root).ProviderPath
$Script = Join-Path -Path $RootPath -ChildPath 'bin\instance.py'
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "$Script does not exist; run instance.py switch <tag> --root $RootPath first"
}
$PythonPath = (Get-Command -Name $Python -CommandType Application | Select-Object -First 1).Source
$PythonDir = Split-Path -Path $PythonPath -Parent
$PythonwPath = Join-Path -Path $PythonDir -ChildPath 'pythonw.exe'
if ($GamingWatch) {
    if (-not (Test-Path -LiteralPath $PythonwPath -PathType Leaf)) {
        throw "$PythonwPath does not exist; -GamingWatch runs the watcher with the pythonw.exe beside $PythonPath"
    }
}
if ($NightExtraction) {
    if (-not (Test-Path -LiteralPath $PythonwPath -PathType Leaf)) {
        throw "$PythonwPath does not exist; -NightExtraction runs extraction apply with the pythonw.exe beside $PythonPath"
    }
}

$UpArgument = '"{0}/bin/instance.py" up --root "{0}"' -f $RootPath
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$Actions = @()
if ($LockAtSignIn) {
    $Actions += New-ScheduledTaskAction -Execute 'rundll32.exe' -Argument 'user32.dll,LockWorkStation'
}
$Actions += New-ScheduledTaskAction -Execute $PythonPath -Argument $UpArgument -WorkingDirectory $RootPath

$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited

# No time limit, and no battery conditions: Task Scheduler's defaults would end a long task
# after 72 hours, and skip or stop it on a PC that a UPS reports as running on battery.
Register-ScheduledTask -TaskName 'MIST daily instance' `
    -Description 'Starts the daily MIST.AI instance at sign-in: instance.py up.' `
    -Action $Actions `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries) `
    -Force | Out-Null

schtasks.exe /query /tn 'MIST daily instance'

# The same trigger, principal and settings: with Task Scheduler's default 72-hour limit the
# watcher, which runs until sign-out, would be ended.
if ($GamingWatch) {
    $WatchArgument = '"{0}/bin/instance.py" gaming-watch --root "{0}"' -f $RootPath
    $WatchAction = New-ScheduledTaskAction -Execute $PythonwPath -Argument $WatchArgument -WorkingDirectory $RootPath
    Register-ScheduledTask -TaskName 'MIST gaming watch' `
        -Description 'Yields the GPU while a game itself runs: instance.py gaming-watch.' `
        -Action $WatchAction `
        -Trigger $Trigger `
        -Principal $Principal `
        -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries) `
        -Force | Out-Null

    schtasks.exe /query /tn 'MIST gaming watch'
}

# The window is kept by level-triggered ticks: every 5 minutes for the whole day, from 00:00,
# plus one at sign-in, since nothing else runs apply after a boot. New-ScheduledTaskTrigger
# -Daily takes no repetition, so the tick borrows a -Once trigger's. StartWhenAvailable runs a
# tick missed while the PC slept; IgnoreNew keeps a slow start from overlapping the next tick;
# 15 minutes ends a run that hangs. The principal and battery settings are the other tasks'.
if ($NightExtraction) {
    $ApplyArgument = '"{0}/bin/instance.py" extraction apply --root "{0}"' -f $RootPath
    $ApplyAction = New-ScheduledTaskAction -Execute $PythonwPath -Argument $ApplyArgument -WorkingDirectory $RootPath
    $Tick = New-ScheduledTaskTrigger -Daily -At '00:00'
    $Tick.Repetition = (New-ScheduledTaskTrigger -Once -At '00:00' -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
    Register-ScheduledTask -TaskName 'MIST extraction window' `
        -Description 'Keeps the night extraction window: instance.py extraction apply every 5 minutes.' `
        -Action $ApplyAction `
        -Trigger @($Tick, $Trigger) `
        -Principal $Principal `
        -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries) `
        -Force | Out-Null

    schtasks.exe /query /tn 'MIST extraction window'
}
