<#
.SYNOPSIS
    Registers the Windows sign-in task "MIST daily instance", which starts the daily MIST.AI
    instance when the current user signs in.

.DESCRIPTION
    Run it as:

        powershell -ExecutionPolicy Bypass -File <script> -Python <python.exe> -Root <root> [-LockAtSignIn]

    The task runs `<Python> <Root>/bin/instance.py up --root <Root>` at the current user's
    sign-in. `up` waits up to 10 minutes for Docker to answer, so the task depends on Docker
    Desktop's own "Start when you sign in" setting: without it nothing starts Docker, and `up`
    exits 1 after the wait.

    -LockAtSignIn is for a machine Raj has set to sign in automatically. The task's first action
    then locks the workstation (rundll32.exe user32.dll,LockWorkStation), ahead of `up`.

    This script stores no credential and does not turn automatic sign-in on. The task runs
    only while the user is signed in, as that user, without elevation. Running the script again
    replaces the task; unregister-signin-task.ps1 removes it. deploy/daily/README.md, "Start
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
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Python,

    [Parameter(Mandatory = $true)]
    [string]$Root,

    [switch]$LockAtSignIn
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RootPath = (Resolve-Path -LiteralPath $Root).ProviderPath
$Script = Join-Path -Path $RootPath -ChildPath 'bin\instance.py'
if (-not (Test-Path -LiteralPath $Script -PathType Leaf)) {
    throw "$Script does not exist; run instance.py switch <tag> --root $RootPath first"
}
$PythonPath = (Get-Command -Name $Python -CommandType Application | Select-Object -First 1).Source

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
