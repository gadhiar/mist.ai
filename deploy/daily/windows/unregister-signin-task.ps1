<#
.SYNOPSIS
    Removes the Windows sign-in task "MIST daily instance", for a rollback.

.DESCRIPTION
    Run it as:

        powershell -ExecutionPolicy Bypass -File <script>

    It removes the task that register-signin-task.ps1 registered, and nothing else. It does not
    stop the instance: `instance.py down --root <root>` does that.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Task = Get-ScheduledTask -TaskName 'MIST daily instance' -ErrorAction SilentlyContinue
if ($null -eq $Task) {
    Write-Output "No task named 'MIST daily instance' is registered."
}
else {
    Unregister-ScheduledTask -TaskName 'MIST daily instance' -Confirm:$false
    Write-Output "Removed the task 'MIST daily instance'."
}
