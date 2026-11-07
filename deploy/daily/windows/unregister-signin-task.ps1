<#
.SYNOPSIS
    Removes the Windows sign-in tasks "MIST daily instance" and "MIST gaming watch", for a
    rollback.

.DESCRIPTION
    Run it as:

        powershell -ExecutionPolicy Bypass -File <script>

    It removes the tasks that register-signin-task.ps1 registered, each only if it is
    registered, and nothing else. It does not stop the instance: `instance.py down --root
    <root>` does that. A watcher already running keeps running until sign-out.
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

$WatchTask = Get-ScheduledTask -TaskName 'MIST gaming watch' -ErrorAction SilentlyContinue
if ($null -eq $WatchTask) {
    Write-Output "No task named 'MIST gaming watch' is registered."
}
else {
    Unregister-ScheduledTask -TaskName 'MIST gaming watch' -Confirm:$false
    Write-Output "Removed the task 'MIST gaming watch'."
}
