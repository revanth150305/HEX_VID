[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("supervised", "weakly", "infer", "evaluate")]
    [string]$Mode,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Environment missing. Run .\scripts\setup.ps1 first." }
$ScriptName = @{ supervised = "train_supervised.py"; weakly = "train_weakly.py"; infer = "demo.py"; evaluate = "evaluate.py" }[$Mode]
$EntryPoint = Join-Path $ProjectRoot "scripts\$ScriptName"
if (-not (Test-Path $EntryPoint)) { throw "Missing $EntryPoint." }
& $Python $EntryPoint @ExtraArgs
exit $LASTEXITCODE

