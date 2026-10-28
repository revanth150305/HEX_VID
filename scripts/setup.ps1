[CmdletBinding()]
param(
    [ValidateSet("cpu", "cu118", "cu121", "cu124", "rocm6.2", "directml")]
    [string]$Torch = "cpu",
    [switch]$NativeSort,
    [switch]$SkipTorch,
    [string]$EnvironmentName = ".venv",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if ([System.IO.Path]::IsPathRooted($EnvironmentName) -or $EnvironmentName -match '[\\/]') {
    throw "EnvironmentName must be a simple directory name inside the project."
}

if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-Python([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Arguments" }
}

Invoke-Python @("--version")
Invoke-Python @("-c", "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Python 3.10-3.12 is required'")
if (-not (Test-Path "$EnvironmentName\Scripts\python.exe")) {
    Invoke-Python @("-m", "venv", $EnvironmentName)
}

$VenvPython = Join-Path $ProjectRoot "$EnvironmentName\Scripts\python.exe"
function Invoke-VenvPython([string[]]$Arguments) {
    & $VenvPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Virtual-environment Python command failed: $Arguments" }
}

Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-VenvPython @("-c", "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Environment must use Python 3.10-3.12; recreate it and rerun setup'")

if (-not $SkipTorch -and $Torch -eq "directml") {
    Invoke-VenvPython @("-m", "pip", "install", "--pre", "torch-directml")
} elseif (-not $SkipTorch) {
    $TorchIndex = if ($Torch -eq "cpu") { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/$Torch" }
    Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "--index-url", $TorchIndex, "torch", "torchvision")
}

Invoke-VenvPython @("-m", "pip", "install", "--upgrade-strategy", "only-if-needed", "-r", "requirements.txt")

if ($NativeSort) {
    Write-Warning "Native sort-tracker is primarily supported on Linux and may require a C++ compiler, OpenCV, and Eigen."
    Invoke-VenvPython @("-m", "pip", "install", "-r", "requirements-tracker-native.txt")
}

@("data/raw", "data/processed", "data/subset", "checkpoints", "outputs", "results", "logs", "configs", "src", "tests") |
    ForEach-Object { New-Item -ItemType Directory -Force $_ | Out-Null }

Invoke-VenvPython @("-c", "import cv2, torch, transformers; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available()); print('opencv=', cv2.__version__); print('transformers=', transformers.__version__)")
Write-Host "Setup complete. Activate with: .\$EnvironmentName\Scripts\Activate.ps1"

