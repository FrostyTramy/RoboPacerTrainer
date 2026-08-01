# RoboPacer – Hailo HEF compilation (standalone, bypasses web UI)
# Usage:  .\engine\compile\run_compile.ps1  [-ModelName steering_net]
param([string]$ModelName = "steering_net")

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent  # D:\RoboPacer
$EngineDir   = Split-Path $PSScriptRoot -Parent                        # D:\RoboPacer\engine

Write-Host "`n=== RoboPacer: Hailo HEF Compilation  model=$ModelName ===" -ForegroundColor Cyan

$onnx  = Join-Path $ProjectRoot "models\$ModelName.onnx"
$calib = Join-Path $ProjectRoot "models\calib_data.npy"
$whl   = Join-Path $EngineDir   "resources\hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl"

foreach ($f in @($onnx, $calib, $whl)) {
    if (-not (Test-Path $f)) { Write-Error "Missing: $f"; exit 1 }
}
Write-Host "Input files OK" -ForegroundColor Green

docker info *>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker not running. Start Docker Desktop (Linux containers mode)."
    exit 1
}

Write-Host "`nBuilding hailo-dfc image ..."
docker build -t hailo-dfc -f "$EngineDir\compile\Dockerfile" "$EngineDir"
if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed"; exit 1 }

Write-Host "`nCompiling inside Docker ..."
docker run --rm -v "${ProjectRoot}:/workspace" hailo-dfc bash /workspace/engine/compile/compile.sh $ModelName
if ($LASTEXITCODE -ne 0) { Write-Error "DFC compilation failed"; exit 1 }

$hef = Join-Path $ProjectRoot "models\${ModelName}.hef"
if (Test-Path $hef) {
    $size = (Get-Item $hef).Length / 1MB
    Write-Host "`nHEF ready: $hef  ($([Math]::Round($size,2)) MB)" -ForegroundColor Green
} else {
    Write-Warning "HEF not found at expected path."
}
