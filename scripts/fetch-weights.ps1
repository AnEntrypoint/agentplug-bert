$ErrorActionPreference = "Stop"

$WeightsDir = Join-Path (Split-Path -Parent $PSScriptRoot) "weights"
New-Item -ItemType Directory -Force -Path $WeightsDir | Out-Null

$Items = @(
  @{
    Url  = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors"
    Path = Join-Path $WeightsDir "bge-small-en-v1.5.safetensors"
    Sha  = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
  },
  @{
    Url  = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json"
    Path = Join-Path $WeightsDir "bge-tokenizer.json"
    Sha  = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
  }
)

function Test-Sha {
  param([string]$Path, [string]$Expected)
  if (-not (Test-Path $Path)) { return $false }
  $actual = (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLower()
  return $actual -eq $Expected.ToLower()
}

foreach ($it in $Items) {
  if (Test-Sha -Path $it.Path -Expected $it.Sha) {
    Write-Output "ok: $($it.Path) (sha matches)"
    continue
  }
  Write-Output "fetching $($it.Url) -> $($it.Path)"
  Invoke-WebRequest -Uri $it.Url -OutFile $it.Path -UseBasicParsing
  if (-not (Test-Sha -Path $it.Path -Expected $it.Sha)) {
    Write-Error "sha mismatch for $($it.Path)"
    exit 1
  }
}

Write-Output "weights ready in $WeightsDir"
