$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Installez Python 3.11 puis relancez ce script.' }
}
& $projectPython -c "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(name) is None for name in ('streamlit', 'torch', 'transformers', 'peft', 'accelerate', 'safetensors')))"
if ($LASTEXITCODE -ne 0) {
    & $projectPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Installation des dependances impossible.' }
}
& $projectPython -m streamlit run (Join-Path $PSScriptRoot 'app.py') @args
exit $LASTEXITCODE
