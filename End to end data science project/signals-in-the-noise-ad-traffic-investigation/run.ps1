param(
    [int]$Rows = 500000,
    [string]$DataDir = "data"
)

$ErrorActionPreference = "Stop"
python -m src.pipeline --rows $Rows --data-dir $DataDir
