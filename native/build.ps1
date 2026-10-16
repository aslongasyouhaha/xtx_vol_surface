param([string]$Compiler = 'g++')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$nativeDeps = Join-Path $projectRoot 'data/outputs/native_deps'
New-Item -ItemType Directory -Force -Path $nativeDeps | Out-Null
$version = '1.15.1'
$archive = Join-Path $nativeDeps "highs-$version-windows.zip"
$archiveHash = '26302D9024F307E09128A45A58898917287351DCF754C55AEBC07742237F78BF'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/ERGO-Code/HiGHS/releases/download/v$version/highs-$version-x86_64-windows-mit.zip" -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $archiveHash) {
    throw 'Official HiGHS archive checksum mismatch.'
}
$libraryRoot = Join-Path $nativeDeps "highs-$version"
if (-not (Test-Path -LiteralPath (Join-Path $libraryRoot 'include/highs/Highs.h'))) {
    Expand-Archive -LiteralPath $archive -DestinationPath $libraryRoot
}
python -m pip install --target $nativeDeps --no-deps --upgrade "highspy==$version"
if ($LASTEXITCODE -ne 0) { throw 'Installing optional highspy dependency failed.' }
$includeRoot = Join-Path $libraryRoot 'include/highs'
$source = Join-Path $PSScriptRoot 'incremental_highs.cpp'
$executable = Join-Path $nativeDeps 'incremental_highs.exe'
& $Compiler '-O3' '-std=c++17' '-static-libgcc' '-static-libstdc++' '-I' $includeRoot $source '-o' $executable
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
Write-Output "Built $executable"
