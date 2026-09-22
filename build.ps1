param(
    [string]$Compiler = 'C:\msys64\mingw64\bin\gcc.exe',
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'build'),
    [switch]$Test
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw "Compiler not found: $Compiler"
}
$env:Path = "$(Split-Path -Parent $Compiler);$env:Path"
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed ($LASTEXITCODE)." }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $flags = @('-std=c11', '-O3', '-march=znver4', '-flto',
               '-Wall', '-Wextra', '-Wpedantic', '-Iinclude')
    $core = @('src/mkv128_core.c', 'src/mkv128_reference.c')
    $simd = @('src/mkv128_simd_avx2.c', 'src/mkv128_simd_avx512.c',
              'src/mkv128_simd_modes.c')

    Invoke-Checked $Compiler ($flags + @('tools/generate_tables.c',
        '-o', "$OutputDirectory/generate_tables.exe"))
    Invoke-Checked $Compiler ($flags + @('tests/test_vector.c',
        '-o', "$OutputDirectory/test_vector.exe"))
    Invoke-Checked $Compiler ($flags + @('tests/selftest.c') + $core + @(
        '-o', "$OutputDirectory/selftest.exe"))
    Invoke-Checked $Compiler ($flags + @('benchmark/benchmark.c') + $core + $simd + @(
        '-o', "$OutputDirectory/benchmark.exe"))

    if ($Test) {
        Invoke-Checked "$OutputDirectory/selftest.exe" @()
        Invoke-Checked "$OutputDirectory/benchmark.exe" @('1', '1', '1', 'selftest')
        '' | & "$OutputDirectory/test_vector.exe"
        if ($LASTEXITCODE -ne 0) { throw 'Test vector failed.' }
    }
    Write-Host "Built: $OutputDirectory"
}
finally {
    Pop-Location
}

# Windows x64 / AMD Zen 4 / MinGW-w64 GCC.
# Build and test: .\build.ps1 -Test
# Benchmark:     .\build\benchmark.exe 16 20000000 5
# Speed: MB/s (1 MB = 1,000,000 bytes); AC power required for benchmarking.
# Speedup: relative to T-table with the same thread count.
# Interactive:   .\build\test_vector.exe
# The standalone test_vector uses AVX-512 T-table gather.
# The benchmark selects the VBMI/GFNI implementation when available.
