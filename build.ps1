# Ensure Go is installed and available
& "$PSScriptRoot\install_go.ps1"

$goExe = "$PSScriptRoot\go_dist\go\bin\go.exe"
if (-not (Test-Path $goExe)) {
    Write-Error "Go executable not found at $goExe"
    exit 1
}

# Run tests
Write-Host "Running parser unit tests..."
Push-Location "$PSScriptRoot\parser"
& $goExe test -v
$testResult = $LASTEXITCODE
Pop-Location

if ($testResult -ne 0) {
    Write-Error "Parser tests failed!"
    exit 1
}

# Compile main.go
Write-Host "Compiling codelens-parser.exe..."
& $goExe build -o "$PSScriptRoot\codelens-parser.exe" "$PSScriptRoot\parser\main.go"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: $PSScriptRoot\codelens-parser.exe"
} else {
    Write-Error "Build failed!"
    exit 1
}
