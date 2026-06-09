$goVersion = "1.22.4"
$zipUrl = "https://go.dev/dl/go$goVersion.windows-amd64.zip"
$destDir = "$PSScriptRoot\go_dist"
$zipFile = "$destDir\go.zip"

if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir | Out-Null
}

if (-not (Test-Path "$destDir\go\bin\go.exe")) {
    if (-not (Test-Path $zipFile) -or (Get-Item $zipFile).Length -eq 0) {
        if (Test-Path $zipFile) {
            Remove-Item $zipFile
        }
        Write-Host "Downloading Go $goVersion using curl.exe..."
        curl.exe -L -o "$zipFile" "$zipUrl"
    }
    Write-Host "Extracting Go..."
    Expand-Archive -Path $zipFile -DestinationPath $destDir
    if (Test-Path $zipFile) {
        Remove-Item $zipFile
    }
}
Write-Host "Go is ready at $destDir\go\bin\go.exe"
