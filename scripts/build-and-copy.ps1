
$frontend = ".\eco-traffic-vision"
$dist = Join-Path $frontend "dist"
$manifestSrc = Join-Path $dist ".vite\manifest.json"
$assetsSrc = Join-Path $dist "assets"
$staticDest = "./static"
$assetsDest = Join-Path $staticDest "assets"

Write-Output "Building front-end in $frontend..."
Push-Location $frontend
npm run build
Pop-Location

if (Test-Path $assetsDest) {
    Write-Output "Removing existing $assetsDest"
    Remove-Item -Recurse -Force $assetsDest
}

Write-Output "Copying $assetsSrc to $assetsDest"
Copy-Item -Recurse -Force $assetsSrc $assetsDest

if (Test-Path $manifestSrc) {
    Write-Output "Copying manifest to $staticDest\manifest.json"
    Copy-Item -Force $manifestSrc (Join-Path $staticDest "manifest.json")
} else {
    Write-Output "Warning: manifest not found at $manifestSrc"
}

Write-Output "Done. Remember to restart Flask if not running in debug mode."
