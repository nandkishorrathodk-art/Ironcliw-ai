$root = 'C:\Users\nandk\JARVIS'
$skipPatterns = @('\.git\\', 'node_modules\\', '__pycache__\\', '\.jarvis_cache\\', 'model_checkpoints\\', 'ml_models\\', 'vsms_history', '\.min\.js', 'package-lock\.json', 'rename_jarvis\.ps1')
$count = 0
$updated = @()

$files = Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
    $ext = $_.Extension.ToLower()
    $allowed = @('.js', '.html', '.css', '.json', '.py', '.md', '.txt', '.sh', '.bat', '.yaml', '.yml', '.ts', '.env', '.cfg', '.ini', '.toml', '.jsx', '.tsx')
    if ($allowed -notcontains $ext) { return $false }
    foreach ($p in $skipPatterns) {
        if ($_.FullName -match $p) { return $false }
    }
    return $true
}

foreach ($file in $files) {
    try {
        $content = [System.IO.File]::ReadAllText($file.FullName, [System.Text.Encoding]::UTF8)
        if ($content -cmatch 'JARVIS') {
            $newContent = $content -creplace 'JARVIS', 'Ironcliw'
            [System.IO.File]::WriteAllText($file.FullName, $newContent, [System.Text.Encoding]::UTF8)
            $count++
            $updated += $file.FullName.Replace($root, '')
        }
    } catch {
        # skip unreadable files
    }
}

Write-Host "=== JARVIS -> Ironcliw rename complete ==="
Write-Host "Total files updated: $count"
Write-Host ""
if ($updated.Count -gt 0) {
    Write-Host "Sample updated files (first 30):"
    $updated | Select-Object -First 30 | ForEach-Object { Write-Host "  $_" }
    if ($updated.Count -gt 30) {
        Write-Host "  ... and $($updated.Count - 30) more files"
    }
}
