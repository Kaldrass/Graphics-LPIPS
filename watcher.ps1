# --- Paramètres ---
$bat   = "D:\These\Graphics-LPIPS\train_multiple.bat"  # <-- ton .bat
$quiet = 300   # secondes de "silence" avant déclenchement
$poll  = 5    # intervalle de sondage en secondes

function Get-RenderBlender {
    Get-CimInstance Win32_Process -Filter "Name='blender.exe'" |
        Where-Object { $_.CommandLine -match 'render_single\.py' }
}

$accum = 0
Write-Host "Watch started: waiting for render_single.py Blender jobs to finish..."
while ($true) {
    $procs = Get-RenderBlender
    if ($procs) {
        $accum = 0
        Start-Sleep -Seconds $poll
        continue
    }
    $accum += $poll
    if ($accum -ge $quiet) { break }
    Start-Sleep -Seconds $poll
}

Start-Process -FilePath $bat
Write-Host "Bat launched."
