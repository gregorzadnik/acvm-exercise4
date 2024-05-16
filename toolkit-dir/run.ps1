$trackerPath = "examples\python_kalman.py"
$resultsPath = "..\workspace-dir\results\kalman"
$content = Get-Content $trackerPath -Encoding UTF8

$qs = (1, 50, 100, 200, 300, 400, 500, 800, 1000)
$models = ("RW, NCV, NCA")
$particles = (10, 20, 30, 40, 80, 120, 150, 200, 300)
$sigmas = (0.1, 0.5, 1, 1.5, 3, 4, 5, 6, 10)
$factors = (0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 2.5)
$allOverlaps = [System.Collections.ArrayList]@()
$allFailures = [System.Collections.ArrayList]@()
$allSpeeds = [System.Collections.ArrayList]@()
foreach($n in $particles){
    if(Test-Path $resultsPath){
        Get-ChildItem -Path $resultsPath -Recurse | Remove-Item -Force -Recurse
        Remove-Item $resultsPath -Force
    }
    Write-Host "n=$n"
    $newRow = "    def __init__(self, sigma=0.5, nbins=16, q=1, model=`"NCV`", n=$n, hell_sig=1, alpha=0.05):"
    $content[162] = $newRow
    $content | Out-File $trackerPath -Encoding utf8 -Force
    $content
    python evaluate_tracker.py --workspace_path ../workspace-dir --tracker kalman_tracker
    python .\calculate_measures.py --workspace_path ../workspace-dir --tracker kalman_tracker
    $res = Get-Content -Raw "../workspace-dir/analysis/kalman/results.json" | ConvertFrom-Json
    $allOverlaps.Add($res.average_overlap)
    $allFailures.Add($res.total_failures)
    $allSpeeds.Add($res.average_speed)
}
Write-Host
$particles | Write-Host
$allFailures | Write-Host
$allOverlaps | ForEach-Object {[math]::Round($_, 2)}
$allSpeeds | ForEach-Object {[math]::Round($_)}