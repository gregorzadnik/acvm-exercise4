$trackerPath = "examples\python_moss.py"
$resultsPath = "..\workspace-dir\results\moss"
$content = Get-Content $trackerPath -Encoding UTF8

$alphas = (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15)
$sigmas = (0.1, 0.5, 1, 1.5, 3, 4, 5, 6, 10)
$factors = (0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2, 2.5)
$allOverlaps = [System.Collections.ArrayList]@()
$allFailures = [System.Collections.ArrayList]@()
foreach($f in $factors){
    if(Test-Path $resultsPath){
        Get-ChildItem -Path $resultsPath -Recurse | Remove-Item -Force -Recurse
        Remove-Item $resultsPath -Force
    }
    Write-Host "Sigma=$f"
    $newRow = "    def __init__(self, sigma=3, lam = 3, enlarge_factor=$f, alpha=0.05):"
    $content[81] = $newRow
    $content | Out-File $trackerPath -Encoding utf8 -Force
    $content
    python evaluate_tracker.py --workspace_path ../workspace-dir --tracker moss_tracker
    python .\calculate_measures.py --workspace_path ../workspace-dir --tracker moss_tracker
    $res = Get-Content -Raw "../workspace-dir/analysis/moss/results.json" | ConvertFrom-Json
    $allOverlaps.Add($res.average_overlap)
    $allFailures.Add($res.total_failures)
}
$factors | Write-Host
$allFailures | Write-Host
$allOverlaps | ForEach-Object {[math]::Round($_, 2)}