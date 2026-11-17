param(
    [switch]$MeasureRuntime  # Optional: also measure wall-clock runtime
)

# We expect spark-submit to return a non-zero exit when test.parquet is missing.
# Keep going and just harvest the MAP lines.
$ErrorActionPreference = 'Continue'

Write-Host "Running baseline (03)..."
try {
    spark-submit --master local[*] 03_baseline_model.py *> 03_output.txt
} catch {
    Write-Host "spark-submit (03) returned an error (likely missing test set). Continuing..."
}

Write-Host "Running ALS (04)..."
try {
    spark-submit --master local[*] 04_latent_factor_model.py *> 04_output.txt
} catch {
    Write-Host "spark-submit (04) returned an error (likely missing test set). Continuing..."
}

# Extract metrics into metrics.txt
"=== Baseline (03) MAP@100 metrics ===" | Out-File metrics.txt
Select-String "Mean Average Precision" 03_output.txt | Out-File -Append metrics.txt

"`n=== ALS (04) MAP@100 metrics ===" | Out-File -Append metrics.txt
Select-String "Mean Average Precision" 04_output.txt | Out-File -Append metrics.txt

if ($MeasureRuntime) {
    "`n=== Runtimes (wall-clock) ===" | Out-File -Append metrics.txt

    "03_baseline_model.py time:" | Out-File -Append metrics.txt
    Measure-Command { spark-submit --master local[*] 03_baseline_model.py *> $null } | Out-File -Append metrics.txt

    "`n04_latent_factor_model.py time:" | Out-File -Append metrics.txt
    Measure-Command { spark-submit --master local[*] 04_latent_factor_model.py *> $null } | Out-File -Append metrics.txt
}

Write-Host "Done. See metrics.txt for summarized metrics." -ForegroundColor Green
