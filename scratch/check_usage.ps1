$path = "C:\Users\ASUS\Installation\AgenticAI\Multi_Agent_System\frontend\src\features\verify\VerifyPage.tsx"
$content = Get-Content -LiteralPath $path -Raw
$comps = @('ConfidenceMeter','StatusBadge','InvestigationTimeline','ClaimBreakdownCard','EvidenceInspector','ClaimConfidenceBreakdown','FinalVerdictExplanation','UnifiedReport','RawPayloadInspector','LiveInvestigationPanel','LogEntry')
foreach ($c in $comps) {
    $n = ($content -split "`n" | Where-Object { $_ -like "*$c*" }).Count
    Write-Output ($c + ": " + $n)
}
