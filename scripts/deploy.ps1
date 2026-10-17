param([Parameter(Mandatory=$true)][string]$VmIp)
$key = "$env:USERPROFILE\.ssh\sakura_key"
Write-Host "Redeploying on $VmIp ..."
ssh -t -i $key sakura@$VmIp "sudo systemctl restart fashion-api && sleep 5 && curl -s localhost:8000/health"
curl.exe "http://${VmIp}:8000/health"
