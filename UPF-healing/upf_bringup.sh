#!/usr/bin/env bash
# upf_bringup.sh
# Reset e bring-up completo do UPF (bess + bess-routectl + pfcpiface)
# e recarrega o pipeline up4.bess.
#
# Uso: sudo ./upf_bringup.sh

set -e

echo "=== [1/5] Parando containers ==="
sudo docker stop pfcpiface bess-routectl bess 2>/dev/null || true

echo "=== [2/5] Subindo bess ==="
sudo docker start bess
sleep 8

echo "=== [3/5] Subindo bess-routectl ==="
sudo docker start bess-routectl
sleep 3

echo "=== [4/5] Carregando pipeline up4.bess ==="
sudo docker exec bess bessctl run file /opt/bess/bessctl/conf/up4.bess
sleep 3

echo "=== [5/5] Subindo pfcpiface ==="
sudo docker start pfcpiface
sleep 5

echo
echo "=== Validação ==="

# Pipeline carregado?
N_MODULES=$(sudo docker exec bess bessctl show pipeline 2>&1 | wc -l)
if [ "$N_MODULES" -gt 5 ]; then
    echo "[OK] Pipeline carregado ($N_MODULES linhas)"
else
    echo "[FAIL] Pipeline vazio ou erro. Saída:"
    sudo docker exec bess bessctl show pipeline 2>&1 | head -10
    exit 1
fi

# pfcpiface escutando?
if sudo docker logs --tail 20 pfcpiface 2>&1 | grep -q "listening for new PFCP connections"; then
    echo "[OK] pfcpiface escutando em :8805"
else
    echo "[WARN] pfcpiface pode não estar escutando ainda. Últimas linhas:"
    sudo docker logs --tail 5 pfcpiface 2>&1
fi

# Regras já estão? (esperado: 0 antes de criar sessões via pfcpsim)
N_PDR=$(sudo docker exec bess bessctl show module pdrLookup 2>&1 | grep -oP '\d+(?= rules)' || echo "?")
echo "[INFO] pdrLookup rules: $N_PDR (esperado 0 antes de criar sessões PFCP)"

# Métrica de sessões
N_SESS=$(curl -s http://192.168.90.1:8080/metrics 2>/dev/null | grep -oP 'pfcp_sessions\{[^}]+\} \K\d+' || echo "?")
echo "[INFO] pfcp_sessions: $N_SESS (esperado 0 antes de pfcpsim)"

echo
echo "=== Próximo passo ==="
echo "No host do TRex, criar sessões PFCP via pfcpsim:"
echo
echo "  sudo docker rm -f pfcpsim 2>/dev/null"
echo "  sudo docker run --rm -d --network host --name pfcpsim pfcpsim:patched \\"
echo "    -p 12345 --interface enp9s0"
echo "  sleep 2"
echo "  sudo docker exec pfcpsim pfcpctl -s localhost:12345 service configure \\"
echo "    --n3-addr 192.168.70.1 --remote-peer-addr 192.168.90.1"
echo "  sudo docker exec pfcpsim pfcpctl -s localhost:12345 service associate"
echo "  sudo docker exec pfcpsim pfcpctl -s localhost:12345 session create \\"
echo "    --count 4 --baseID 1 --ue-pool 10.250.0.0/24 --gnb-addr 192.168.70.2 \\"
echo "    --app-filter 'udp:any:any:allow:100' \\"
echo "    --session-mbr-uplink   1000000,1000000,1000000,1000000 \\"
echo "    --session-mbr-downlink 1000000,1000000,1000000,1000000 \\"
echo "    --app-mbr-uplink       200000,100000,50000,50000 \\"
echo "    --app-mbr-downlink     200000,100000,50000,50000"
