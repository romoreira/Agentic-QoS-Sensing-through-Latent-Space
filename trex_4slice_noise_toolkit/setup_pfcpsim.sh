#!/bin/bash

# Interrompe a execução do script caso algum comando crítico falhe
set -e

echo "🧹 Limpando containers anteriores (se existirem)..."
# O || true garante que o script não pare se o container não existir
sudo docker rm -f pfcpsim 2>/dev/null || true

echo "🚀 Iniciando o container pfcpsim..."
sudo docker run --rm -d --network host \
  --name pfcpsim \
  pfcpsim:patched \
  -p 12345 \
  --interface enp9s0

echo "⏳ Aguardando 2 segundos para o serviço iniciar adequadamente..."
sleep 2

echo "⚙️  Configurando endereços N3 e Remote Peer..."
sudo docker exec pfcpsim pfcpctl -s localhost:12345 service configure \
  --n3-addr 192.168.70.1 \
  --remote-peer-addr 192.168.90.1

echo "🔗 Estabelecendo associação (service associate)..."
sudo docker exec pfcpsim pfcpctl -s localhost:12345 service associate

echo "📡 Criando as 4 sessões com as regras de QoS..."
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session create \
  --count 4 \
  --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000,1000000 \
  --app-mbr-uplink       200000,100000,50000,50000 \
  --app-mbr-downlink     200000,100000,50000,50000

echo "✅ Configuração concluída com sucesso!"
