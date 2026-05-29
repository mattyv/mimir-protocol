#!/usr/bin/env bash
# Phase 3: spin up a Vast RTX 3090, run the single-vector vs MLP+KV
# head-to-head comparison on Qwen 7B-Instruct, capture artifacts + log,
# and destroy the instance unattended.
#
# Usage: ./scripts/vast_phase3.sh
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_STEPS="${N_STEPS:-3000}"
N_SYNTHETIC="${N_SYNTHETIC:-30}"
R="${R:-32}"
LR="${LR:-1e-4}"
DISK_GB=80
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "═══════════════════════════════════════════════════"
echo " Mimir Phase 3 — single-vector vs MLP+KV on $MODEL"
echo "═══════════════════════════════════════════════════"

# ── Find a high-reliability 3090 (avoids the hung-download nodes we hit) ─────
echo "→ Searching for RTX 3090 (reliability >= 0.99)..."
OFFER_ID=$(vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=80 reliability>=0.99' \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; o=json.load(sys.stdin); print(o[0]['id']) if o else exit(1)")
echo "  offer $OFFER_ID"

# ── Create instance ──────────────────────────────────────────────────────────
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "  instance $INSTANCE_ID"

# ── Wait for running + SSH ───────────────────────────────────────────────────
echo "→ Waiting for instance..."
for _ in $(seq 1 90); do
  STATUS=$(vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; \
    m=[i for i in json.load(sys.stdin) if i['id']==$INSTANCE_ID]; \
    print(m[0]['actual_status'] if m else 'unknown')")
  [[ "$STATUS" == "running" ]] && break
  sleep 10
done
read -r SSH_HOST SSH_PORT < <(vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; \
  m=[i for i in json.load(sys.stdin) if i['id']==$INSTANCE_ID][0]; print(m['ssh_host'], m['ssh_port'])")
echo "  SSH: $SSH_HOST:$SSH_PORT"
SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST"
for _ in $(seq 1 30); do $SSH "echo ok" &>/dev/null && break; sleep 5; done

# ── Setup + CUDA verify ──────────────────────────────────────────────────────
echo "→ Setting up..."
$SSH bash <<'SETUP'
set -e
git clone https://github.com/mattyv/mimir-protocol.git
cd mimir-protocol
pip install transformers>=5.5.0 accelerate>=1.0.1 sentencepiece 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('CUDA OK', torch.version.cuda)"
SETUP

# ── Launch comparison in tmux ────────────────────────────────────────────────
echo "→ Launching Phase 3 comparison (tmux: phase3)..."
$SSH bash <<TRAIN
cd mimir-protocol
tmux new-session -d -s phase3 "PYTHONPATH=src python3 -m marker.run_single_vector_demo \
  --model-name '$MODEL' \
  --n-steps $N_STEPS \
  --n-synthetic $N_SYNTHETIC \
  --r $R \
  --lr $LR \
  --include-skills \
  --save-dir /root/mimir-protocol/single_vector_out \
  2>&1 | tee /tmp/phase3.log; echo DONE >> /tmp/phase3.log"
TRAIN

echo ""
echo "═══════════════════════════════════════════════════"
echo " Launched. Instance $INSTANCE_ID at $SSH_HOST:$SSH_PORT"
echo " Logs:   $SSH tail -f /tmp/phase3.log"
echo " Watch:  ./scripts/vast_phase3_watch.sh $INSTANCE_ID $SSH_HOST $SSH_PORT"
echo " Kill:   vastai destroy instance $INSTANCE_ID"
echo "═══════════════════════════════════════════════════"

# Emit machine-readable line for the watcher.
echo "PHASE3_INSTANCE $INSTANCE_ID $SSH_HOST $SSH_PORT"
