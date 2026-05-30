#!/usr/bin/env bash
# Phase 4: value-token architecture vs MLP+KV on Qwen 7B-Instruct.
# Finds a reliable GPU, clones repo, runs the comparison in tmux.
#
# Usage: ./scripts/vast_phase4.sh
# After launch, run the watcher:
#   ./scripts/vast_phase4_watch.sh INSTANCE_ID SSH_HOST SSH_PORT

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
N_STEPS="${N_STEPS:-3000}"
N_SYNTHETIC="${N_SYNTHETIC:-30}"
R="${R:-32}"
LR="${LR:-1e-4}"
DISK_GB=80
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

echo "═══════════════════════════════════════════════════════"
echo " Mimir Phase 4 — value-token vs MLP+KV on $MODEL"
echo "═══════════════════════════════════════════════════════"

# ── Find a high-reliability GPU (RTX 3090 or better, 24GB+) ──────────────────
echo "→ Searching for GPU (reliability >= 0.99, 24GB+)..."
OFFER_ID=$(vastai search offers \
  'num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=80 reliability>=0.99' \
  --order dph_total --limit 5 --raw 2>/dev/null | \
  python3 -c "
import sys, json
offers = json.load(sys.stdin)
# Skip known bad GPU names that have caused issues
skip = set()
for o in offers:
    name = o.get('gpu_name', '')
    if name not in skip:
        print(o['id'])
        break
else:
    exit(1)
")
echo "  offer $OFFER_ID"

# ── Create instance ───────────────────────────────────────────────────────────
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "  instance $INSTANCE_ID"

# ── Wait for running + SSH ────────────────────────────────────────────────────
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

# ── Setup ─────────────────────────────────────────────────────────────────────
echo "→ Setting up..."
$SSH bash <<'SETUP'
set -e
git clone https://github.com/mattyv/mimir-protocol.git
cd mimir-protocol
pip install transformers>=5.5.0 accelerate>=1.0.1 sentencepiece 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print('CUDA OK', torch.version.cuda)"
SETUP

# ── Launch in tmux ────────────────────────────────────────────────────────────
echo "→ Launching Phase 4 comparison (tmux: phase4)..."
$SSH bash <<TRAIN
cd mimir-protocol
tmux new-session -d -s phase4 "PYTHONPATH=src python3 -m marker.run_value_token_demo \
  --model-name '$MODEL' \
  --n-steps $N_STEPS \
  --n-synthetic $N_SYNTHETIC \
  --r $R \
  --lr $LR \
  --save-dir /root/mimir-protocol/value_token_out \
  2>&1 | tee /tmp/phase4.log; echo DONE >> /tmp/phase4.log"
TRAIN

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Launched. Instance $INSTANCE_ID at $SSH_HOST:$SSH_PORT"
echo " Watch: ssh -p $SSH_PORT root@$SSH_HOST tail -f /tmp/phase4.log"
echo " Watcher: ./scripts/vast_phase4_watch.sh $INSTANCE_ID $SSH_HOST $SSH_PORT"
echo " Kill: vastai destroy instance $INSTANCE_ID"
echo "═══════════════════════════════════════════════════════"

echo "PHASE4_INSTANCE $INSTANCE_ID $SSH_HOST $SSH_PORT"
