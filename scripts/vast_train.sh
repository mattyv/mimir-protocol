#!/usr/bin/env bash
# Spin up a Vast.ai A100-80GB instance, set it up, and run axiom training.
# Usage: ./scripts/vast_train.sh [--model MODEL] [--n-steps N] [--n-synthetic N]
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
N_STEPS="${N_STEPS:-3000}"
N_SYNTHETIC="${N_SYNTHETIC:-30}"
DISK_GB="${DISK_GB:-120}"
GPU_RAM="${GPU_RAM:-79}"
GPU_QUERY="${GPU_QUERY:-gpu_name=A100_PCIE}"
REPO="https://github.com/mattyv/mimir-protocol.git"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --n-steps) N_STEPS="$2"; shift 2 ;;
    --n-synthetic) N_SYNTHETIC="$2"; shift 2 ;;
    --gpu-query) GPU_QUERY="$2"; shift 2 ;;
    --gpu-ram) GPU_RAM="$2"; shift 2 ;;
    --disk) DISK_GB="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "═══════════════════════════════════════════════════"
echo " Mimir-Protocol Vast.ai Training"
echo " Model:       $MODEL"
echo " Steps:       $N_STEPS"
echo " Synthetic:   $N_SYNTHETIC"
echo "═══════════════════════════════════════════════════"

# ── Find best A100-80GB instance ───────────────────────────────────────────
echo ""
echo "→ Searching for GPU ($GPU_QUERY, >=${GPU_RAM}GB)..."
OFFER_ID=$(vastai search offers \
  "$GPU_QUERY num_gpus=1 gpu_ram>=$GPU_RAM cuda_vers>=12.0 disk_space>=$DISK_GB reliability>=0.95" \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id']) if offers else exit(1)")

if [[ -z "$OFFER_ID" ]]; then
  echo "✗ No A100-80GB available. Try A100-SXM or check back later."
  exit 1
fi

PRICE=$(vastai search offers \
  "$GPU_QUERY num_gpus=1 gpu_ram>=$GPU_RAM cuda_vers>=12.0 disk_space>=$DISK_GB reliability>=0.95" \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; offers=json.load(sys.stdin); print(f\"\${offers[0]['dph_total']:.2f}/hr\")")

echo "  Found offer $OFFER_ID at $PRICE"

# ── Create instance ────────────────────────────────────────────────────────
echo ""
echo "→ Creating instance..."
INSTANCE_JSON=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --ssh --direct --raw 2>/dev/null)
INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['new_contract'])")
echo "  Instance ID: $INSTANCE_ID"

# ── Wait until running ─────────────────────────────────────────────────────
echo ""
echo "→ Waiting for instance to start..."
for i in $(seq 1 60); do
  STATUS=$(vastai show instances --raw 2>/dev/null | \
    python3 -c "import sys,json; instances=json.load(sys.stdin); \
      match=[i for i in instances if i['id']==$INSTANCE_ID]; \
      print(match[0]['actual_status'] if match else 'unknown')")
  echo "  Status: $STATUS"
  if [[ "$STATUS" == "running" ]]; then
    break
  fi
  sleep 10
done

# ── Get SSH details ────────────────────────────────────────────────────────
SSH_DETAILS=$(vastai show instances --raw 2>/dev/null | \
  python3 -c "import sys,json; instances=json.load(sys.stdin); \
    match=[i for i in instances if i['id']==$INSTANCE_ID][0]; \
    print(match['ssh_host'], match['ssh_port'])")
SSH_HOST=$(echo "$SSH_DETAILS" | cut -d' ' -f1)
SSH_PORT=$(echo "$SSH_DETAILS" | cut -d' ' -f2)
echo ""
echo "  SSH: $SSH_HOST:$SSH_PORT"

SSH="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$SSH_HOST"

# ── Wait for SSH to be ready ───────────────────────────────────────────────
echo ""
echo "→ Waiting for SSH..."
for i in $(seq 1 20); do
  if $SSH "echo ok" &>/dev/null; then
    echo "  SSH ready"
    break
  fi
  sleep 5
done

# ── Setup ──────────────────────────────────────────────────────────────────
echo ""
echo "→ Setting up environment..."
$SSH bash << 'SETUP'
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/mattyv/mimir-protocol.git
cd mimir-protocol
# Use system Python (Docker image already has CUDA torch)
# Install only non-torch deps
pip install transformers>=5.5.0 accelerate>=1.0.1 sentencepiece
echo "Setup complete"
SETUP

# ── Launch training in tmux (persists if SSH drops) ───────────────────────
echo ""
echo "→ Launching training (tmux session: mimir)..."
$SSH bash << TRAIN
cd mimir-protocol
tmux new-session -d -s mimir "PYTHONPATH=src python3 -m marker.run_axiom_mlp_demo \
  --model-name '$MODEL' \
  --n-steps $N_STEPS \
  --n-synthetic $N_SYNTHETIC \
  --save-dir /root/mimir-protocol/axioms_out \
  2>&1 | tee /tmp/training.log; echo DONE >> /tmp/training.log"
echo "Training running in tmux. Attach with: ssh -p $SSH_PORT root@$SSH_HOST -t tmux attach -t mimir"
TRAIN

echo ""
echo "═══════════════════════════════════════════════════"
echo " Training launched!"
echo ""
echo " Monitor:  ssh -p $SSH_PORT root@$SSH_HOST -t tmux attach -t mimir"
echo " Logs:     ssh -p $SSH_PORT root@$SSH_HOST tail -f /tmp/training.log"
echo " Axioms:   scp -P $SSH_PORT -r root@$SSH_HOST:/root/mimir-protocol/axioms_out ./axioms_7b"
echo " Instance: $INSTANCE_ID (remember to destroy when done)"
echo ""
echo " To destroy:  vastai destroy instance $INSTANCE_ID"
echo "═══════════════════════════════════════════════════"
