#!/usr/bin/env bash
# Stage-1 gist k-sweep: ONE Vast RTX 3090 per k, in parallel. Each node trains
# a gist adapter from scratch at its k and prints the gap_closed capacity
# curve. NO push (--repo empty) — the sweep's deliverable is the numbers, not
# the adapters (Stage-2 uses the existing k=8 pilot). Each arm is self-
# contained; a dead node just needs that one k relaunched.
#
# Answers the owed granularity question (STAGE2_PLAN "one vector per thought"):
#   k=1  -> pre-registered prediction: COLLAPSES (single-slot attention has no
#           query-dependent readout).
#   k=4 vs k=8 -> ~=  => sentence content is low-rank, coarse units suffice.
#   k=16 >> k=8 -> sentences are underfunded, finer granularity helps.
#
# HF_TOKEN (env) authenticates the 7B download (no push). Prints the instance
# id per arm; arm a poller on each.
#
#   HF_TOKEN=... ./scripts/vast_ksweep.sh            # k = 1 4 8 16
#   KS="1 16" HF_TOKEN=... ./scripts/vast_ksweep.sh  # subset
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=90
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
KS="${KS:-1 4 8 16}"
STEPS="${STEPS:-8000}"     # pilot plateaued ~7500
EVAL_EVERY="${EVAL_EVERY:-500}"
HELDOUT="${HELDOUT:-512}"
TIMEOUT="${TIMEOUT:-6h}"

launch_arm() {
  local K="$1"
  echo "→ [k=$K] searching RTX 3090 (rel>=0.98 inet>=500 driver>=535)..."
  local OFFER_ID=""
  for try in 1 2 3 4 5; do
    OFFER_ID=$(vastai search offers \
      'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.4 driver_version>=535 disk_space>=100 reliability>=0.98 inet_down>=500 rentable=true' \
      --order dph_total --limit 1 --raw 2>/dev/null | \
      python3 -c "import sys,json
try: o=json.load(sys.stdin); print(o[0]['id'] if o else '')
except Exception: print('')" 2>/dev/null)
    [ -n "$OFFER_ID" ] && break
    echo "  [k=$K] no offer (try $try/5), retrying in 10s..."; sleep 10
  done
  [ -z "$OFFER_ID" ] && { echo "[k=$K] NO OFFERS after retries — skipping"; return 1; }

  read -r -d '' ONSTART <<EOS || true
exec > /proc/1/fd/1 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
( while true; do echo "  ...setup heartbeat \$(date -u +%H:%M:%S)"; sleep 40; done ) &
HB=\$!
cd /root
echo "=== [k=$K] clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -2
cd /root/mimir-protocol
echo "=== pip ==="
pip install 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -4 \
  || { sleep 20; pip install 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -4; }
python -c "import torch,peft,bitsandbytes,datasets; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA True')" || { kill \$HB; echo "SETUPFAIL (no CUDA — driver/image mismatch, e.g. error 804; relaunch for another node)"; echo "ALLDONE"; exit 1; }
python -c "from huggingface_hub import whoami; print('HF auth ok:', whoami().get('name'))" \
  || { kill \$HB; echo "SETUPFAIL (bad/revoked HF token)"; echo "ALLDONE"; exit 1; }
echo "=== download ${MODEL} (20min cap) ==="
timeout 1200 python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}'); print('MODEL CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (download too slow)"; echo "ALLDONE"; exit 1; }
echo "=== [k=$K] gist train from scratch (steps=${STEPS}, NO push) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_gist_pilot \
  --model-name ${MODEL} --gist-k $K --max-steps ${STEPS} \
  --eval-every ${EVAL_EVERY} --ckpt-every 0 --heldout-n ${HELDOUT} 2>&1 | tee /root/ksweep.log
kill \$HB 2>/dev/null
echo "KSWEEP_K=$K RC=\${PIPESTATUS[0]}" | tee -a /root/ksweep.log
echo "ALLDONE" | tee -a /root/ksweep.log
EOS

  local ENV_ARG=""
  [ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"
  local IID
  IID=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
  echo "INSTANCE_K${K}=$IID   poller:  bash scripts/vast_poll_destroy.sh $IID 400"
}

for K in $KS; do launch_arm "$K"; done
echo "→ all arms launched. Arm a poller on each instance id above."
