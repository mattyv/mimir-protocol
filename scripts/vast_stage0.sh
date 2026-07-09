#!/usr/bin/env bash
# Speculative draft-and-verify baseline: 0.5B drafts for 7B on a Vast RTX 3090
# (both models co-resident, ~16GB total). Onstart-driven (no SSH), ALLDONE
# sentinel, poll with vastai logs then destroy.
#
# Usage: ./scripts/vast_spec_decode.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=80

echo "→ Searching RTX 3090 (reliability >= 0.98)..."
OFFER_ID=$(vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=90 reliability>=0.98 inet_down>=300 rentable=true' \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; o=json.load(sys.stdin); print(o[0]['id']) if o else exit(1)")
echo "  offer $OFFER_ID"

read -r -d '' ONSTART <<'EOS' || true
# Route onstart output to PID 1's stdout so `vastai logs` surfaces it.
exec > /proc/1/fd/1 2>&1
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -3
cd /root/mimir-protocol
echo "=== pip ==="
pip install -q 'transformers>=4.45,<5' 'accelerate>=1.0' sentencepiece hf_transfer 2>&1 | tail -3 \
  || { sleep 20; pip install -q 'transformers>=4.45,<5' 'accelerate>=1.0' sentencepiece hf_transfer 2>&1 | tail -3; }
python -c "import torch,transformers; print('CUDA', torch.cuda.is_available(), 'tf', transformers.__version__)" \
  || { echo "SETUPFAIL"; echo "ALLDONE"; exit 1; }
# hf_transfer = Rust parallel downloader; the plain HF link stalled 3 nodes at
# 7B model-load. Heartbeat so a slow-but-live download doesn't trip the stall.
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "=== downloading 7B (hf_transfer) ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B'); print('7B CACHED')" 2>&1 | tail -2
echo "=== run: stage0 drift budget ==="
PYTHONPATH=src python -u -m marker.run_stage0_soft \
  --model-name Qwen/Qwen2.5-7B \
  --n-steps 64 2>&1 | tee /root/stage0.log
echo "STAGE0_RC=${PIPESTATUS[0]}" | tee -a /root/stage0.log
echo "=== run: specdec Finding-2 cross-check (reference-prefill) ==="
PYTHONPATH=src python -u -m marker.run_spec_decode \
  --verifier Qwen/Qwen2.5-7B --drafter Qwen/Qwen2.5-0.5B \
  --max-new 80 --reference-prefill 2>&1 | tee -a /root/stage0.log
echo "XCHECK_RC=${PIPESTATUS[0]}" | tee -a /root/stage0.log
echo "ALLDONE" | tee -a /root/stage0.log
EOS

echo "→ Creating instance..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
