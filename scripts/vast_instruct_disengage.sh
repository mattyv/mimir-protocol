#!/usr/bin/env bash
# Instruct Phase 3 (skills on a chat model): launch a Vast RTX 3090, run
# run_instruct_disengage \
# container's egress is HTTPS-only), print the instance id. Poll with
# `vastai logs <id>` for the ALLDONE sentinel, then `vastai destroy instance`.
#
# Usage: ./scripts/vast_instruct_skills.sh
set -euo pipefail

BRANCH="claude/project-review-6rx97z"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=80

echo "→ Searching RTX 3090 (reliability >= 0.98)..."
OFFER_ID=$(vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=90 reliability>=0.98 rentable=true' \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; o=json.load(sys.stdin); print(o[0]['id']) if o else exit(1)")
echo "  offer $OFFER_ID"

read -r -d '' ONSTART <<'EOS' || true
# Route onstart output to PID 1's stdout so `vastai logs` surfaces it (the
# default onstart.log is not returned by the logs API).
exec > /proc/1/fd/1 2>&1
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -3
cd /root/mimir-protocol
echo "=== pip ==="
pip install -q 'transformers>=4.45,<5' 'accelerate>=1.0' sentencepiece 2>&1 | tail -3
python -c "import torch,transformers; print('CUDA', torch.cuda.is_available(), 'tf', transformers.__version__)"
echo "=== run ==="
PYTHONPATH=src python -m marker.run_instruct_disengage \
  --instruct-name Qwen/Qwen2.5-7B-Instruct \
  --max-new 200 2>&1 | tee /root/disengage.log
echo "EXITRC=${PIPESTATUS[0]}" | tee -a /root/disengage.log
echo "ALLDONE" | tee -a /root/disengage.log
EOS

echo "→ Creating instance..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
