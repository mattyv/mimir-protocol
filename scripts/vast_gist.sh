#!/usr/bin/env bash
# Stage-1 gist pilot on a Vast RTX 3090 (4-bit QLoRA Qwen2.5-7B).
#
# HF_TOKEN (from the launching shell) is injected as a container env var: it
# authenticates the 7B download AND the checkpoint push to the private repo.
# Read from env ONLY — never hardcoded. Disposable, repo-scoped; revoke after.
#
#   Shakedown (~$0.15, proves push+resume): STEPS=500 CKPT_EVERY=200 HF_TOKEN=... ./scripts/vast_gist.sh
#   Pilot     (20M tokens):                 HF_TOKEN=... ./scripts/vast_gist.sh
#   Resume:                                 RESUME=1 HF_TOKEN=... ./scripts/vast_gist.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=90
REPO="${REPO:-mattyvee/mimir-artifacts}"
STEPS="${STEPS:-4000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
TIMEOUT="${TIMEOUT:-8h}"
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"

echo "→ Searching RTX 3090 (reliability >= 0.98, inet_down >= 300)..."
OFFER_ID=$(vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=100 reliability>=0.98 inet_down>=300 rentable=true' \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; o=json.load(sys.stdin); print(o[0]['id']) if o else exit(1)")
echo "  offer $OFFER_ID"

read -r -d '' ONSTART <<EOS || true
exec > /proc/1/fd/1 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -2
cd /root/mimir-protocol
echo "=== pip ==="
pip install -q 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -3 \
  || { sleep 20; pip install -q 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -3; }
python -c "import torch,peft,bitsandbytes,datasets; print('CUDA', torch.cuda.is_available())" || { echo "SETUPFAIL"; echo "ALLDONE"; exit 1; }
echo "=== 7B download (authenticated) ==="
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B'); print('7B CACHED')" 2>&1 | tail -2
echo "=== run: gist pilot (steps=${STEPS} ckpt=${CKPT_EVERY} repo=${REPO}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_gist_pilot \
  --model-name Qwen/Qwen2.5-7B --repo ${REPO} \
  --max-steps ${STEPS} --eval-every ${EVAL_EVERY} --ckpt-every ${CKPT_EVERY} ${RESUME_FLAG} 2>&1 | tee /root/gist.log
echo "GIST_RC=\${PIPESTATUS[0]}" | tee -a /root/gist.log
echo "ALLDONE" | tee -a /root/gist.log
EOS

ENV_ARG=""
[ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"

echo "→ Creating instance...${HF_TOKEN:+ (HF auth on)}"
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
