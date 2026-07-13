#!/usr/bin/env bash
# Fast-lane confidence probe on a Vast RTX 3090 — the FASTLANE_PLAN gate.
# ONE cheap step (~$0.30): reuse the ALREADY-TRAINED predictor (no training),
# encode held-out reasoning chains, ask whether an inference-time confidence
# signal separates the predictor's correct next-thought guesses from its wrong
# ones AT A USABLE SKIP-RATE (precision@coverage, doc-split confirm).
#
# Defaults probe the SAME distribution stage2_cot_openr1 was trained on
# (open-r1/OpenR1-Math-220k, unit=sentence), SKIPPING the first 2000 docs (its
# training/eval range) so the chains are genuinely held out. window=4 (not the
# training window 6) keeps the within-doc pools non-degenerate (Fable re-review:
# floor(16/6)<=2 windows/doc collapses the pools; floor(16/4)=4 gives real pools).
#
#   HF_TOKEN=... ./scripts/vast_confidence.sh
#   NDOCS=600 WINDOW=4 SKIPDOCS=2000 HF_TOKEN=... ./scripts/vast_confidence.sh
#
# HF_TOKEN authenticates the 7B download + predictor fetch + manifest push.
# Disposable, repo-scoped; revoke after the campaign. The probe pushes its
# manifest to REPO/confidence_probe (durable — survives node death).
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=90
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
REPO="${REPO:-mattyvee/mimir-artifacts}"        # Stage-1 gist adapter + predictor + push
SUBDIR="${SUBDIR:-stage2_cot_openr1}"           # which trained predictor to probe
DATASET="${DATASET:-open-r1/OpenR1-Math-220k}"  # match the predictor's training corpus
UNIT="${UNIT:-sentence}"                        # openr1 was encoded sentence-unit
NDOCS="${NDOCS:-500}"
SKIPDOCS="${SKIPDOCS:-2000}"                    # skip stage2_cot_openr1's train range (NDOCS=2000)
WINDOW="${WINDOW:-4}"                           # non-degenerate within-doc pools (see header)
NDROPOUT="${NDROPOUT:-8}"
TIMEOUT="${TIMEOUT:-90m}"

echo "→ Searching RTX 3090 (rel>=0.98 inet>=500 cuda>=12.4)..."
OFFER_ID=""
for try in 1 2 3 4 5; do
  OFFER_ID=$(vastai search offers \
    'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.4 disk_space>=100 reliability>=0.98 inet_down>=500 rentable=true' \
    --order 'reliability-' --limit 1 --raw 2>/dev/null | \
    python3 -c "import sys,json
try: o=json.load(sys.stdin); print(o[0]['id'] if o else '')
except Exception: print('')" 2>/dev/null)
  [ -n "$OFFER_ID" ] && break
  echo "  no offer (try $try/5), retrying in 10s..."; sleep 10
done
[ -z "$OFFER_ID" ] && { echo "NO OFFERS after retries — pool too thin, try later"; exit 1; }
echo "  offer $OFFER_ID"

read -r -d '' ONSTART <<EOS || true
exec > /proc/1/fd/1 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=60
( while true; do echo "  ...setup heartbeat \$(date -u +%H:%M:%S)"; sleep 40; done ) &
HB=\$!
curl -sS -m 10 -o /dev/null https://huggingface.co || { kill \$HB; echo "SETUPFAIL (huggingface.co unreachable from this node — CN geolocation; relaunch)"; echo "ALLDONE"; exit 1; }
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -2
cd /root/mimir-protocol
echo "=== pip ==="
for pt in 1 2 3 4; do
  pip install --timeout 100 --retries 5 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -3 && python -c 'import peft,bitsandbytes,transformers,datasets' 2>/dev/null && break
  echo "  pip attempt \$pt failed (flaky node network), retrying in 15s..."; sleep 15
done
python -c 'import peft,bitsandbytes,transformers,datasets' || { kill \$HB; echo "SETUPFAIL (pip could not install deps after retries; relaunch)"; echo "ALLDONE"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA True')" || { kill \$HB; echo "SETUPFAIL (no CUDA — driver/image mismatch, e.g. error 804; relaunch)"; echo "ALLDONE"; exit 1; }
echo "=== HF token check (fail fast) ==="
python -c "from huggingface_hub import whoami; print('HF auth ok:', whoami().get('name'))" \
  || { kill \$HB; echo "SETUPFAIL (bad/revoked HF token)"; echo "ALLDONE"; exit 1; }
echo "=== download ${MODEL} (authenticated, 20min cap) ==="
timeout 1200 python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}'); print('MODEL CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (download too slow)"; echo "ALLDONE"; exit 1; }
echo "=== CONFIDENCE PROBE (subdir=${SUBDIR} dataset=${DATASET} unit=${UNIT} ndocs=${NDOCS} skip=${SKIPDOCS} window=${WINDOW}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_confidence \
  --model-name ${MODEL} --repo ${REPO} \
  --artifacts-repo ${REPO} --subdir ${SUBDIR} --out-repo ${REPO} \
  --dataset ${DATASET} --unit ${UNIT} \
  --n-docs ${NDOCS} --skip-docs ${SKIPDOCS} --window ${WINDOW} --n-dropout ${NDROPOUT} 2>&1 | tee /root/conf.log
echo "PROBE_RC=\${PIPESTATUS[0]}" | tee -a /root/conf.log
kill \$HB 2>/dev/null
echo "ALLDONE" | tee -a /root/conf.log
EOS

ENV_ARG=""
[ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"

echo "→ Creating instance...${HF_TOKEN:+ (HF auth on)}"
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
echo "→ arm the poller:  bash scripts/vast_poll_destroy.sh $INSTANCE_ID 200"
