#!/usr/bin/env bash
# STAGE-3 G7 TRAINING on a Vast GPU (pattern of vast_gist_dict.sh): trains the
# `g7` LoRA + GistVocab on a clean frozen 4-bit base to predict, per step, the
# 8-per-slot dictionary ids, then render the committed step's text -- see
# GIST_LM_PLAN.md "STAGES 2+3 DESIGN v3" for the sequence layout and gates.
#
# SMOKE_RUN=1 gives the ~$1 spend gate from the design doc (2k solutions, 1
# epoch, both eval blocks) BEFORE the full run -- pass = train gist_ce < 6.5
# AND below the bigram baseline; next-id accuracy > bigram; op-from-predicted
# > majority (see run_g7.smoke_verdict for the exact gate). This is a REAL
# GPU smoke (2k real solutions through the real 7B), distinct from --smoke
# (the fully offline, tiny-model, synthetic-corpus path this script never
# uses -- that one is for local CPU test runs only, see run_g7.py).
#
#   HF_TOKEN=... SMOKE_RUN=1 ./scripts/vast_g7.sh
#   GPU=RTX_4090 HF_TOKEN=... ./scripts/vast_g7.sh   # full stage-3 run
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=100
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
REPO="${REPO:-mattyvee/mimir-artifacts}"       # holds dict_kv_K4096.pt (stage-1) + the corpus (stage-2)
OUT_REPO="${OUT_REPO:-mattyvee/mimir-artifacts}"
CORPUS_SUBDIR="${CORPUS_SUBDIR:-gist_corpus_v0}"
DICT_PATH="${DICT_PATH:-gist_dict/dict_kv_K4096.pt}"
GPU="${GPU:-RTX_4090}"
SMOKE_RUN="${SMOKE_RUN:-}"

if [ -n "$SMOKE_RUN" ]; then
  N_SOLUTIONS="${N_SOLUTIONS:-2000}"
  EPOCHS="${EPOCHS:-1}"
  EVAL_BLOCKS="${EVAL_BLOCKS:-1,2}"
  TIMEOUT="${TIMEOUT:-150m}"
  POLLER_CAP="${POLLER_CAP:-190}"
else
  N_SOLUTIONS="${N_SOLUTIONS:-}"                 # unset = the full pushed corpus
  EPOCHS="${EPOCHS:-2}"
  EVAL_BLOCKS="${EVAL_BLOCKS:-1,2}"
  TIMEOUT="${TIMEOUT:-300m}"
  POLLER_CAP="${POLLER_CAP:-340}"
fi
N_SOLUTIONS_FLAG=""
[ -n "$N_SOLUTIONS" ] && N_SOLUTIONS_FLAG="--n-solutions ${N_SOLUTIONS}"

# cpu_ram is in GB in the vast search API (see vast_gist_dict.sh's note); a
# 4-bit 7B + LoRA + GistVocab (~37M fp32 masters) + a few thousand packed
# sequences comfortably fit in 32GB host RAM.
echo "→ Searching ${GPU} (rel>=0.98 inet_down>=500 inet_up>=200 cuda>=12.4, gpu_ram>=23 cpu_ram>=32 disk>=100)..."
OFFER_ID=""
for try in 1 2 3 4 5; do
  OFFER_ID=$(vastai search offers \
    "gpu_name=${GPU} num_gpus=1 gpu_ram>=23 cpu_ram>=32 cuda_vers>=12.4 disk_space>=100 reliability>=0.98 inet_down>=500 inet_up>=200 rentable=true" \
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( while true; do echo "  ...setup heartbeat \$(date -u +%H:%M:%S)"; sleep 40; done ) &
HB=\$!
curl -sS -m 10 -o /dev/null https://huggingface.co || { kill \$HB; echo "SETUPFAIL (huggingface.co unreachable from this node — CN geolocation; relaunch)"; echo "ALLDONE"; exit 1; }
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -2
cd /root/mimir-protocol
echo "=== pip ==="
for pt in 1 2 3 4; do
  pip install --timeout 100 --retries 5 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -3 && python -c 'import peft,bitsandbytes,transformers,datasets,safetensors' 2>/dev/null && break
  echo "  pip attempt \$pt failed (flaky node network), retrying in 15s..."; sleep 15
done
python -c 'import peft,bitsandbytes,transformers,datasets,safetensors' || { kill \$HB; echo "SETUPFAIL (pip could not install deps after retries; relaunch)"; echo "ALLDONE"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA True')" || { kill \$HB; echo "SETUPFAIL (no CUDA — driver/image mismatch, e.g. error 804; relaunch)"; echo "ALLDONE"; exit 1; }
echo "=== HF token check (fail fast) ==="
python -c "from huggingface_hub import whoami; print('HF auth ok:', whoami().get('name'))" \
  || { kill \$HB; echo "SETUPFAIL (bad/revoked HF token)"; echo "ALLDONE"; exit 1; }
echo "=== download ${MODEL} (authenticated, 20min cap) ==="
timeout 1200 python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}'); print('MODEL CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (download too slow)"; echo "ALLDONE"; exit 1; }
echo "=== dictionary + corpus (from ${REPO}) ==="
timeout 900 python -c "from huggingface_hub import hf_hub_download; hf_hub_download('${REPO}', '${DICT_PATH}', local_dir='/root/mimir-protocol'); print('DICT CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (dictionary download failed)"; echo "ALLDONE"; exit 1; }
timeout 1800 python -c "from huggingface_hub import snapshot_download; snapshot_download('${REPO}', allow_patterns='${CORPUS_SUBDIR}/*.jsonl', local_dir='/root/mimir-protocol'); print('CORPUS CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (corpus download failed)"; echo "ALLDONE"; exit 1; }
echo "=== G7 TRAIN (smoke_run=${SMOKE_RUN:-0} n_solutions=${N_SOLUTIONS:-all} epochs=${EPOCHS} eval_blocks=${EVAL_BLOCKS}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_g7 \
  --model-name "${MODEL}" --dict-path "${DICT_PATH}" --corpus-dir "${CORPUS_SUBDIR}" \
  --out-repo "${OUT_REPO}" --epochs "${EPOCHS}" --eval-blocks "${EVAL_BLOCKS}" ${N_SOLUTIONS_FLAG} \
  2>&1 | tee /root/g7.log
echo "G7_RC=\${PIPESTATUS[0]}" | tee -a /root/g7.log
kill \$HB 2>/dev/null
echo "ALLDONE" | tee -a /root/g7.log
EOS

ENV_ARG=""
[ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"

echo "→ Creating instance...${HF_TOKEN:+ (HF auth on)}"
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
# poller hard cap MUST exceed TIMEOUT + setup (~20m), or the poller kills the
# node while the run is still inside its own budget
echo "→ arm the poller:  bash scripts/vast_poll_destroy.sh $INSTANCE_ID ${POLLER_CAP}"
