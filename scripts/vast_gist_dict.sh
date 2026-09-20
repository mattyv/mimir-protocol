#!/usr/bin/env bash
# STAGE-1 GIST DICTIONARY FIDELITY on a Vast GPU: snap each step's canonical
# gist-KV to the nearest entry of a small per-slot dictionary -- can the
# frozen reader still write the step correctly, and is the operation still
# readable from the 8 IDs? See scratchpad/gist_dict_stage1_spec.md for the
# full cell table, dictionary configs, and the stage1_verdict gate order.
# The manifest emits the numbers; the human + Fable read the verdict.
#
#   HF_TOKEN=... ./scripts/vast_gist_dict.sh
#   GPU=RTX_4090 NFIT=20000 NEVAL=150 HF_TOKEN=... ./scripts/vast_gist_dict.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=100
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
REPO="${REPO:-mattyvee/mimir-artifacts}"       # stage-1 adapter + render_adapter_oneform
DATASET="${DATASET:-openai/gsm8k}"
NFIT="${NFIT:-40000}"                          # fit-set steps (GSM8K train + OpenR1), doc-disjoint from eval
NEVAL="${NEVAL:-200}"                          # GSM8K TEST eval steps (Wilson CI ~±0.04 at p≈0.9)
KS="${KS:-256,1024,4096}"                      # comma-separated K for the plain kv_K* configs
GPU="${GPU:-RTX_3090}"
# Wall-clock budget (node 51724858 measured): 40k fit encodes ~60-80m on the
# 4-bit 7B + chunked GPU k-means ~15-35m (the fit-shard push, ~20GB fp16 ≈
# 13-17m at inet_up>=200, runs in a BACKGROUND thread overlapped with it) +
# native scored once + 4 GPU-eval configs x (200+150) steps x 2 generated
# conditions (~1.5-2h) + CPU diagnostics (~30m) ≈ 3.5-4.5h. Each dictionary
# is pushed the moment it is built and the shards right after the encode, so
# a timeout or OOM can only cost work not yet done -- never the encode.
TIMEOUT="${TIMEOUT:-330m}"
# LOAD_SHARDS=1 resumes from the pushed fit-shard cache on ${REPO} (skips the
# encode; requires a previous run to have gotten past the shard push).
LOAD_SHARDS="${LOAD_SHARDS:-}"
LOAD_DICTS="${LOAD_DICTS:-}"      # LOAD_DICTS=1 also skips k-means (dicts pushed per-config)
RESUME_FLAG=""
[ -n "$LOAD_SHARDS" ] && RESUME_FLAG="--load-shards"
[ -n "$LOAD_DICTS" ] && RESUME_FLAG="${RESUME_FLAG} --load-dicts"

# cpu_ram is in GB in the vast search API (CLAUDE.md's own "cpu_ram>=<GB*1024>"
# guidance predates the fix in commit "vast_render: cpu_ram search clause is
# in GB, not MB" -- this script follows the CORRECTED, currently-working
# convention vast_summary_probe.sh already uses, a plain GB number, not
# GB*1024). 40k fit steps' fp16 KV shards run ~2.3GB PER SLOT (never all 8
# slots at once, see gist_dict.py/run_gist_dict.py's shard-streaming design)
# plus small readouts -- 32GB host RAM comfortably covers the peak (k-means
# itself now runs on fp16 GPU tensors with chunked fp32 math, ~4GB VRAM on
# top of the resident model; see gist_dict._CHUNK_ROWS).
# inet_up>=200: the background fit-shard push uploads ~20GB fp16 (≈13-17min
# at 200Mbps, overlapped with k-means) plus ~6GB of dictionaries; a node
# with fast download but a trickle upload would burn the eval budget on it.
echo "→ Searching ${GPU} (rel>=0.98 inet_down>=500 inet_up>=200 cuda>=12.4, cpu_ram>=32GB disk>=100)..."
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
echo "=== GIST DICTIONARY FIDELITY (dataset=${DATASET} n-fit=${NFIT} n-eval=${NEVAL} ks=${KS}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_gist_dict \
  --model-name "${MODEL}" --repo "${REPO}" --out-repo "${REPO}" \
  --dataset "${DATASET}" --n-fit "${NFIT}" --n-eval "${NEVAL}" --ks "${KS}" \
  --push-shards ${RESUME_FLAG} \
  --eval --diagnose 2>&1 | tee /root/gist_dict.log
echo "GIST_DICT_RC=\${PIPESTATUS[0]}" | tee -a /root/gist_dict.log
kill \$HB 2>/dev/null
echo "ALLDONE" | tee -a /root/gist_dict.log
EOS

ENV_ARG=""
[ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"

echo "→ Creating instance...${HF_TOKEN:+ (HF auth on)}"
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
# poller hard cap MUST exceed TIMEOUT (330m) + setup (~20m), or the poller
# kills the node while the run is still inside its own budget
echo "→ arm the poller:  bash scripts/vast_poll_destroy.sh $INSTANCE_ID 380"
