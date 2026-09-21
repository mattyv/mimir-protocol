#!/usr/bin/env bash
# STAGE 2 CORPUS TOKENIZER on a Vast GPU: turns GSM8K train + OpenR1 solutions
# into the stage-2 gist corpus (question + per-step [8]-id groups + answer),
# doc-disjoint from every eval set stage 1/3 read. See
# scratchpad/stage2_build_order.md and src/marker/run_tokenize_corpus.py for
# the full spec. Resumable: shards already pushed to ${OUT_REPO}/${OUTSUB}
# are skipped, never re-encoded (a bigger --n-openr1 later is a pure resume).
#
#   HF_TOKEN=... ./scripts/vast_tokenize_corpus.sh
#   NGSM8K=7473 NOPENR1=12500 HF_TOKEN=... ./scripts/vast_tokenize_corpus.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=100
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
REPO="${REPO:-mattyvee/mimir-artifacts}"       # stage-1 adapter + gist_dict/dict_kv_K4096.pt
OUT_REPO="${OUT_REPO:-$REPO}"
GSM8K_DATASET="${GSM8K_DATASET:-openai/gsm8k}"
OPENR1_DATASET="${OPENR1_DATASET:-open-r1/OpenR1-Math-220k}"
DICT_SUBDIR="${DICT_SUBDIR:-gist_dict}"
DICT_NAME="${DICT_NAME:-kv_K4096}"
NGSM8K="${NGSM8K:-7473}"                       # GSM8K train candidates (7473 = all)
NOPENR1="${NOPENR1:-12500}"
SHARD="${SHARD:-2000}"                         # solutions per JSONL shard
SEQCAP="${SEQCAP:-512}"
MAXGROUPS="${MAXGROUPS:-4}"
OUTSUB="${OUTSUB:-gist_corpus_K4096}"
GPU="${GPU:-RTX_3090}"
# Wall-clock budget (GIST_LM_PLAN.md "Corpus v0": ~57k solutions x ~8 steps
# ~= 450k single-span encodes; 40k stage-1 fit steps took ~60-75m on a 3090 ->
# ~12-14h for the full 57k corpus. This script's default (7.5k GSM8K + 12.5k
# OpenR1 ~= 20k solutions, the v0 launch size) is a fraction of that -- 7h
# covers it with room for the two dataset streams + per-shard push overhead.
# Shards are pushed (with the manifest) the moment each one fills, so a
# timeout can only cost work not yet shard-complete -- never the whole run.
TIMEOUT="${TIMEOUT:-420m}"
# poller hard cap must exceed TIMEOUT + setup (~20m'ish for clone/pip/model
# download), matching vast_gist_dict.sh's own convention
POLLER_CAP="${POLLER_CAP:-480}"

# cpu_ram is in GB in the vast search API (CLAUDE.md's own "cpu_ram>=<GB*1024>"
# guidance predates the fix -- this script follows the CORRECTED, currently-
# working convention vast_gist_dict.sh/vast_summary_probe.sh already use, a
# plain GB number). The tokenizer's own on-disk cache (per-piece JSON files)
# and shard buffers are tiny next to the resident 7B; 32GB comfortably covers
# the peak the same way it does for run_gist_dict's fit-shard build.
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
echo "=== TOKENIZE CORPUS (n-gsm8k=${NGSM8K} n-openr1=${NOPENR1} shard-size=${SHARD} out-subdir=${OUTSUB}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_tokenize_corpus \
  --model-name "${MODEL}" --repo "${REPO}" --out-repo "${OUT_REPO}" \
  --gsm8k-dataset "${GSM8K_DATASET}" --openr1-dataset "${OPENR1_DATASET}" \
  --dict-subdir "${DICT_SUBDIR}" --dict-name "${DICT_NAME}" \
  --n-gsm8k "${NGSM8K}" --n-openr1 "${NOPENR1}" --shard-size "${SHARD}" \
  --seq-cap "${SEQCAP}" --max-groups-per-step "${MAXGROUPS}" --out-subdir "${OUTSUB}" \
  2>&1 | tee /root/tokenize_corpus.log
echo "TOKENIZE_CORPUS_RC=\${PIPESTATUS[0]}" | tee -a /root/tokenize_corpus.log
kill \$HB 2>/dev/null
echo "ALLDONE" | tee -a /root/tokenize_corpus.log
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
