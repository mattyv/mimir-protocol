#!/usr/bin/env bash
# Stage-2 CoT run on a Vast RTX 3090: TWO gated steps on ONE node.
#   1. reason_check ON THIS RUN'S OWN CORPUS — does the FineWeb-trained gist
#      encode these reasoning units? gap_closed >= GATE (default 0.4) ->
#      proceed; below -> STOP (Stage-1 re-fit is the next fork, not this run).
#      ~$0.30 if it stops. Always runs (Fable: an unchecked corpus makes a weak
#      Stage-2 result uninterpretable — encoder-blind vs succession-failed).
#   2. CoT Stage-2 run — encode reasoning traces into gist sequences, train the
#      next-thought predictor, eval on the registered gates (recall@5_128 +
#      within-doc succession control). Smaller predictor + finer eval than the
#      raw-text run (which overfit by step 500).
#
#   A (GSM8K):  HF_TOKEN=... ./scripts/vast_cot.sh
#   B (OpenR1): DATASET=open-r1/OpenR1-Math-220k UNIT=sentence NDOCS=2000 \
#               WINDOW=6 OUTSUBDIR=stage2_cot_openr1 HF_TOKEN=... ./scripts/vast_cot.sh
#
# HF_TOKEN (from env) authenticates the 7B download + artifact push. Never
# hardcoded. Disposable, repo-scoped; revoke after the campaign.
#
#   ./scripts/vast_cot.sh                 # HF_TOKEN=... in env
#   NDOCS=3000 GATE=0.4 HF_TOKEN=... ./scripts/vast_cot.sh
set -euo pipefail

IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=90
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
REPO="${REPO:-mattyvee/mimir-artifacts}"
DATASET="${DATASET:-openai/gsm8k}"
DSCONFIG="${DSCONFIG:-}"       # HF config (gsm8k auto-defaults to 'main')
TEXTFIELD="${TEXTFIELD:-}"     # row field (gsm8k->answer, else->solution)
UNIT="${UNIT:-line}"           # line (gsm8k step-per-line) | sentence (long solutions)
SPLIT="${SPLIT:-}"             # reason_check split (gsm8k->test, else->train)
NPROB="${NPROB:-150}"          # reason_check problems
GATE="${GATE:-0.4}"            # gap_closed threshold to proceed
NDOCS="${NDOCS:-7000}"         # STREAMED docs, not kept: min_sents=window+1 drops
                               # ~55% of GSM8K (median 3 steps) -> 7000 keeps ~3100.
                               # More kept data is the cheapest overfit fix.
STEPS="${STEPS:-4000}"
WINDOW="${WINDOW:-3}"          # reasoning traces are short
DMODEL="${DMODEL:-384}"        # smaller than the raw run's 640 (overfit)
LAYERS="${LAYERS:-4}"
EVAL_EVERY="${EVAL_EVERY:-250}"
OUTSUBDIR="${OUTSUBDIR:-stage2_predictor}"  # distinct per parallel run (no clobber)
TIMEOUT="${TIMEOUT:-3h}"

echo "→ Searching RTX 3090 (reliability >= 0.98, inet_down >= 500)..."
OFFER_ID=$(vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 gpu_ram>=23 cuda_vers>=12.0 disk_space>=100 reliability>=0.98 inet_down>=500 rentable=true' \
  --order dph_total --limit 1 --raw 2>/dev/null | \
  python3 -c "import sys,json; o=json.load(sys.stdin); print(o[0]['id']) if o else exit(1)")
echo "  offer $OFFER_ID"

read -r -d '' ONSTART <<EOS || true
exec > /proc/1/fd/1 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
( while true; do echo "  ...setup heartbeat \$(date -u +%H:%M:%S)"; sleep 40; done ) &
HB=\$!
cd /root
echo "=== clone ==="
git clone --branch claude/project-review-6rx97z --single-branch https://github.com/mattyv/mimir-protocol.git 2>&1 | tail -2
cd /root/mimir-protocol
echo "=== pip ==="
pip install 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -4 \
  || { sleep 20; pip install 'transformers>=4.45,<5' 'accelerate>=1.0' peft bitsandbytes datasets sentencepiece hf_transfer safetensors 2>&1 | tail -4; }
python -c "import torch,peft,bitsandbytes,datasets; print('CUDA', torch.cuda.is_available())" || { kill \$HB; echo "SETUPFAIL"; echo "ALLDONE"; exit 1; }
echo "=== HF token check (fail fast) ==="
python -c "from huggingface_hub import whoami; print('HF auth ok:', whoami().get('name'))" \
  || { kill \$HB; echo "SETUPFAIL (bad/revoked HF token)"; echo "ALLDONE"; exit 1; }
echo "=== download ${MODEL} (authenticated, 20min cap) ==="
timeout 1200 python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}'); print('MODEL CACHED')" 2>&1 | tail -2 \
  || { kill \$HB; echo "SETUPFAIL (download too slow)"; echo "ALLDONE"; exit 1; }

DS_ARGS="--dataset ${DATASET} --unit ${UNIT}"
[ -n "${DSCONFIG}" ] && DS_ARGS="\$DS_ARGS --dataset-config ${DSCONFIG}"
[ -n "${TEXTFIELD}" ] && DS_ARGS="\$DS_ARGS --text-field ${TEXTFIELD}"
CHECK_ARGS="\$DS_ARGS"
[ -n "${SPLIT}" ] && CHECK_ARGS="\$CHECK_ARGS --split ${SPLIT}"

echo "=== STEP 1: reason_check ON THIS RUN'S CORPUS (${DATASET}, ${NPROB} problems) ==="
# encoder gate on the run's own distribution — a weak Stage-2 result on an
# unchecked corpus can't separate encoder-blind from succession-failed.
timeout 30m env PYTHONPATH=src python -u -m marker.reason_check \
  --model-name ${MODEL} --repo ${REPO} --n-problems ${NPROB} \$CHECK_ARGS 2>&1 | tee /root/reason.log
GC=\$(grep 'gap_closed=' /root/reason.log | tail -1 | sed -E 's/.*gap_closed=([-0-9.]+).*/\1/')
echo "REASON_GAP_CLOSED=\${GC:-none}"
PASS=\$(python3 -c "print('yes' if float('\${GC:-0}') >= ${GATE} else 'no')" 2>/dev/null || echo no)
if [ "\$PASS" != "yes" ]; then
  kill \$HB 2>/dev/null
  echo "REASON GATE FAIL (gap_closed=\${GC:-none} < ${GATE}) — encoder is blind to this corpus; Stage-1 re-fit is the next fork, NOT this run."
  echo "ALLDONE"
  exit 0
fi
echo "REASON GATE PASS (gap_closed=\${GC} >= ${GATE}) — proceeding to CoT Stage-2 run."
echo "=== STEP 2: CoT Stage-2 run (dataset=${DATASET} unit=${UNIT} ndocs=${NDOCS} window=${WINDOW} dmodel=${DMODEL} layers=${LAYERS}) ==="
timeout ${TIMEOUT} env PYTHONPATH=src python -u -m marker.run_stage2 \
  --model-name ${MODEL} --repo ${REPO} --out-repo ${REPO} --out-subdir ${OUTSUBDIR} \
  --corpus cot \$DS_ARGS \
  --n-docs ${NDOCS} --steps ${STEPS} --window ${WINDOW} \
  --d-model ${DMODEL} --layers ${LAYERS} --eval-every ${EVAL_EVERY} 2>&1 | tee /root/cot.log
kill \$HB 2>/dev/null
echo "COT_RC=\${PIPESTATUS[0]}" | tee -a /root/cot.log
echo "ALLDONE" | tee -a /root/cot.log
EOS

ENV_ARG=""
[ -n "${HF_TOKEN:-}" ] && ENV_ARG="-e HF_TOKEN=${HF_TOKEN}"

echo "→ Creating instance...${HF_TOKEN:+ (HF auth on)}"
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" --disk "$DISK_GB" --onstart-cmd "$ONSTART" --env "$ENV_ARG" --raw 2>/dev/null | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "INSTANCE $INSTANCE_ID"
echo "→ arm the poller:  bash scripts/vast_poll_destroy.sh $INSTANCE_ID 200"
