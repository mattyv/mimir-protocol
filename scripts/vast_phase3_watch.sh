#!/usr/bin/env bash
# Unattended watcher for the Phase 3 Vast run. Polls for DONE, captures the
# single-vector artifacts + the comparison log, then DESTROYS the instance
# unconditionally (billing must stop even if scp fails).
#
# Usage: ./scripts/vast_phase3_watch.sh INSTANCE_ID SSH_HOST SSH_PORT

set -u
INSTANCE_ID="$1"
SSH_HOST="$2"
SSH_PORT="$3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT root@$SSH_HOST"
SUMMARY="$REPO_DIR/runs/phase3-summary.txt"
mkdir -p "$REPO_DIR/runs" "$REPO_DIR/single_vector_7b"

echo "watching instance $INSTANCE_ID for DONE (max 3h)" > "$SUMMARY"

# Poll up to 3 hours.
for _ in $(seq 1 180); do
  if $SSH "grep -q DONE /tmp/phase3.log" 2>/dev/null; then
    echo "DONE detected $(date)" >> "$SUMMARY"
    break
  fi
  sleep 60
done

TS=$(date +%Y%m%d-%H%M)
# Capture (best effort).
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "root@$SSH_HOST:/root/mimir-protocol/single_vector_out/*.pt" \
  "$REPO_DIR/single_vector_7b/" 2>>"$SUMMARY" || true
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "root@$SSH_HOST:/tmp/phase3.log" \
  "$REPO_DIR/runs/phase3-${TS}.log" 2>>"$SUMMARY" || true

{
  echo "--- single_vector_7b ---"
  ls -lh "$REPO_DIR/single_vector_7b/"*.pt 2>/dev/null
  echo "--- score summary (tail of log) ---"
  tail -20 "$REPO_DIR/runs/phase3-${TS}.log" 2>/dev/null
} >> "$SUMMARY"

# Destroy unconditionally.
yes | vastai destroy instance "$INSTANCE_ID" >> "$SUMMARY" 2>&1
echo "destroyed $INSTANCE_ID $(date)" >> "$SUMMARY"
vastai show instances --raw 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print('remaining:', len(d))" >> "$SUMMARY"
