#!/usr/bin/env bash
# Unattended watcher for the Phase 4 Vast run.
# Usage: ./scripts/vast_phase4_watch.sh INSTANCE_ID SSH_HOST SSH_PORT

set -u
INSTANCE_ID="$1"
SSH_HOST="$2"
SSH_PORT="$3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT root@$SSH_HOST"
SUMMARY="$REPO_DIR/runs/phase4-summary.txt"
mkdir -p "$REPO_DIR/runs" "$REPO_DIR/value_token_7b"

echo "watching instance $INSTANCE_ID for DONE (max 4h)" > "$SUMMARY"

for _ in $(seq 1 240); do
  if $SSH_CMD "grep -q DONE /tmp/phase4.log" 2>/dev/null; then
    echo "DONE detected $(date)" >> "$SUMMARY"
    break
  fi
  sleep 60
done

TS=$(date +%Y%m%d-%H%M)

scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "root@$SSH_HOST:/root/mimir-protocol/value_token_out/*.pt" \
  "$REPO_DIR/value_token_7b/" 2>>"$SUMMARY" || true
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "root@$SSH_HOST:/tmp/phase4.log" \
  "$REPO_DIR/runs/phase4-${TS}.log" 2>>"$SUMMARY" || true

{
  echo "--- value_token_7b ---"
  ls -lh "$REPO_DIR/value_token_7b/"*.pt 2>/dev/null
  echo "--- score summary (tail of log) ---"
  tail -30 "$REPO_DIR/runs/phase4-${TS}.log" 2>/dev/null
} >> "$SUMMARY"

yes | vastai destroy instance "$INSTANCE_ID" >> "$SUMMARY" 2>&1
echo "destroyed $INSTANCE_ID $(date)" >> "$SUMMARY"
vastai show instances --raw 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print('remaining:', len(d))" >> "$SUMMARY"
