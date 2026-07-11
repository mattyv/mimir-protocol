#!/usr/bin/env bash
# Poll a Vast instance to completion, then DESTROY it — the overnight-billing
# safety net (no node may bill unmanaged). Two independent stop conditions:
#   1. the run signals ALLDONE / SETUPFAIL in its logs -> destroy promptly
#   2. a HARD wall-clock cap -> destroy unconditionally, even if the log never
#      says ALLDONE (hung setup, dead node, lost stdout). This is the guarantee.
# Destruction is verified; if the first destroy call fails it retries.
#
#   ./scripts/vast_poll_destroy.sh <INSTANCE_ID> [MAX_MINUTES] [LOGFILE]
set -uo pipefail

ID="${1:?instance id required}"
MAX_MIN="${2:-200}"        # hard cap (minutes) — destroy no matter what past this
LOG="${3:-/tmp/vast_${ID}.log}"
POLL_SEC=60

start=$(date +%s)
echo "poll+destroy watching instance $ID (hard cap ${MAX_MIN}m, log $LOG)"

destroy() {
  echo "→ destroying instance $ID ..."
  for attempt in 1 2 3 4 5; do
    # `vastai destroy` prompts "[y/N]" and ABORTS on EOF non-interactively —
    # pipe `yes` so the confirmation is answered (this bit us once: run done,
    # node idle-billing, 5 aborted destroys).
    yes | vastai destroy instance "$ID" 2>&1 | tail -1
    sleep 5
    if ! vastai show instances --raw 2>/dev/null | python3 -c "import sys,json; ids=[str(i.get('id')) for i in json.load(sys.stdin)]; sys.exit(0 if '$ID' in ids else 1)"; then
      echo "✓ instance $ID gone (verified not in instance list)"
      return 0
    fi
    echo "  still present, retry $attempt ..."
    sleep $((attempt * 5))
  done
  echo "!! COULD NOT CONFIRM DESTROY of $ID — MANUAL 'vastai destroy instance $ID' REQUIRED"
  return 1
}

# is the instance in a healthy running/loading state? echoes yes|no|gone.
instance_state() {
  vastai show instances --raw 2>/dev/null | python3 -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: print('yes'); sys.exit()  # API hiccup -> don't act on it
m={str(i.get('id')): i for i in d}
i=m.get('$ID')
if i is None: print('gone'); sys.exit()
st=str(i.get('actual_status')); it=str(i.get('intended_status'))
print('no' if it=='stopped' or st in ('exited','offline') else 'yes')" 2>/dev/null || echo yes
}

reason=""; badstate=0
while true; do
  now=$(date +%s)
  mins=$(( (now - start) / 60 ))
  if [ "$mins" -ge "$MAX_MIN" ]; then
    reason="HARD CAP ${MAX_MIN}m reached"
    break
  fi
  # pull recent logs (best-effort; a not-yet-booted node returns nothing)
  vastai logs "$ID" --tail 400 > "$LOG.new" 2>/dev/null && mv "$LOG.new" "$LOG"
  if grep -q "ALLDONE" "$LOG" 2>/dev/null; then reason="ALLDONE"; break; fi
  if grep -q "SETUPFAIL" "$LOG" 2>/dev/null; then reason="SETUPFAIL"; break; fi
  # instance-state watch: a node whose onstart exited non-zero goes
  # intent=stopped and NEVER emits ALLDONE — the log-only poller would spin to
  # the hard cap. Require 2 consecutive bad reads (tolerate a transient).
  case "$(instance_state)" in
    gone) reason="INSTANCE GONE (already destroyed/expired)"; break ;;
    no)   badstate=$((badstate+1)); [ "$badstate" -ge 2 ] && { reason="INSTANCE STOPPED (onstart exited / preempted)"; break; } ;;
    *)    badstate=0 ;;
  esac
  sleep "$POLL_SEC"
done

echo "=== stop condition: $reason (after ${mins}m) ==="
echo "--- final log tail ---"
tail -25 "$LOG" 2>/dev/null || echo "(no log captured)"
echo "----------------------"
destroy
