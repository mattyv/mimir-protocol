#!/bin/bash
# ponytail: polls the 4 gist-k training nodes, commits+pushes their logs to
# the vast-logs branch on every change so another session can `git pull` them.
# No auto-shutdown of instances — that's a separate, explicit decision.
set -u
cd /tmp/vast-logs-wt
SSHOPT=(-o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes)
NODES=("18250 ssh5.vast.ai n1" "23804 ssh2.vast.ai n2" "24118 ssh6.vast.ai n3" "24126 ssh5.vast.ai n4")
STATUS=/tmp/vast_poll_status.txt
DONE=""  # space-separated tags; bash 3.2 (macOS default) has no associative arrays

log(){ echo "[$(date -u +%H:%M:%S)Z] $*" | tee -a "$STATUS"; }

pull_one() { # port host tag
  local port=$1 host=$2 tag=$3
  ssh "${SSHOPT[@]}" -p "$port" "root@$host" \
    "cat /root/ksweep.log 2>/dev/null" > "runs/vast_logs/${tag}.log" 2>/dev/null
}

alive_one() { # port host -> prints pid or empty
  local port=$1 host=$2
  ssh "${SSHOPT[@]}" -p "$port" "root@$host" "pgrep -f run_gist_pilot | head -1" 2>/dev/null
}

commit_if_changed() {
  git add runs/vast_logs/*.log 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "vast logs: snapshot $(date -u +%Y-%m-%dT%H:%MZ)"
    git push -q origin vast-logs 2>&1 | tail -5
    log "pushed log snapshot"
  fi
}

is_done() { case " $DONE " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
count_done() { echo $DONE | wc -w | tr -d ' '; }

log "poller starting"
while [ "$(count_done)" -lt 4 ]; do
  for n in "${NODES[@]}"; do
    read -r port host tag <<< "$n"
    is_done "$tag" && continue
    pull_one "$port" "$host" "$tag"
    pid=$(alive_one "$port" "$host")
    if [ -z "$pid" ]; then
      reachable=$(ssh "${SSHOPT[@]}" -p "$port" "root@$host" "echo ok" 2>/dev/null)
      if [ "$reachable" = "ok" ]; then
        log "$tag: FINISHED (process exited)"
        DONE="$DONE $tag"
      else
        log "$tag: unreachable, will retry"
      fi
    else
      log "$tag: running (pid $pid)"
    fi
  done
  commit_if_changed
  [ "$(count_done)" -lt 4 ] && sleep 900
done
log "ALL NODES DONE. Logs on vast-logs branch. Instances still running/billing — awaiting your shutdown decision."
