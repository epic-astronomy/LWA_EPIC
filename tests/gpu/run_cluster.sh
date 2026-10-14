#!/usr/bin/env bash
# Run the epic++ GPU tests and nvbench benchmarks across the worker nodes.
#
# The build directory is on /FileStore, which every worker mounts, so the
# executables built once (e.g. by run_on_worker.sh on epic1) run in place.
# Benchmark configurations are split into shards (benchmark x WeightMode) and
# handed out round-robin to the nodes whose GPU is idle; nodes with a running
# GPU process (e.g. a live pipeline) are skipped.
#
# usage: tests/gpu/run_cluster.sh [build dir] [hosts...]
#   default hosts: epic1 epic2 epic3 epic4 192.168.40.25 (epic5) epic6
# Results: <build dir>/results/cluster-<timestamp>/{<host>-shard<i>.md,json,log}, merged.md
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-$here/../../build-gpu-tests-epic1}"
shift || true
hosts=("$@")
[[ ${#hosts[@]} -eq 0 ]] && hosts=(epic1 epic2 epic3 epic4 192.168.40.25 epic6)
for exe in bench_imager test_imager gulp_budget; do
  [[ -x "$build/$exe" ]] || { echo "missing $build/$exe; build first (run_on_worker.sh)"; exit 2; }
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$build/results/cluster-$stamp"
mkdir -p "$out"

ssh_cmd() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$@"; }

# Nodes with an idle GPU
idle=()
for h in "${hosts[@]}"; do
  apps="$(ssh_cmd "$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader' 2>/dev/null)" || { echo "skip $h: unreachable"; continue; }
  if [[ -n "$apps" ]]; then echo "skip $h: GPU busy (pids $apps)"; continue; fi
  idle+=("$h")
done
[[ ${#idle[@]} -eq 0 ]] && { echo "no idle GPU nodes"; exit 1; }
echo "nodes: ${idle[*]}"

# Shards: one per benchmark x weight mode (both Wproj values inside)
shards=()
for b in epic_imager_kernel epic_gulp; do
  for m in -1 0 1 2 4; do shards+=("-b $b -a WeightMode=[$m]"); done
done

# Each node runs the correctness tests once, then its shards in sequence
pids=()
for i in "${!idle[@]}"; do
  h="${idle[$i]}"
  (
    ssh_cmd "$h" "$build/test_imager --quick" > "$out/$h-tests.log" 2>&1
    echo "$h tests: $(tail -1 "$out/$h-tests.log")"
    for ((s = i; s < ${#shards[@]}; s += ${#idle[@]})); do
      tag="$h-shard$s"
      ssh_cmd "$h" "$build/bench_imager ${shards[$s]} --json $out/$tag.json --md $out/$tag.md" > "$out/$tag.log" 2>&1
      echo "$h shard $s (${shards[$s]}): exit $?"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# Merge: one table per benchmark with the host that measured each row
{
  echo "# epic++ imaging kernel, nvbench across nodes ($stamp)"
  for b in epic_imager_kernel epic_gulp; do
    echo; echo "## $b"; header=0
    for f in "$out"/*-shard*.md; do
      host="$(basename "$f" | sed 's/-shard[0-9]*\.md//')"
      awk -v b="$b" -v host="$host" -v hdr="$header" '
        /^## / {inb = ($2 == b); next}
        inb && /^\|/ {
          if ($0 ~ /^\| *Support/) { if (hdr == 0) print "| Host " $0; next }
          if ($0 ~ /^\|-/) { if (hdr == 0) { print "|------" $0; hdr = 1 } ; next }
          print "| " host " " $0
        }' "$f"
      grep -q "^## $b" "$f" && header=1
    done
  done
} > "$out/merged.md"
echo "merged: $out/merged.md"
