#!/usr/bin/env bash
# Write the current state of this kit (machine, download, setup, jobs, runs, uploads) into STATUS.md and upload it to
# the project's private HuggingFace repo, so the project contact can follow progress without anyone forwarding logs.
# Contains no secrets (aborts if the HF token string would end up in the file).
#   bash report_status.sh           # write STATUS.md and upload it   (needs the HF token file and internet)
#   DRY=1 bash report_status.sh     # only write and print STATUS.md
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$KIT"
OUT="$KIT/STATUS.md"; HF_TOKEN_FILE="${HF_TOKEN_FILE:-$( [ -s "$HOME/.corl_hf_token" ] && echo "$HOME/.corl_hf_token" || echo "$HOME/.cache/huggingface/token" )}"
NS="${UPLOAD_NAMESPACE:-logan7000}"; REPO="${STATUS_REPO:-$NS/repro-arm-status}"; SITE="${UPLOAD_SITE:-arm}"
TS=$(date -u +%Y%m%dT%H%M%SZ); HOST=$(hostname -s)
{
echo "# status $TS  (site $SITE, host $HOST, $(uname -m))"
echo; echo "## machine (where this was run)"
echo "- GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "- apptainer: $(apptainer --version 2>/dev/null || echo 'not found on this node')"
echo "- free disk on the kit filesystem: $(df -h "$KIT" 2>/dev/null | awk 'NR==2{print $4}')   page size: $(getconf PAGESIZE)"
echo; echo "## download"
for env in collab anvil; do
  f=kit_files_$env.tsv; [ -f "$f" ] || continue
  n=$(grep -c . "$f"); have=$(cut -f1 "$f" | while read -r p; do [ -f "$p" ] && echo x; done | wc -l)
  echo "- $env: $have / $n package files present"
done
for c in 05a21d7f 00e0eed5 4dc11923; do [ -f "code/$c/ORIGINAL_COMMIT" ] && echo "- code/$c: commit $(cut -c1-12 "code/$c/ORIGINAL_COMMIT")"; done
echo; echo "## setup"
for env in collab anvil; do
  [ -e "envs/$env" ] || [ -f "setup_$env.log" ] || continue
  echo "### $env"
  echo "- ENV_READY: $([ -f "envs/$env/ENV_READY" ] && cat "envs/$env/ENV_READY" || echo no)"
  if [ -f "setup_$env.log" ]; then
    echo "- 'ALL CHECKS PASS' in setup_$env.log: $(grep -c 'ALL CHECKS PASS' "setup_$env.log")"
    echo '```'; tail -n 30 "setup_$env.log" | cut -c1-300; echo '```'
  fi
  if [ "$env" = collab ] && [ -f envs/collab/pip_check.txt ]; then
    d=$(diff <(grep -v '^#' expected_pip_check_collab.txt | sort) <(sort envs/collab/pip_check.txt) | grep -c '^[<>]')
    echo "- pip check vs expected_pip_check_collab.txt: $([ "$d" = 0 ] && echo identical || echo "$d differing lines")"
  fi
done
echo; echo "## jobs"; echo '```'
squeue -u "$USER" -o "%i %j %T %M %l %R" 2>/dev/null || echo "squeue not available here"
echo '```'
echo; echo "## runs"
for d in runs/*/; do
  d=${d%/}; [ -d "$d" ] || continue; case "$d" in *_return|*.partial) continue ;; esac
  echo "### $(basename "$d")"
  for u in . group_A group_B; do
    [ -f "$d/$u/train.log" ] || continue
    log=$(tr '\r' '\n' < "$d/$u/train.log")
    prog=$(grep -oE '[0-9]+/(136|2) \[' <<< "$log" | tail -1 | tr -d ' [')
    evs=$(grep -oE "'eval_reward': [0-9.]+" <<< "$log" | awk '{printf "%.3f ", $2}')
    rw=$(grep -oE "'reward': [0-9.]+" <<< "$log" | tail -3 | awk '{printf "%.3f ", $2}')
    errs=$(grep -cE 'Traceback|RuntimeError|CUDA error|NCCL error|OutOfMemory|Killed|FATAL|: nan|nan,' <<< "$log")
    echo "- ${u/#./model}: step ${prog:-?} | eval_reward: ${evs:-none yet} | last train reward: ${rw:-none yet} | error-like lines: $errs"
  done
  [ -f "$d/launcher_stdout.log" ] && { echo '```'; tr '\r' '\n' < "$d/launcher_stdout.log" | grep -v '^\s*$' | tail -n 12 | cut -c1-300; echo '```'; }
  if [ -d "${d}_return" ]; then
    echo "- packaged: $(basename "${d}_return")"; [ -f "${d}_return/UPLOADED.txt" ] && sed 's/^/  - /' "${d}_return/UPLOADED.txt"
  fi
done
} > "$OUT" 2>&1
if [ -s "$HF_TOKEN_FILE" ] && grep -qF "$(cat "$HF_TOKEN_FILE")" "$OUT"; then echo "FATAL: STATUS.md would contain the HF token; not uploading"; exit 1; fi
if [ "${DRY:-0}" = 1 ]; then cat "$OUT"; echo; echo "[dry run] STATUS.md written, not uploaded"; exit 0; fi
[ -s "$HF_TOKEN_FILE" ] || { echo "FATAL: HF token file $HF_TOKEN_FILE missing; STATUS.md written but not uploaded"; exit 1; }

SIF="$KIT/images/cuda-12.9.1-cudnn-devel-arm64.sif"; KPY="$KIT/envs/collab/venv/bin/python"
export U_REPO="$REPO" U_OUT="$OUT" U_NAME="status/${SITE}_${HOST}_${TS}.md" U_LATEST="STATUS_latest_${SITE}.md" HF_TOKEN_FILE
if [ "$(uname -m)" = aarch64 ] && [ -x "$KPY" ] && [ -f "$SIF" ]; then
  for v in U_REPO U_OUT U_NAME U_LATEST; do export "APPTAINERENV_$v=${!v}"; done
  run_py() { apptainer exec --cleanenv -B "$KIT" -B "$(dirname "$HF_TOKEN_FILE")" --env HF_TOKEN_FILE="$HF_TOKEN_FILE" "$SIF" "$KPY" "$@"; }
else run_py() { python3 "$@"; }; fi
run_py - <<'PY'
import os
from huggingface_hub import HfApi
E = os.environ
api = HfApi(token=open(E["HF_TOKEN_FILE"]).read().strip())
api.create_repo(E["U_REPO"], repo_type="model", private=True, exist_ok=True)
for path in (E["U_NAME"], E["U_LATEST"]):
    api.upload_file(path_or_fileobj=E["U_OUT"], path_in_repo=path, repo_id=E["U_REPO"], repo_type="model", commit_message="status")
print(f"[status] uploaded to {E['U_REPO']}: {E['U_NAME']} (and {E['U_LATEST']})")
PY
