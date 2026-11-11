#!/usr/bin/env bash
# ONLY needed if your GPU compute nodes cannot reach huggingface.co. Run this once on an ARM64 node that HAS internet,
# after setup_env_arm64.sh, with the same HF cache the training will use (default $KIT/cache/hf, override with HF_HOME_DIR):
#   bash prefetch_hf.sh collab     # models + data for ttrl_llama3b and corlplus_qrephr3b
#   bash prefetch_hf.sh anvil      # models + data for ttrl_qwen3b
# It downloads the two models and loads the datasets once with the env's own `datasets` version (so the processed
# cache matches). Then submit training with  --export=ALL,...,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1  (run_exp_arm64.sh
# forwards these two into the container only when they are set). NOT TESTED on the target cluster.
set -euo pipefail
ENVNAME="${1:?usage: prefetch_hf.sh collab|anvil}"
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME_DIR="$(realpath -m "${HF_HOME_DIR:-$KIT/cache/hf}")"; HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
case "$ENVNAME" in collab) CUDAV=12.9.1 ;; anvil) CUDAV=12.8.1 ;; *) echo "unknown env $ENVNAME"; exit 2 ;; esac
ENV="$KIT/envs/$ENVNAME/venv"; SIF="$KIT/images/cuda-$CUDAV-cudnn-devel-arm64.sif"
[ -f "$KIT/envs/$ENVNAME/ENV_READY" ] && [ -f "$SIF" ] || { echo "FATAL: run setup_env_arm64.sh $ENVNAME first"; exit 1; }
[ -s "$HF_TOKEN_FILE" ] || { echo "FATAL: HF token file $HF_TOKEN_FILE missing"; exit 1; }
mkdir -p "$HF_HOME_DIR"
export APPTAINERENV_HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
apptainer exec --cleanenv -B "$KIT" -B "$HF_HOME_DIR" --env PYTHONNOUSERSITE=1 --env ENVNAME="$ENVNAME" \
  --env HF_HOME="$HF_HOME_DIR" --env HF_DATASETS_CACHE="$HF_HOME_DIR/datasets" "$SIF" "$ENV/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
import datasets
env = os.environ["ENVNAME"]
models = ["Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-3B-Instruct"]
for m in models:
    print("model  ", m, "->", snapshot_download(m))
print("dataset q1716523669/MATH-Level345:", datasets.load_dataset("q1716523669/MATH-Level345"))
if env == "collab":   # Co-RL+ reads the rephrased dataset's two configs (code/05a21d7f/projects/co-grpo-dp/dataset.py)
    for cfg in ("original", "rephrased"):
        print(f"dataset q1716523669/MATH-Level345-Rephrased-DeepSeek [{cfg}]:",
              datasets.load_dataset("q1716523669/MATH-Level345-Rephrased-DeepSeek", cfg, split="train"))
print("prefetch done; cache:", os.environ["HF_HOME"])
PY
