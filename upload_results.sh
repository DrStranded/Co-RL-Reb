#!/usr/bin/env bash
# Push one packaged run (made by collect_results.sh) to the project's wandb and HuggingFace. Run it on a node with internet.
#
#   DRY=1 bash upload_results.sh runs/<RUN>_return    # check everything, print the plan, write nothing
#   bash upload_results.sh runs/<RUN>_return          # same checks, then wandb sync, then HF upload + verification
#
# Credentials (sent to you separately; never put them inside the kit):
#   HF token  : file $HF_TOKEN_FILE (default ~/.cache/huggingface/token); must be able to write to logan7000
#   wandb key : env WANDB_API_KEY (or an api.wandb.ai entry in ~/.netrc); must belong to entity
#               logan-yang2002-johns-hopkins-university (checked before anything is written)
# What it writes:
#   wandb : `wandb sync` of every offline run in <RUN>_return/wandb into logan-yang2002-johns-hopkins-university/Co-learning
#   HF    : one PRIVATE model repo per model per run, e.g. logan7000/repro-corlplus-qrephr3b-seed43-groupA-qwen25-3b-arm-full
#           (an *_8gpu run gets "-8gpu" in the name). README.md (which records the run name) is uploaded first, then
#           best/ endpoint/ train.log trainer_state.json selection.txt; then every local file is checked on the Hub by size.
# Safety: SMOKE runs are refused before anything happens. An existing repo is never overwritten; UPLOAD_RESUME=1
#   continues an interrupted upload, and only if the repo's README.md names this same run.
# Optional env: UPLOAD_SITE (default arm), UPLOAD_NAMESPACE (default logan7000), UPLOAD_PY (python to use).
# Python: UPLOAD_PY if given; on aarch64 the kit's env run inside the kit's container (so the host glibc does not matter);
#   otherwise python3 from PATH, which needs huggingface_hub>=0.25 and wandb (pip install --user huggingface_hub wandb).
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RET="$(cd "${1:?usage: upload_results.sh runs/<RUN>_return}" && pwd)"
RUNNAME="$(basename "$RET")"
DRY="${DRY:-0}"; SITE="${UPLOAD_SITE:-arm}"; NS="${UPLOAD_NAMESPACE:-logan7000}"; RESUME="${UPLOAD_RESUME:-0}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-$( [ -s "$HOME/.corl_hf_token" ] && echo "$HOME/.corl_hf_token" || echo "$HOME/.cache/huggingface/token" )}"
[ -z "${WANDB_API_KEY:-}" ] && [ -s "$HOME/.corl_wandb_key" ] && export WANDB_API_KEY="$(cat "$HOME/.corl_wandb_key")"   # dedicated key file
WANDB_ENTITY=logan-yang2002-johns-hopkins-university; WANDB_PROJECT=Co-learning

# 0) what is this run? (before anything else)
[[ "$RUNNAME" =~ ^(SMOKE_)?(corlplus_qrephr3b|ttrl_qwen3b|ttrl_llama3b)(_8gpu)?_seed([0-9]+)_.*_return$ ]] \
  || { echo "FATAL: '$RUNNAME' is not a collect_results.sh package of a known experiment"; exit 1; }
SMOKE="${BASH_REMATCH[1]}"; EXP="${BASH_REMATCH[2]}"; G8="${BASH_REMATCH[3]}"; SEED="${BASH_REMATCH[4]}"
[ -z "$SMOKE" ] || { echo "FATAL: this is a SMOKE run; smoke runs are never uploaded (nothing was written)"; exit 1; }
case "$EXP" in ttrl_qwen3b) ENVN=anvil; CUDAV=12.8.1 ;; *) ENVN=collab; CUDAV=12.9.1 ;; esac
[ -f "$RET/SHA256SUMS" ] || { echo "FATAL: $RET has no SHA256SUMS"; exit 1; }
echo "[check] package integrity"; (cd "$RET" && sha256sum -c --quiet SHA256SUMS) || { echo "FATAL: package checksum mismatch"; exit 1; }
[ -s "$HF_TOKEN_FILE" ] || { echo "FATAL: HF token file $HF_TOKEN_FILE missing"; exit 1; }

# python runner
SIF="$KIT/images/cuda-$CUDAV-cudnn-devel-arm64.sif"; KPY="$KIT/envs/$ENVN/venv/bin/python"
if [ -n "${UPLOAD_PY:-}" ]; then run_py() { "$UPLOAD_PY" "$@"; }
elif [ "$(uname -m)" = aarch64 ] && [ -x "$KPY" ] && [ -f "$SIF" ]; then
  [ -n "${WANDB_API_KEY:-}" ] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
  run_py() { apptainer exec --cleanenv -B "$KIT" -B "$RET" -B "$(dirname "$HF_TOKEN_FILE")" \
               --env PYTHONNOUSERSITE=1 --env HF_TOKEN_FILE="$HF_TOKEN_FILE" "$SIF" "$KPY" "$@"; }
else run_py() { python3 "$@"; }; fi
run_py -c "import huggingface_hub, wandb" || { echo "FATAL: the python used here lacks huggingface_hub or wandb"; exit 1; }

# 1) all checks and the plan (writes nothing)
export U_RET="$RET" U_RUN="${RUNNAME%_return}" U_EXP="$EXP" U_G8="$G8" U_SEED="$SEED" U_SITE="$SITE" U_NS="$NS" U_RESUME="$RESUME" \
       U_ENTITY="$WANDB_ENTITY" U_PLAN="$RET/.upload_plan.json"
for v in U_RET U_RUN U_EXP U_G8 U_SEED U_SITE U_NS U_RESUME U_ENTITY U_PLAN; do export "APPTAINERENV_$v=${!v}"; done
run_py - <<'PY'
import base64, json, netrc, os, sys, urllib.request
from huggingface_hub import HfApi, hf_hub_download
E = os.environ
# wandb key must exist and belong to the project's entity
key = E.get("WANDB_API_KEY")
if not key:
    try: key = (netrc.netrc().authenticators("api.wandb.ai") or (None, None, None))[2]
    except Exception: key = None
if not key: sys.exit("FATAL: no wandb key (export WANDB_API_KEY=... or run wandb login)")
q = json.dumps({"query": "{ viewer { username entity teams { edges { node { name } } } } }"}).encode()
req = urllib.request.Request("https://api.wandb.ai/graphql", data=q, headers={"Content-Type": "application/json",
      "Authorization": "Basic " + base64.b64encode(b"api:" + key.encode()).decode()})
v = json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]["viewer"]
teams = [e["node"]["name"] for e in v["teams"]["edges"]] + [v["entity"]]
print(f"[wandb] key user={v['username']} teams={teams}")
if E["U_ENTITY"] not in teams: sys.exit(f"FATAL: this wandb key has no access to {E['U_ENTITY']}")
# HF token must be a write token
api = HfApi(token=open(E["HF_TOKEN_FILE"]).read().strip())
who = api.whoami(); role = who.get("auth", {}).get("accessToken", {}).get("role")
print(f"[hf] token user={who.get('name')} role={role}")
if role not in ("write", "fineGrained"): sys.exit("FATAL: the HF token cannot write")
seed, g8 = E["U_SEED"], ("-8gpu" if E["U_G8"] else "")
names = {("corlplus_qrephr3b", "group_A"): f"corlplus-qrephr3b{g8}-seed{seed}-groupA-qwen25-3b",
         ("corlplus_qrephr3b", "group_B"): f"corlplus-qrephr3b{g8}-seed{seed}-groupB-llama32-3b-instruct",
         ("ttrl_qwen3b", "model"): f"ttrl-qwen25-3b{g8}-seed{seed}",
         ("ttrl_llama3b", "model"): f"ttrl-llama32-3b-instruct{g8}-seed{seed}"}
plan = []
for unit in [u for u in ("group_A", "group_B", "model") if os.path.isdir(os.path.join(E["U_RET"], u))]:
    rid = f"{E['U_NS']}/repro-{names[(E['U_EXP'], unit)]}-{E['U_SITE']}-full"; d = os.path.join(E["U_RET"], unit)
    local = {}
    for top in ("best", "endpoint"):
        for root, _, files in os.walk(os.path.join(d, top)):
            for f in files:
                p = os.path.join(root, f); local[os.path.relpath(p, d)] = os.path.getsize(p)
    for f in ("train.log", "trainer_state.json", "selection.txt"):
        if os.path.isfile(os.path.join(d, f)): local[f] = os.path.getsize(os.path.join(d, f))
    exists = api.repo_exists(rid)
    if exists:
        if E["U_RESUME"] != "1": sys.exit(f"FATAL: {rid} already exists; not overwriting (UPLOAD_RESUME=1 only continues an interrupted upload of this same run)")
        try: rd = open(hf_hub_download(rid, "README.md", token=api.token)).read()
        except Exception: rd = ""
        if f"Run `{E['U_RUN']}`" not in rd: sys.exit(f"FATAL: {rid} exists but its README.md does not name run {E['U_RUN']}; refusing to touch it")
    print(f"[hf] {unit} -> {rid} (private)  exists={exists}  {len(local)} files, {sum(local.values()) / 1e9:.1f} GB")
    plan.append({"unit": unit, "repo": rid, "dir": d, "local": local})
if not plan: sys.exit("FATAL: no group_A/group_B/model folder in the package")
json.dump(plan, open(E["U_PLAN"], "w"))
PY
for d in "$RET"/wandb/offline-run-*; do [ -d "$d" ] && echo "[wandb] $( [ "$DRY" = 1 ] && echo 'would sync' || echo 'will sync') $(basename "$d") -> $WANDB_ENTITY/$WANDB_PROJECT"; done
if [ "$DRY" = 1 ]; then rm -f "$RET/.upload_plan.json"; echo "[dry run] all checks passed, nothing written"; exit 0; fi

# 2) wandb
for d in "$RET"/wandb/offline-run-*; do
  [ -d "$d" ] || continue
  run_py -m wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$d"
done

# 3) HuggingFace: README first (records the run), then the files, then verify
run_py - <<'PY'
import json, os, sys
from huggingface_hub import HfApi
E = os.environ
api = HfApi(token=open(E["HF_TOKEN_FILE"]).read().strip())
env_check = open(os.path.join(E["U_RET"], "env_check.txt")).read() if os.path.isfile(os.path.join(E["U_RET"], "env_check.txt")) else ""
failed = []
for p in json.load(open(E["U_PLAN"])):
    rid, d, local = p["repo"], p["dir"], p["local"]
    sel = open(os.path.join(d, "selection.txt")).read() if os.path.isfile(os.path.join(d, "selection.txt")) else ""
    api.create_repo(rid, repo_type="model", private=True, exist_ok=True)
    readme = (f"# Re-run: {E['U_EXP']}{'_8gpu' if E['U_G8'] else ''}, seed {E['U_SEED']}, {p['unit']}\n\n"
              f"Run `{E['U_RUN']}` ({E['U_SITE']}).\n\n"
              "- `best/`: checkpoint with the highest in-training MATH-500 eval_reward (temperature 0.6)\n"
              "- `endpoint/`: last saved checkpoint\n- `train.log`, `trainer_state.json`, `selection.txt`: full logs\n\n"
              f"## Selection\n```\n{sel}```\n\n## Environment\n```\n{env_check}```\n")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=rid, repo_type="model", commit_message="README (run identity)")
    api.upload_large_folder(repo_id=rid, repo_type="model", folder_path=d,
                            allow_patterns=["best/**", "endpoint/**", "train.log", "trainer_state.json", "selection.txt"])
    remote = {f.path: f.size for f in api.list_repo_tree(rid, repo_type="model", recursive=True) if getattr(f, "size", None) is not None}
    missing = sorted(k for k in local if k not in remote)
    wrong = sorted(k for k in local if k in remote and remote[k] != local[k])
    must = ["endpoint/", "trainer_state.json", "train.log"] + (["best/"] if any(k.startswith("best/") for k in local) else [])
    lacking = [k for k in must if not any((r.startswith(k) and r.endswith(".safetensors")) if k.endswith("/") else r == k for r in remote)]
    ok = not (missing or wrong or lacking)
    print(f"[hf] verify {rid}: {len(local) - len(missing) - len(wrong)}/{len(local)} files present with matching size"
          + ("" if ok else f"  MISSING {missing[:5]} SIZE-MISMATCH {wrong[:5]} LACKING {lacking}"))
    if not ok: failed.append(rid)
if failed: sys.exit(f"FATAL: verification failed for {failed}; rerun with UPLOAD_RESUME=1")
import datetime
open(os.path.join(E["U_RET"], "UPLOADED.txt"), "w").write("".join(f"{p['repo']}  verified {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC\n" for p in json.load(open(E["U_PLAN"]))))
print("[hf] all repos uploaded and verified (UPLOADED.txt written)")
PY
rm -f "$RET/.upload_plan.json"
