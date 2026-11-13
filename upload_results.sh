#!/usr/bin/env bash
# Push one packaged run (made by collect_results.sh) to the project's wandb and to a PUBLIC HuggingFace repo.
# Run it on a node with internet.
#
#   DRY=1 bash upload_results.sh runs/<RUN>_return    # all checks, build the public copy in <RUN>_return/.public, upload nothing
#   bash upload_results.sh runs/<RUN>_return          # same checks, then wandb sync, then HF upload + verification
#
# Credentials (sent to you separately; never put them inside the kit):
#   HF token  : file ~/.corl_hf_token (or $HF_TOKEN_FILE); must be able to write to logan7000
#   wandb key : file ~/.corl_wandb_key (or env WANDB_API_KEY, or an api.wandb.ai entry in ~/.netrc); must belong to entity
#               logan-yang2002-johns-hopkins-university (checked before anything is written)
# What it writes:
#   wandb : `wandb sync` of every offline run in <RUN>_return/wandb into logan-yang2002-johns-hopkins-university/Co-learning
#   HF    : one PUBLIC model repo per model per run, e.g. logan7000/llm-math345-ttrl-llama32-3b-seed43-full
#           (an *_8gpu run gets "-8gpu" in the name), containing
#             best/ endpoint/  model files only: weights, config, tokenizer, trainer_state.json (never training_args.bin).
#                              They are checked, never modified: if a weight/config/tokenizer file names a path, host or
#                              user, nothing is uploaded; a trainer_state.json that does is left out (its scrubbed copy
#                              is in logs/).
#             logs/            train.log, selection.txt, compare_config.txt, first_steps.txt, trainer_state.json,
#                              training arguments of best/endpoint as JSON, env_check.txt, launcher.sbatch,
#                              launcher_stdout.log, the wandb text logs (not the binary .wandb files).
#                              All pass through scrub_logs.py (paths, host names, user names, e-mails, IPs replaced);
#                              a log that still names one of them afterwards is left out and listed in the output;
#                              if that happens to train.log or trainer_state.json, nothing is uploaded.
#             README.md        run name, contents, which steps are best/endpoint.
#           README.md (which records the run name) is uploaded first, then everything else; then every local file is
#           checked on the Hub by size.
# Safety: SMOKE runs are refused before anything happens. An existing repo is never overwritten; UPLOAD_RESUME=1
#   continues an interrupted upload, and only if the repo's README.md names this same run.
# Optional env: UPLOAD_NAMESPACE (default logan7000), UPLOAD_PY (python to use for the HF/wandb steps).
# Python: the public copy is built with python3 from PATH (stdlib only, >= 3.6). HF/wandb steps: UPLOAD_PY if given; on aarch64
#   the kit's env inside the kit's container; otherwise python3 from PATH, which then needs huggingface_hub>=0.25 and wandb.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RET="$(cd "${1:?usage: upload_results.sh runs/<RUN>_return}" && pwd)"
RUNNAME="$(basename "$RET")"
DRY="${DRY:-0}"; NS="${UPLOAD_NAMESPACE:-logan7000}"; RESUME="${UPLOAD_RESUME:-0}"
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
command -v python3 >/dev/null || { echo "FATAL: python3 not found"; exit 1; }
[ -f "$KIT/scrub_logs.py" ] || { echo "FATAL: scrub_logs.py missing from the kit"; exit 1; }

# 1) the public copy (writes only inside <RUN>_return/.public): model files checked, logs scrubbed
PUB="$RET/.public"; rm -rf "$PUB"; mkdir -p "$PUB"
python3 -I - "$RET" "$PUB" "$EXP" "$G8" "$SEED" "${RUNNAME%_return}" "$KIT" "$NS" "$HF_TOKEN_FILE" "$WANDB_ENTITY" <<'PY'
import getpass, glob, json, os, re, shutil, socket, subprocess, sys
ret, pub, exp, g8, seed, run, kit, ns, tokfile, entity = sys.argv[1:11]
g8s = "-8gpu" if g8 else ""
NAMES = {("ttrl_llama3b", "model"): f"llm-math345-ttrl-llama32-3b{g8s}-seed{seed}-full",
         ("ttrl_qwen3b", "model"): f"llm-math345-ttrl-qwen25-3b{g8s}-seed{seed}-full",
         ("corlplus_qrephr3b", "group_A"): f"cogrpo-decoupled-qwen25-3b-rephr-x-llama32-3b-math345{g8s}-seed{seed}-groupA-qwen-full",
         ("corlplus_qrephr3b", "group_B"): f"cogrpo-decoupled-qwen25-3b-rephr-x-llama32-3b-math345{g8s}-seed{seed}-groupB-llama-full"}
TITLE = {"ttrl_llama3b": "TTRL, Llama-3.2-3B-Instruct", "ttrl_qwen3b": "TTRL, Qwen2.5-3B",
         "corlplus_qrephr3b": "co-GRPO with decoupled data (Qwen2.5-3B on rephrased problems x Llama-3.2-3B-Instruct)"}
MODEL_FILE = re.compile(r"^(config\.json|generation_config\.json|model(-\d+-of-\d+)?\.safetensors|model\.safetensors\.index\.json|"
                        r"tokenizer\.json|tokenizer_config\.json|special_tokens_map\.json|added_tokens\.json|vocab\.json|merges\.txt|"
                        r"chat_template\.jinja|trainer_state\.json)$")
VOCAB = {"tokenizer.json", "vocab.json", "merges.txt"}   # base-model vocabulary: only checked for paths
PATHISH = re.compile(rb"(?<![\w.~/-])/(?:home|scratch|users?|mnt|raid|lustre|gpfs|work|data|projects?|tmp|weka)/", re.I)
WORDS = re.compile(rb"rebuttal|repro(?!duc|cess)|corl|\bco[-_ ]rl\b|anvil", re.I)
run_shown = run.replace("corlplus", "cogrpoplus")

# names that must not appear in model files: this machine, this account, and the hosts the run's own logs name
home, user = os.path.expanduser("~"), getpass.getuser()
strong = {home, kit, os.path.realpath(kit), socket.getfqdn(), socket.gethostname(), socket.gethostname().split(".")[0], user}
HOST = re.compile(r"\bhost=([A-Za-z][\w.-]+)|\"host\"\s*:\s*\"([^\"]+)\"")
for f in [os.path.join(ret, "launcher_stdout.log")] + glob.glob(os.path.join(ret, "wandb", "*", "files", "wandb-metadata.json")):
    if os.path.isfile(f):
        for m in HOST.finditer(open(f, errors="replace").read()):
            h = m.group(1) or m.group(2)
            strong.update((h, h.split(".")[0]))
strong = {s for s in strong if s and len(s) >= 5}

def model_file_problem(p, name):
    if name.endswith(".safetensors"):
        with open(p, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            data = fh.read(min(n, 100_000_000))
    else:
        data = open(p, "rb").read()
    if name in VOCAB:
        return bool(PATHISH.search(data) or any(s.encode() in data for s in (home, kit)))
    low = data.lower()
    return bool(PATHISH.search(data) or WORDS.search(data) or any(s.lower().encode() in low for s in strong))

common = [(os.path.join(ret, f), "logs/" + f) for f in ("env_check.txt", "launcher.sbatch", "launcher_stdout.log")
          if os.path.isfile(os.path.join(ret, f))]
wroot = os.path.join(ret, "wandb")
for root, dirs, files in os.walk(wroot):
    dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d)) and d != "tmp"]
    for f in files:
        p = os.path.join(root, f)
        if not os.path.islink(p) and f.endswith((".log", ".txt", ".json", ".yaml")):
            common.append((p, "logs/wandb/" + os.path.relpath(p, wroot)))

plan, failed = [], []
units = [u for u in ("group_A", "group_B", "model") if os.path.isdir(os.path.join(ret, u))]
if not units:
    sys.exit("FATAL: no group_A/group_B/model folder in the package")
for u in units:
    d, out = os.path.join(ret, u), os.path.join(pub, u)
    left_out = []
    for top in ("best", "endpoint"):
        src = os.path.join(d, top)
        if not os.path.isdir(src):
            continue
        for name in sorted(os.listdir(src)):
            p = os.path.join(src, name)
            if not os.path.isfile(p):
                continue
            if not MODEL_FILE.match(name):
                left_out.append(f"{top}/{name}")
                continue
            if model_file_problem(p, name):
                if name == "trainer_state.json":      # a scrubbed copy is in logs/
                    left_out.append(f"{top}/{name} (names a path/host/user)")
                    continue
                failed.append(f"{u}/{top}/{name}")
                continue
            dst = os.path.join(out, top, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
    pairs = [(os.path.join(d, f), "logs/" + f) for f in ("train.log", "selection.txt", "compare_config.txt", "first_steps.txt",
                                                        "trainer_state.json") if os.path.isfile(os.path.join(d, f))]
    pairs += [(os.path.join(d, top, "training_args.bin"), f"logs/training_args_{top}.json") for top in ("best", "endpoint")
              if os.path.isfile(os.path.join(d, top, "training_args.bin"))]
    pairs += common
    r = subprocess.run([sys.executable, "-I", os.path.join(kit, "scrub_logs.py"), "--dest", out, "--extra", entity,
                        "--secret-file", tokfile, "--secret-file", "~/.corl_wandb_key"] + [f"{s}={t}" for s, t in pairs],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)   # python >= 3.5
    print(f"[public] {u}: " + r.stdout.strip().replace("\n", f"\n[public] {u}: "))
    if r.returncode not in (0, 3):
        sys.exit(f"FATAL: scrub_logs.py failed for {u}: {r.stderr[-1500:]}")
    for need in ("logs/train.log", "logs/trainer_state.json"):
        if not os.path.isfile(os.path.join(out, need)):
            failed.append(f"{u}: {need} was withheld by the scrubber (see above)")
    for x in left_out:
        if not x.endswith("training_args.bin"):       # training_args.bin is expected to stay out (its JSON is in logs/)
            print(f"[public] {u}: not public: {x}")
    sel = open(os.path.join(out, "logs", "selection.txt")).read() if os.path.isfile(os.path.join(out, "logs", "selection.txt")) else ""
    readme = (f"# {TITLE[exp]}{' (8 GPUs)' if g8 else ''}, seed {seed}, {u}\n\n"
              f"Run `{run_shown}`.\n\n"
              "- `best/`: checkpoint with the highest in-training MATH-500 eval_reward (temperature 0.6)\n"
              "- `endpoint/`: last saved checkpoint\n"
              "- `logs/`: training log, launcher and environment logs, training arguments (JSON), wandb text logs; "
              "machine paths, host names and user names are replaced by placeholders such as `<path>`\n\n"
              f"## Selection\n```\n{sel}```\n")
    if WORDS.search(readme.encode()) or any(s.lower() in readme.lower() for s in strong):
        sys.exit(f"FATAL: the README for {u} would name a path, host or user")
    open(os.path.join(out, "README.md"), "w").write(readme)
    local = {}
    for root, _, files in os.walk(out):
        for f in files:
            p = os.path.join(root, f)
            local[os.path.relpath(p, out)] = os.path.getsize(p)
    if not any(k.startswith("endpoint/") and k.endswith(".safetensors") for k in local):
        failed.append(f"{u}: no endpoint weights")
    plan.append({"unit": u, "repo": f"{ns}/{NAMES[(exp, u)]}", "dir": out, "local": local, "mark": f"Run `{run_shown}`"})
if failed:
    sys.exit("FATAL: nothing uploaded, report this: " + "; ".join(failed))
json.dump(plan, open(os.path.join(pub, "plan.json"), "w"))
PY

# python runner for the HF / wandb steps
SIF="$KIT/images/cuda-$CUDAV-cudnn-devel-arm64.sif"; KPY="$KIT/envs/$ENVN/venv/bin/python"
if [ -n "${UPLOAD_PY:-}" ]; then run_py() { "$UPLOAD_PY" "$@"; }
elif [ "$(uname -m)" = aarch64 ] && [ -x "$KPY" ] && [ -f "$SIF" ]; then
  [ -n "${WANDB_API_KEY:-}" ] && export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
  run_py() { apptainer exec --cleanenv -B "$KIT" -B "$RET" -B "$(dirname "$HF_TOKEN_FILE")" \
               --env PYTHONNOUSERSITE=1 --env HF_TOKEN_FILE="$HF_TOKEN_FILE" "$SIF" "$KPY" "$@"; }
else run_py() { python3 "$@"; }; fi
run_py -c "import huggingface_hub, wandb" || { echo "FATAL: the python used here lacks huggingface_hub or wandb"; exit 1; }

# 2) credentials and the plan (writes nothing)
export U_RET="$RET" U_RUN="${RUNNAME%_return}" U_RESUME="$RESUME" U_ENTITY="$WANDB_ENTITY" U_PLAN="$PUB/plan.json"
for v in U_RET U_RUN U_RESUME U_ENTITY U_PLAN; do export "APPTAINERENV_$v=${!v}"; done
run_py - <<'PY'
import base64, json, netrc, os, sys, urllib.request
from huggingface_hub import HfApi, hf_hub_download
E = os.environ
# wandb key must exist and belong to the project's entity
key = E.get("WANDB_API_KEY")
if not key:
    try: key = (netrc.netrc().authenticators("api.wandb.ai") or (None, None, None))[2]
    except Exception: key = None
if not key: sys.exit("FATAL: no wandb key (put it in ~/.corl_wandb_key)")
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
for p in json.load(open(E["U_PLAN"])):
    rid, exists = p["repo"], api.repo_exists(p["repo"])
    if exists:
        if E["U_RESUME"] != "1": sys.exit(f"FATAL: {rid} already exists; not overwriting (UPLOAD_RESUME=1 only continues an interrupted upload of this same run)")
        try: rd = open(hf_hub_download(rid, "README.md", token=api.token)).read()
        except Exception: rd = ""
        if p["mark"] not in rd: sys.exit(f"FATAL: {rid} exists but its README.md does not name run {E['U_RUN']}; refusing to touch it")
        if api.model_info(rid).private: sys.exit(f"FATAL: {rid} exists but is private; refusing to touch it")
    n = len(p["local"]); gb = sum(p["local"].values()) / 1e9
    nlog = sum(1 for k in p["local"] if k.startswith("logs/"))
    print(f"[hf] {p['unit']} -> {rid} (PUBLIC)  exists={exists}  {n} files ({nlog} in logs/), {gb:.1f} GB")
PY
for d in "$RET"/wandb/offline-run-*; do [ -d "$d" ] && echo "[wandb] $( [ "$DRY" = 1 ] && echo 'would sync' || echo 'will sync') $(basename "$d") -> $WANDB_ENTITY/$WANDB_PROJECT"; done
if [ "$DRY" = 1 ]; then echo "[dry run] all checks passed, nothing written (public copy to inspect: $PUB)"; exit 0; fi

# 3) wandb
for d in "$RET"/wandb/offline-run-*; do
  [ -d "$d" ] || continue
  run_py -m wandb sync --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" "$d"
done

# 4) HuggingFace: README first (records the run), then everything else, then verify
run_py - <<'PY'
import datetime, json, os, sys
from huggingface_hub import HfApi
E = os.environ
api = HfApi(token=open(E["HF_TOKEN_FILE"]).read().strip())
failed = []
plan = json.load(open(E["U_PLAN"]))
for p in plan:
    rid, d, local = p["repo"], p["dir"], p["local"]
    api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
    if api.model_info(rid).private: sys.exit(f"FATAL: {rid} is private; expected public")
    api.upload_file(path_or_fileobj=os.path.join(d, "README.md"), path_in_repo="README.md", repo_id=rid, repo_type="model",
                    commit_message="README (run identity)")
    api.upload_large_folder(repo_id=rid, repo_type="model", folder_path=d, allow_patterns=sorted(local))
    remote = {f.path: f.size for f in api.list_repo_tree(rid, repo_type="model", recursive=True) if getattr(f, "size", None) is not None}
    missing = sorted(k for k in local if k not in remote)
    wrong = sorted(k for k in local if k in remote and remote[k] != local[k])
    lacking = [k for k in ("endpoint/",) if not any(r.startswith(k) and r.endswith(".safetensors") for r in remote)]
    lacking += [k for k in ("logs/train.log", "logs/trainer_state.json", "README.md") if k not in remote]
    ok = not (missing or wrong or lacking)
    print(f"[hf] verify {rid}: {len(local) - len(missing) - len(wrong)}/{len(local)} files present with matching size"
          + ("" if ok else f"  MISSING {missing[:5]} SIZE-MISMATCH {wrong[:5]} LACKING {lacking}"))
    if not ok: failed.append(rid)
if failed: sys.exit(f"FATAL: verification failed for {failed}; rerun with UPLOAD_RESUME=1")
open(os.path.join(E["U_RET"], "UPLOADED.txt"), "w").write(
    "".join(f"{p['repo']}  public  verified {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC\n" for p in plan))
print("[hf] all repos uploaded and verified (UPLOADED.txt written)")
PY
