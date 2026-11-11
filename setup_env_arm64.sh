#!/usr/bin/env bash
# Build one of the two paper training environments on an ARM64 (aarch64) GPU node, fully OFFLINE,
# from this kit. Run it ON a GPU node (it ends with a real flash-attn kernel test on the GPU).
#
#   bash setup_env_arm64.sh collab   # Co-RL+ / TTRL-Llama / 3B heter : py3.11.14, torch 2.9.1+cu129, vllm 0.14.0
#   bash setup_env_arm64.sh anvil    # TTRL-Qwen (Anvil paper run)    : py3.12.13, torch 2.9.0+cu128, vllm 0.11.2
#
# Inputs (env vars, all optional):
#   KIT      kit root (default: directory of this script)
#   ENVROOT  where envs are created (default: $KIT/envs)
#   SIF      CUDA container to build/run inside. Default: $KIT/images/cuda-<ver>-cudnn-devel-arm64.sif;
#            if missing it is pulled BY DIGEST (the arm64 image of nvidia/cuda:<ver>-cudnn-devel-ubuntu22.04 that this
#            kit was checked against): collab -> 12.9.1 sha256:edb9bb6c..., anvil -> 12.8.1 sha256:799b227b...
# Under Slurm (a GPU is needed for the final self-test), run it from inside the kit directory, e.g.
#   srun -N1 --gres=gpu:1 -c 16 --mem=64G -t 02:00:00 bash setup_env_arm64.sh collab 2>&1 | tee setup_collab.log
#
# Install order (offline, everything --no-deps: the pin list IS the full closure of the original env):
#   0) build tools at the original env's exact versions (the bundled Python ships different pip/setuptools)
#   1) torch triple, then every pinned binary wheel
#   2) the source-only packages (collab 12, anvil 7), built AFTER all wheels are in, because DeepSpeed's
#      setup.py imports torch (DS_BUILD_OPS=0: no CUDA ops are compiled)
#   3) flash-attn (prebuilt aarch64 wheel, same 2.8.3 source) and trl (editable, from the original commit tree)
#   4) verification: every pin at its exact version, key versions, GPU flash-attn kernel, latex2sympy2, vllm import
# Compiler: the bundled Python's sysconfig says CC=clang, but the CUDA devel image has gcc (build-essential is a
# dependency of cuda-nvcc) and no clang, so C extensions (netifaces, pycosat, crcmod) are built with CC=gcc.
set -euo pipefail
ENVNAME="${1:?usage: setup_env_arm64.sh collab|anvil}"
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$KIT/setup_env_arm64.sh" ] || KIT="${SLURM_SUBMIT_DIR:-$KIT}"   # under sbatch BASH_SOURCE is a spooled copy
[ -f "$KIT/setup_env_arm64.sh" ] || { echo "FATAL: cannot find the kit (got KIT=$KIT); pass KIT=/path/to/arm64_kit"; exit 1; }
KIT="$(realpath -m "$KIT")"
ENVROOT="${ENVROOT:-$KIT/envs}"
case "$KIT$ENVROOT" in *[[:space:]:]*) echo "FATAL: the kit path must not contain spaces or ':' ($KIT)"; exit 1 ;; esac
[ "$(uname -m)" = aarch64 ] || { echo "FATAL: this node is $(uname -m), not aarch64"; exit 1; }
echo "[setup] host $(hostname) | GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) | page size $(getconf PAGESIZE)"
case "$ENVNAME" in
  collab) PYTAR=$(ls "$KIT"/extra/cpython-3.11.14+*-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz); CUDAV=12.9.1; DIGEST=sha256:edb9bb6c7f45c624cb86bd1103e47471a609679a2b96ffb1bad59929f697a41e
          FAWHL="$KIT/extra/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_aarch64.whl"; TRLSRC="$KIT/code/05a21d7f"
          TOOLS="pip==26.3 setuptools==84.2.1 wheel==0.48.0 packaging==24.2 ninja==1.13.0"      # = x86 env_collab
          TORCH3="torch==2.9.1+cu129 torchvision==0.24.1 torchaudio==2.9.1"
          WANT='{"torch":"2.9.1+cu129","vllm":"0.14.0","transformers":"4.57.1","deepspeed":"0.18.6","accelerate":"1.12.0","datasets":"4.6.1","numpy":"1.26.3","triton":"3.5.1","peft":"0.18.1","sympy":"1.14.0","trl":"1.2.0.dev0","setuptools":"84.2.1"}' ;;
  anvil)  PYTAR=$(ls "$KIT"/extra/cpython-3.12.13+*-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz); CUDAV=12.8.1; DIGEST=sha256:799b227be38cc134465aa75feb94ad6cd7c6f9ebf8a014de4485818d11bbbfa8
          FAWHL="$KIT/extra/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_aarch64.whl"; TRLSRC="$KIT/code/4dc11923"
          TOOLS="pip==26.0.1 setuptools==80.10.2 wheel==0.46.3 packaging==26.0 ninja==1.13.0"   # = Anvil run requirements
          TORCH3="torch==2.9.0+cu128 torchvision==0.24.0 torchaudio==2.9.0"
          WANT='{"torch":"2.9.0+cu128","vllm":"0.11.2","transformers":"4.57.0","deepspeed":"0.18.0","accelerate":"1.13.0","datasets":"4.8.4","numpy":"2.2.6","triton":"3.5.0","peft":"0.18.1","trl":"1.2.0.dev0","setuptools":"80.10.2"}' ;;
  *) echo "unknown env $ENVNAME"; exit 2 ;;
esac
SIF="${SIF:-$KIT/images/cuda-$CUDAV-cudnn-devel-arm64.sif}"
mkdir -p "$(dirname "$SIF")" "$ENVROOT"
[ -f "$SIF" ] || apptainer pull "$SIF" "docker://nvidia/cuda@$DIGEST"   # = nvidia/cuda:$CUDAV-cudnn-devel-ubuntu22.04 (arm64), pinned
E="$ENVROOT/$ENVNAME"; rm -rf "$E"; mkdir -p "$E"
echo "[setup] env=$ENVNAME  python=$(basename "$PYTAR")  sif=$SIF  -> $E"

apptainer exec --nv --cleanenv -B "$KIT" -B "$ENVROOT" \
  --env PYTHONNOUSERSITE=1 --env KIT="$KIT" --env E="$E" --env PYTAR="$PYTAR" --env FAWHL="$FAWHL" \
  --env TRLSRC="$TRLSRC" --env ENVNAME="$ENVNAME" --env WANT="$WANT" --env TMPDIR="$E/tmp" \
  --env TOOLS="$TOOLS" --env TORCH3="$TORCH3" \
  --env CC="gcc -pthread" --env CXX="g++ -pthread" --env LDCXXSHARED="g++ -pthread -shared" \
  --env DS_BUILD_OPS=0 --env PIP_NO_INDEX=1 --env PIP_FIND_LINKS="$KIT/wheelhouse_$ENVNAME" \
  --env PIP_DISABLE_PIP_VERSION_CHECK=1 --env PIP_CONFIG_FILE=/dev/null \
  "$SIF" bash -s <<'CONTAINER'
set -euo pipefail
mkdir -p "$TMPDIR"
command -v gcc >/dev/null || { echo "FATAL: no gcc in this container (needed for netifaces/pycosat/crcmod, and by Triton at run time)"; exit 1; }
echo "[setup] $(gcc --version | head -1) | $(nvcc --version | tail -1)"
tar -xzf "$PYTAR" -C "$E"                         # -> $E/python
"$E/python/bin/python3" -m venv --copies "$E/venv"
source "$E/venv/bin/activate"
# 0) build tools, exact versions of the original env
python -m pip install --no-deps --only-binary=:all: $TOOLS
# 1) torch triple, then all pinned binary wheels
python -m pip install --no-deps --only-binary=:all: $TORCH3
python - <<'PY'
import os, re
K, env, T = os.environ["KIT"], os.environ["ENVNAME"], os.environ["TMPDIR"]
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
src = {norm(re.sub(r"-[0-9][^-]*\.(tar\.gz|zip)$", "", f)) for f in os.listdir(f"{K}/wheelhouse_{env}") if f.endswith((".tar.gz", ".zip"))}
pins = [l.strip() for l in open(f"{K}/pins_{env}.txt") if l.strip()]
w = [p for p in pins if norm(p.split("==")[0]) not in src]
s = [p for p in pins if norm(p.split("==")[0]) in src]
open(f"{T}/pins_wheels.txt", "w").write("\n".join(w) + "\n")
open(f"{T}/pins_source.txt", "w").write("\n".join(s) + "\n")
print(f"[setup] {len(pins)} pins = {len(w)} binary wheels + {len(s)} source-only: {' '.join(s)}", flush=True)
PY
python -m pip install --no-deps --only-binary=:all: -r "$TMPDIR/pins_wheels.txt"
# 2) source-only packages, built against the now-complete env
python -m pip install --no-deps --no-build-isolation -r "$TMPDIR/pins_source.txt"
# 3) flash-attn and trl from the original commit tree
python -m pip install --no-deps "$FAWHL"
python -m pip install --no-deps --no-build-isolation -e "$TRLSRC"
# 4) verification
python - <<'PY'
import importlib, importlib.metadata as md, json, os, re, sys, sysconfig
from packaging.version import Version
bad = []
# every pin (+ torch triple + build tools) installed at exactly the pinned version, read from this venv's site-packages only
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
inst = {norm(d.metadata["Name"]): d.version for d in md.distributions(path=[sysconfig.get_paths()["purelib"]])}
pins = [l.strip() for l in open(f'{os.environ["KIT"]}/pins_{os.environ["ENVNAME"]}.txt') if l.strip()]
pins += os.environ["TORCH3"].split() + os.environ["TOOLS"].split()
def same(a, b):
    try: return Version(a) == Version(b)
    except Exception: return a == b
off = []
for p in pins:
    n, v = p.split("==", 1); got = inst.get(norm(n))
    if got is None or not same(got, v): off.append(f"{p} (installed: {got})")
print(f"  pins at exact version: {len(pins) - len(off)}/{len(pins)}" + (f"  MISMATCH {off}" if off else ""))
if off: bad.append("pins")
want = json.loads(os.environ["WANT"])
for m, v in want.items():
    try:
        mod = importlib.import_module(m); got = mod.__version__
    except Exception as e:
        print(f"  {m:13s} IMPORT FAILED {e}"); bad.append(m); continue
    ok = got == v
    print(f"  {m:13s} {got:16s} {'ok' if ok else 'WANT ' + v}"); bad += [] if ok else [m]
import trl, torch, platform
print("  python", platform.python_version(), platform.machine(), " trl from", trl.__file__)
if "/code/" not in trl.__file__: bad.append("trl-location")
print("  cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available(): bad.append("cuda"); print("FAILED", bad); sys.exit(1)
print("  GPU", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
cv = torch.backends.cudnn.version(); nv = torch.cuda.nccl.version(); nv = ".".join(map(str, nv)) if isinstance(nv, tuple) else str(nv)
print(f"  cudnn loaded {cv} (want 91002 = pinned 9.10.2)   nccl loaded {nv} (want 2.27.5 = pinned)")
if cv != 91002: bad.append("cudnn")
if nv != "2.27.5": bad.append("nccl")
if os.environ["ENVNAME"] == "collab":
    import crcmod._crcfunext, netifaces, pycosat; print("  C extensions built: crcmod._crcfunext, netifaces, pycosat")
from flash_attn import flash_attn_func
q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
o = flash_attn_func(q, k, v, causal=True)
ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
err = (o.float() - ref.float()).abs().max().item(); print(f"  flash-attn vs sdpa max|diff| = {err:.3g}")
if err > 2e-2: bad.append("fa-kernel")
from latex2sympy2 import latex2sympy; print("  latex2sympy2:", latex2sympy(r"\frac{1}{2}"))
import vllm; print("  vllm import ok")
print("FAILED" if bad else "ALL CHECKS PASS", bad); sys.exit(1 if bad else 0)
PY
python -m pip check > "$E/pip_check.txt" 2>&1 || true
echo "[setup] pip check (informational): $(head -3 "$E/pip_check.txt" | tr '\n' ' ')  -> $E/pip_check.txt"
python -m pip freeze > "$E/freeze.txt"
CONTAINER
date --iso-8601=seconds > "$E/ENV_READY"
echo "[setup] $ENVNAME ready: $E   (freeze in $E/freeze.txt)"
