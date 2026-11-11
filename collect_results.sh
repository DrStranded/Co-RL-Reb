#!/usr/bin/env bash
# After a run ends (finished, TIMEOUT or killed), put exactly what has to be returned into ONE folder:
#   <unit>/best/              checkpoint with the highest in-training MATH-500 eval (eval_reward), ties -> earliest step
#   <unit>/endpoint/          last saved checkpoint (checkpoint-136 when the run finished)
#   <unit>/train.log          full training log
#   <unit>/trainer_state.json full log_history of the endpoint checkpoint (every train step + every eval)
#   <unit>/selection.txt      eval curve and which steps were picked
#   <unit>/compare_config.txt the run's training arguments compared field by field with the paper run's (report only)
#   <unit>/first_steps.txt    metrics of training steps 1 and 2 (to compare with the paper run's)
#   wandb/                    offline wandb runs (synced by upload_results.sh)
#   env_check.txt, launcher.sbatch, launcher_stdout.log, SHA256SUMS
# <unit> = group_A / group_B for Co-RL+ (two models), model for TTRL (one model).
# Checkpoints are hard-linked (no extra disk) when possible, otherwise copied. Only needs python3 (stdlib).
# The package is built in <dest>.partial and renamed at the very end, so a failed run leaves nothing behind.
#
#   bash collect_results.sh runs/<RUN>          -> runs/<RUN>_return/   (then: upload_results.sh runs/<RUN>_return)
#   DRY=1 bash collect_results.sh runs/<RUN>    -> only print what would be picked
set -euo pipefail
RUN_DIR="$(cd "${1:?usage: collect_results.sh runs/<RUN> [dest]}" && pwd)"
DEST="${2:-${RUN_DIR}_return}"; TMP="${DEST}.partial"
command -v python3 >/dev/null || { echo "FATAL: python3 not found"; exit 1; }
if [ -d "$RUN_DIR/group_A" ]; then UNITS="group_A group_B"; else UNITS="."; fi
if [ "${DRY:-0}" != 1 ]; then
  [ -e "$DEST" ] && { echo "FATAL: $DEST already exists"; exit 1; }
  rm -rf "$TMP"; mkdir -p "$TMP"
  trap 'rm -rf "$TMP"; echo "FATAL: packaging failed, nothing kept"' ERR
fi
link_or_copy() { cp -al "$1" "$2" 2>/dev/null || { rm -rf "$2"; cp -a "$1" "$2"; }; }
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REFDIR="${REFDIR:-$TOOLS}"
[ -f "$REFDIR/ref_ttrl_llama3b_is9g1zwx.json" ] || REFDIR="$TOOLS/../../refs"   # layout of the private handoff copy
case "$(basename "$RUN_DIR")" in
  *ttrl_llama3b*) REF_A="json:$REFDIR/ref_ttrl_llama3b_is9g1zwx.json" ;;
  *ttrl_qwen3b*) REF_A="json:$REFDIR/ref_ttrl_qwen3b_fdlcna0t.json" ;;
  *corlplus_qrephr3b*) REF_A="json:$REFDIR/ref_corlplus_qrephr3b_swap_groupA.json" ;;
  *) REF_A="" ;;
esac

for u in $UNITS; do
  src="$RUN_DIR/$u"; name="$u"; [ "$u" = . ] && name=model
  [ -d "$src" ] || { echo "FATAL: missing $src"; false; }
  sel=$(python3 - "$src" <<'PY'
import json, os, re, sys
src = sys.argv[1]
cks = sorted(int(m.group(1)) for m in (re.match(r"^checkpoint-(\d+)$", d) for d in os.listdir(src))
             if m and os.path.isfile(os.path.join(src, m.group(0), "trainer_state.json")))
if not cks:
    sys.exit("no complete checkpoint (with trainer_state.json) under " + src)
end = cks[-1]
st = json.load(open(os.path.join(src, "checkpoint-%d" % end, "trainer_state.json")))
ev = [(h["step"], h["eval_reward"]) for h in st["log_history"] if "eval_reward" in h]
cand = [(s, r) for s, r in ev if s in cks]
best = max(cand, key=lambda x: (x[1], -x[0]))[0] if cand else -1
print("%d %d %d" % (best, end, st.get("max_steps", -1)))
for s, r in ev:
    print("  step %4d  eval_reward %.4f%s%s" % (s, r, "  saved" if s in cks else "", "  <- best" if s == best else ""))
PY
)
  read -r BEST END MAXS <<< "$(head -1 <<< "$sel")"
  status="finished"; [ "$END" -lt "$MAXS" ] && status="NOT finished (stopped at $END of $MAXS): endpoint = last saved checkpoint"
  report="[$name] best = checkpoint-$BEST, endpoint = checkpoint-$END, $status"
  [ "$BEST" = -1 ] && report="[$name] NO eval at any saved checkpoint -> no best; endpoint = checkpoint-$END, $status"
  echo "$report"; tail -n +2 <<< "$sel"
  [ "${DRY:-0}" = 1 ] && continue
  [ -f "$src/train.log" ] || { echo "FATAL: missing $src/train.log"; false; }
  d="$TMP/$name"; mkdir -p "$d"
  { echo "$report"; echo "run: $RUN_DIR"; echo "in-training MATH-500 eval (temperature 0.6):"; tail -n +2 <<< "$sel"; } > "$d/selection.txt"
  [ "$BEST" != -1 ] && link_or_copy "$src/checkpoint-$BEST" "$d/best"
  link_or_copy "$src/checkpoint-$END" "$d/endpoint"
  cp -a "$src/train.log" "$d/train.log"
  cp -a "$src/checkpoint-$END/trainer_state.json" "$d/trainer_state.json"
  # self-check, report only: training arguments vs the paper run (group_B vs this run's group_A), first two steps
  ref="$REF_A"; [ "$u" = group_B ] && ref="targs:$RUN_DIR/group_A/checkpoint-$END/training_args.bin"
  if [ -n "$ref" ] && [ -f "$TOOLS/compare_config.py" ] && [ -f "${ref#*:}" ]; then
    python3 "$TOOLS/compare_config.py" "$src/checkpoint-$END" "$ref" > "$d/compare_config.txt" 2>&1 || echo "(compare_config failed)" >> "$d/compare_config.txt"
    echo "  [$name] compare with $(basename "${ref#*:}"): $(grep -oE '== \([123]\) [0-9]+' "$d/compare_config.txt" | tr '\n' ' ')"
  else echo "(no reference found for this run)" > "$d/compare_config.txt"; fi
  tr '\r' '\n' < "$src/train.log" | grep -oE "\{'loss'[^}]*\}" | head -2 > "$d/first_steps.txt" || true
done
[ "${DRY:-0}" = 1 ] && exit 0

[ -d "$RUN_DIR/wandb" ] && link_or_copy "$RUN_DIR/wandb" "$TMP/wandb"
for f in env_check.txt launcher.sbatch launcher_stdout.log; do [ -f "$RUN_DIR/$f" ] && cp -a "$RUN_DIR/$f" "$TMP/$f"; done
(cd "$TMP" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
trap - ERR
mv "$TMP" "$DEST"
echo "[collect] done: $DEST  ($(du -sh "$DEST" | cut -f1), $(wc -l < "$DEST/SHA256SUMS") files)  -> next: upload_results.sh $DEST"
