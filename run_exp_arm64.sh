#!/bin/bash
# ARM64 (aarch64) version of the x86 launcher run_exp.sbatch. The TRAINING COMMANDS BELOW ARE BYTE-IDENTICAL to that
# x86 launcher, which passed its smoke tests on the original x86 cluster; only machine-specific paths changed.
# Prerequisite: bash setup_env_arm64.sh collab   and/or   bash setup_env_arm64.sh anvil   (writes envs/<env>/ENV_READY)
# Needs ONE whole node with >= 4 GPUs. Submit with your scheduler, e.g. (Slurm; run sbatch from inside the kit directory;
# KIT must be passed because sbatch executes a spooled copy of this script; --no-requeue because a requeued job would
# start over from step 0 -- this script also refuses to run as a requeued job):
#   sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 38:00:00 --export=ALL,KIT=$PWD,EXP=corlplus_qrephr3b,SEED=43 run_exp_arm64.sh
#   EXP = ttrl_qwen3b | ttrl_llama3b | corlplus_qrephr3b ;  SEED = 43/44 (only the sampling seed; data_seed stays 42)
#   SMOKE=1 runs 2 steps.  Optional: KIT, OUT_ROOT, HF_HOME_DIR, HF_TOKEN_FILE; HF_HUB_OFFLINE / HF_DATASETS_OFFLINE /
#   TRANSFORMERS_OFFLINE are forwarded into the container only if set (for compute nodes without internet, see README).
# Re-runs of paper experiments, launched EXACTLY as the original runs:
#   code    = git archive of the original run's commit (code/<commit>)
#   command = the original command line recorded by wandb (copied verbatim below)
#   env     = the original run's package versions (rebuilt for aarch64 from the same pins)
# The ONLY intentional differences from the original command lines:
#   --seed only (new sampling seed; --data_seed stays 42 = paper data order), output/rendezvous paths, wandb offline,
#   no --save_total_limit (keep every checkpoint; does not affect training or best_model),
#   GPU-count conversion (same per-device batch, grad-accum scaled; asserted below), SMOKE=1 adds --max_steps 2.
set -uo pipefail
: "${EXP:?}" "${SEED:?}"
SMOKE="${SMOKE:-0}"
case "$SMOKE" in 0|1) ;; *) echo "FATAL: SMOKE must be 0 or 1 (got '$SMOKE')"; exit 1 ;; esac
[ "${SLURM_RESTART_COUNT:-0}" = 0 ] || { echo "FATAL: this Slurm job was requeued (restart ${SLURM_RESTART_COUNT}); a rerun would silently start over from step 0. Not running: keep the old run directory and tell the person who sent you this kit."; exit 1; }
KIT="${KIT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[ -f "$KIT/setup_env_arm64.sh" ] || KIT="${SLURM_SUBMIT_DIR:-$KIT}"   # under sbatch BASH_SOURCE is a spooled copy
[ -f "$KIT/setup_env_arm64.sh" ] || { echo "FATAL: cannot find the kit (got KIT=$KIT); pass KIT=/path/to/arm64_kit"; exit 1; }
KIT="$(realpath -m "$KIT")"; R="$KIT"
OUT_ROOT="${OUT_ROOT:-$KIT/runs}"; HF_HOME_DIR="${HF_HOME_DIR:-$KIT/cache/hf}"; HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"
OUT_ROOT="$(realpath -m "$OUT_ROOT")"; HF_HOME_DIR="$(realpath -m "$HF_HOME_DIR")"   # the container cds into the code dir
[ "$(uname -m)" = aarch64 ] || { echo "FATAL: not an aarch64 node"; exit 1; }
[ -s "$HF_TOKEN_FILE" ] || { echo "FATAL: HF token file $HF_TOKEN_FILE missing (needed for Llama + private dataset)"; exit 1; }

case "$EXP" in
  ttrl_qwen3b)       CODE=$R/code/4dc11923; ENVN=anvil;  NGPU=4; REF=fdlcna0t ;;  # Anvil run, paper ckpt
  ttrl_llama3b)      CODE=$R/code/00e0eed5; ENVN=collab; NGPU=4; REF=is9g1zwx ;;  # 05-31 run (orig 8 GPU bs1xga192) -> 4 GPU bs1xga384
  corlplus_qrephr3b) CODE=$R/code/05a21d7f; ENVN=collab; NGPU=4; REF=SWAP_20260604_072407 ;;  # orig 4+4 GPU bs3xga128 -> 2+2 GPU bs3xga256
  *) echo "unknown EXP=$EXP"; exit 2 ;;
esac
ENV="$KIT/envs/$ENVN/venv"; [ "$ENVN" = collab ] && CUDAV=12.9.1 || CUDAV=12.8.1
SIF="${SIF:-$KIT/images/cuda-$CUDAV-cudnn-devel-arm64.sif}"
[ -x "$ENV/bin/python" ] && [ -f "$KIT/envs/$ENVN/ENV_READY" ] || { echo "FATAL: env $ENVN not built/verified (run setup_env_arm64.sh $ENVN)"; exit 1; }
[ -f "$SIF" ] || { echo "FATAL: container $SIF missing"; exit 1; }
GOT=$(nvidia-smi -L | wc -l); [ "$GOT" -ge "$NGPU" ] || { echo "FATAL: need $NGPU GPUs, got $GOT"; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Co-RL+ pins its two groups to GPUs "0,1" and "2,3" inside the container, so the job must see exactly GPUs 0-3.
[ "$EXP" != corlplus_qrephr3b ] || [ "$CUDA_VISIBLE_DEVICES" = "0,1,2,3" ] || { echo "FATAL: Co-RL+ needs the job to see GPUs 0,1,2,3 (got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); request a whole 4-GPU node"; exit 1; }
NVIS=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)
[ "$NVIS" = "$NGPU" ] || { echo "FATAL: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES lists $NVIS GPUs; this experiment needs exactly $NGPU"; exit 1; }
[ -f "$CODE/data/math500/test.json" ] || { echo "FATAL: eval file missing"; exit 1; }
# GPU-count conversion check (8 -> 4 GPUs). Keep per-device batch identical (same bnpo micro-batch),
# scale grad-accum so prompts/step = bs*nproc*ga/12 = 128 and generation_batch = bs*nproc*ga = 1536.
case "$EXP" in ttrl_qwen3b) B=3; P=4; G=128 ;; ttrl_llama3b) B=1; P=4; G=384 ;; corlplus_qrephr3b) B=3; P=2; G=256 ;; esac
[ $((B*P*G)) = 1536 ] || { echo "FATAL: generation batch $((B*P*G)) != 1536"; exit 1; }
echo "[run] per-device bs=$B procs/model=$P grad_accum=$G -> completions/step=$((B*P*G)) prompts/step=$((B*P*G/12))"

TS=$(date +%Y%m%d_%H%M%S)
TAG="$EXP"_seed"$SEED"; [ "$SMOKE" = 1 ] && TAG="SMOKE_$TAG"
JID="${SLURM_JOB_ID:-$$}"; RUN="${TAG}_${TS}_j${JID}"
OUT=$OUT_ROOT/$RUN; mkdir -p "$OUT" "$HF_HOME_DIR" || { echo "FATAL: cannot create $OUT / $HF_HOME_DIR"; exit 1; }
exec > >(tee -a "$OUT/launcher_stdout.log") 2>&1   # keep this script's own output (rc lines) inside the run directory
JOBTMP=/tmp/corl-$JID; mkdir -p "$JOBTMP" || { echo "FATAL: cannot create $JOBTMP"; exit 1; }; trap 'rm -rf "$JOBTMP"' EXIT
MAXSTEPS=(); [ "$SMOKE" = 1 ] && MAXSTEPS=(--max_steps 2)
echo "[run] EXP=$EXP SEED=$SEED SMOKE=$SMOKE CODE=$CODE ($(cat $CODE/ORIGINAL_COMMIT)) ENV=$ENV REF=$REF OUT=$OUT host=$(hostname)"
cp "$0" "$OUT/launcher.sbatch" 2>/dev/null || true
# The HF token enters the container through the environment, not the command line (where other users could see it).
export APPTAINERENV_HF_TOKEN="$(cat "$HF_TOKEN_FILE")" APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$(cat "$HF_TOKEN_FILE")"
OFFLINE_ENV=(); for v in HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE; do [ -n "${!v:-}" ] && OFFLINE_ENV+=(--env "$v=${!v}"); done

apptainer exec --nv --cleanenv -B "$KIT" -B "$OUT_ROOT" -B "$HF_HOME_DIR" --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PYTHONNOUSERSITE=1 --env PYTHONPATH="$CODE" --env ENV="$ENV" --env CODE="$CODE" \
  --env EXP="$EXP" --env SEED="$SEED" --env OUT="$OUT" --env RUN="$RUN" --env MAXSTEPS="${MAXSTEPS[*]}" \
  --env HF_HOME="$HF_HOME_DIR" --env HF_DATASETS_CACHE="$HF_HOME_DIR/datasets" \
  --env MATH500_EVAL_PATH=data/math500/test.json \
  --env WANDB_MODE=offline --env WANDB_DIR="$OUT" --env WANDB_PROJECT=Co-learning \
  --env WANDB_ENTITY=logan-yang2002-johns-hopkins-university --env DISABLE_MLFLOW_INTEGRATION=TRUE \
  --env GLOO_SOCKET_IFNAME=lo --env TMPDIR="$JOBTMP" \
  --env VLLM_CACHE_ROOT="$JOBTMP/vllm" --env TRITON_CACHE_DIR="$JOBTMP/triton" --env TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor" \
  ${OFFLINE_ENV[@]+"${OFFLINE_ENV[@]}"} "$SIF" bash -s <<'CONTAINER'
set -uo pipefail
source "$ENV/bin/activate"
cd "$CODE"
read -r -a MS <<< "$MAXSTEPS"
python - <<'PY' 2>&1 | tee "$OUT/env_check.txt"
import torch, vllm, transformers, deepspeed, accelerate, datasets, numpy, trl, flash_attn, sys
print("python", sys.version.split()[0])
for m in (torch, vllm, transformers, deepspeed, accelerate, datasets, numpy, flash_attn, trl):
    print(m.__name__, m.__version__, m.__file__)
PY
case "$EXP" in
  trl_unused) ;;
  ttrl_qwen3b)  # verbatim: wandb fdlcna0t metadata.args (Anvil, 4 GPUs) ; accelerate yaml per ckpt training_args (ZeRO-3, no offload)
    accelerate launch --config_file projects/co-grpo-dp/accelerate_zero3.yaml --num_processes 4 \
      --main_process_port 19350 --gradient_accumulation_steps 128 \
      projects/un-grpo-maj/train_un_grpo.py \
      --model_name_or_path Qwen/Qwen2.5-3B --train_dataset q1716523669/MATH-Level345 \
      --output_dir "$OUT" --run_config "$RUN" \
      --learning_rate 3e-6 --per_device_train_batch_size 3 --gradient_accumulation_steps 128 --num_train_epochs 2 \
      --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
      --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
      --max_completion_length 3072 --num_generations 12 --temperature 1.0 --temperature_eval 0.6 \
      --use_vllm --vllm_mode colocate --vllm_max_model_length 3584 --vllm_gpu_memory_utilization 0.45 \
      --logging_steps 1 --save_strategy steps --save_steps 10 --save_only_model true \
      --eval_strategy steps --eval_steps 10 --num_generations_eval 1 --per_device_eval_batch_size 1 \
      --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0 \
      --seed "$SEED" --data_seed 42 --report_to wandb --wandb_project Co-learning \
      --attn_implementation flash_attention_2 --bf16 true "${MS[@]}" 2>&1 | tee -a "$OUT/train.log"
    exit ${PIPESTATUS[0]} ;;
  ttrl_llama3b)  # verbatim: wandb is9g1zwx metadata.args + wrapper 00e0eed5 run_ungropomaj__llama32_3b.sh
    accelerate launch --config_file projects/co-grpo-dp/accelerate_zero3.yaml --num_processes 4 \
      --main_process_port 19351 --gradient_accumulation_steps 384 \
      projects/un-grpo-maj/train_un_grpo.py \
      --model_name_or_path meta-llama/Llama-3.2-3B-Instruct --train_dataset q1716523669/MATH-Level345 \
      --output_dir "$OUT" --run_config "$RUN" \
      --learning_rate 3e-6 --per_device_train_batch_size 1 --gradient_accumulation_steps 384 --num_train_epochs 2 \
      --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
      --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
      --max_completion_length 3072 --num_generations 12 --temperature 1.0 --temperature_eval 0.6 \
      --use_vllm --vllm_mode colocate --vllm_max_model_length 3584 --vllm_gpu_memory_utilization 0.45 \
      --logging_steps 1 --save_strategy steps --save_steps 10 --save_only_model true \
      --eval_strategy steps --eval_steps 10 --num_generations_eval 1 --per_device_eval_batch_size 1 \
      --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0 \
      --seed "$SEED" --data_seed 42 --report_to wandb --wandb_project Co-learning \
      --attn_implementation flash_attention_2 --bf16 true "${MS[@]}" 2>&1 | tee -a "$OUT/train.log"
    exit ${PIPESTATUS[0]} ;;
  corlplus_qrephr3b)  # verbatim: 05a21d7f run_cogrpo_heter_DECOUPLED_SWAP__qwen25_3b_rephr__llama32_3b_orig.sh
    RDV="$OUT/rdv"; rm -rf "$RDV"; mkdir -p "$OUT/group_A" "$OUT/group_B" "$RDV"
    COMMON=(--train_dataset coreward/math_rephrased --train_dataset_per_group B=coreward/math_original
      --learning_rate 3e-6 --per_device_train_batch_size 3 --gradient_accumulation_steps 256 --num_train_epochs 2
      --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03
      --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}'
      --max_completion_length 3072 --num_generations 12 --temperature 1.0 --temperature_eval 0.6
      --use_vllm --vllm_mode colocate --vllm_max_model_length 3584
      --logging_steps 1 --save_strategy steps --save_steps 10 --save_only_model true
      --eval_strategy steps --eval_steps 10 --num_generations_eval 1 --per_device_eval_batch_size 1
      --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0
      --seed "$SEED" --data_seed 42 --report_to wandb --wandb_project Co-learning
      --rendezvous_dir "$RDV" --run_config "$RUN" --bf16 true --attn_implementation flash_attention_2 "${MS[@]}")
    launch_group () { local grp="$1" gpus="$2" my="$3" peer="$4" port="$5" out="$6"
      CUDA_VISIBLE_DEVICES="$gpus" accelerate launch --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 2 --main_process_port "$port" --gradient_accumulation_steps 256 \
        projects/co-grpo-dp/train_co_grpo_dp.py --group "$grp" \
        --model_name_or_path "$my" --peer_model_name_or_path "$peer" --output_dir "$out" \
        --vllm_gpu_memory_utilization 0.45 "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"; return ${PIPESTATUS[0]}; }
    launch_group A "0,1" Qwen/Qwen2.5-3B meta-llama/Llama-3.2-3B-Instruct 19376 "$OUT/group_A" & PA=$!
    launch_group B "2,3" meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-3B 19377 "$OUT/group_B" & PB=$!
    trap 'kill $PA $PB 2>/dev/null || true' INT TERM
    wait $PA; RA=$?; wait $PB; RB=$?
    echo "[run] group A rc=$RA group B rc=$RB"; exit $(( RA || RB )) ;;
esac
CONTAINER
RC=$?
echo "[run] exit rc=$RC $(date)"
exit $RC
