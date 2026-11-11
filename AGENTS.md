# Instructions for the coding agent operating this kit

You are running a scientific reproduction on an ARM64 (aarch64) GPU cluster on behalf of a collaborator.
The human you work with is doing this as a favor and will not read documentation: do the work yourself, keep
their effort minimal, and ask them only the questions listed in "When to ask the human".
`README.md` (Chinese) has the same procedure with more background; this file is authoritative for you.

## Goal
Train exactly two runs and upload their results:
- `EXP=ttrl_llama3b SEED=43`
- `EXP=ttrl_llama3b SEED=44`

Everything (code commit, command line, package versions) must be identical to the original run; only the random
seed differs. The training command was already smoke-tested on an x86 cluster with this exact launcher, so **no
smoke test is needed here**: build the environment, then submit the two runs.

## Hard rules (never break them)
1. **Do not modify any file in this repository**, and do not change any training argument, package version, script
   logic, or environment variable of the training. If something fails, stop and report; never "fix" it by changing
   versions, arguments or code. The only adaptations you may make are listed in rule 2.
2. **Allowed adaptations, scheduler syntax and storage paths only**:
   - add the site's account / partition / QoS flags;
   - translate resource requests to the site's syntax (e.g. `--gpus-per-node=4` instead of `--gres=gpu:4`);
   - choose where the kit lives (a large filesystem).

   The resource shape must stay:
   - 1 node, 4 GPUs, the whole node (`--exclusive`, all memory);
   - no requeue (`--no-requeue`);
   - walltime ≥ 60 hours.
3. **Never resubmit a failed or interrupted run**, and never delete anything under `runs/`.
4. **Never print, log, commit or upload the HF token or the wandb key.**
   - They live in two dedicated files, `~/.corl_hf_token` and `~/.corl_wandb_key`, which every script picks up
     automatically.
   - **Do not touch `~/.cache/huggingface/token` or `~/.netrc`**: they may hold the human's own credentials.
5. Do not `git commit` or `git push` anything in this repository.
6. **Report at every milestone** with `bash report_status.sh`. It uploads STATUS.md to the project's private HF
   repo, and it is how the project contact follows progress. After each milestone also give the human a 2–3 line
   summary.

## Steps
Run everything from the kit directory, i.e. the root of this repository clone (`cd` into it first).

**Location first.** This repository clone must sit on a filesystem with **≥ 250 GB free**, at a path with **no spaces
and no `:`**. The downloads, environment and checkpoints are all written inside it.
- If it was cloned somewhere else (e.g. a small home directory with a quota), clone it again into a suitable place:
  `git clone https://github.com/DrStranded/Co-RL-Reb.git <big fs>/Co-RL-Reb`. Work only there.
- If you cannot tell which filesystem is large and allowed, ask the human (the cluster usually has a "scratch" or
  "work" area).

**0. Preflight.** If any of the following fails, report and ask the human (see below):
- a compute node is aarch64 (`srun ... uname -m`) and has ≥ 4 GPUs (`nvidia-smi -L`);
- `apptainer --version` ≥ 1.1 (try `module load apptainer` if it is missing);
- the site allows jobs of ≥ 60 h;
- the kit is on a filesystem with ≥ 250 GB free, and its path has no spaces or `:`.

Then set the apptainer cache and temp directories on that filesystem (home quotas are usually too small):
```bash
export APPTAINER_CACHEDIR=$PWD/../apptainer_cache APPTAINER_TMPDIR=$PWD/../apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

**1. Integrity + download** (login node with internet; about 5 GB, about 6–10 min):
```bash
sha256sum -c --quiet SHA256SUMS          # must print nothing
bash download_kit.sh collab              # must end with: ALL CODE AND 332 PACKAGE FILES PRESENT AND VERIFIED
```
Then run `bash report_status.sh`. This needs the HF token, so do step 2 first if the token is not there yet.

**2. Credentials.** The project contact's message to the human contains two secrets: their HuggingFace token
(starts with `hf_`) and their wandb API key. If `~/.corl_hf_token` and `~/.corl_wandb_key` already exist and are
non-empty, this step is done. Otherwise:
- ask the human to paste those two values from that message;
- write them like this, without printing them:
  ```bash
  umask 077; printf '%s' '<HF token>' > ~/.corl_hf_token; printf '%s' '<wandb key>' > ~/.corl_wandb_key
  ```
- all scripts (run, upload, status, prefetch) use these files automatically;
- do not touch the human's own `~/.cache/huggingface/token` or `~/.netrc`.

**3. Build the environment** on an aarch64 GPU compute node (about 30 min; the last check runs on the GPU):
```bash
srun -N1 --gres=gpu:1 -c 16 --mem=64G -t 02:00:00 <site flags> bash setup_env_arm64.sh collab 2>&1 | tee setup_collab.log
```
- **Success**: the log ends with `ALL CHECKS PASS`, and `envs/collab/pip_check.txt` has the same lines as
  `expected_pip_check_collab.txt`. Those 17 conflicts exist in the original environment: do not fix them.
- If the image pull fails because compute nodes have no internet: pull it on the login node, then rerun the setup:
  ```bash
  mkdir -p images
  apptainer pull --arch arm64 images/cuda-12.9.1-cudnn-devel-arm64.sif docker://nvidia/cuda@sha256:edb9bb6c7f45c624cb86bd1103e47471a609679a2b96ffb1bad59929f697a41e
  ```

Then run `bash report_status.sh`.

**4. Submit the two runs** (from the kit directory; `KIT=$PWD` is required because sbatch runs a spooled copy):
```bash
sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 60:00:00 <site flags> --export=ALL,KIT=$PWD,EXP=ttrl_llama3b,SEED=43 run_exp_arm64.sh
sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 60:00:00 <site flags> --export=ALL,KIT=$PWD,EXP=ttrl_llama3b,SEED=44 run_exp_arm64.sh
```
Then run `bash report_status.sh` and tell the human both job IDs.

**5. While running.** Whenever the human checks in, or about once a day, run `bash report_status.sh`.
- Each run writes to `runs/ttrl_llama3b_seed4X_<time>_j<jobid>/`.
- Expected duration is about 2 days (the original run took 23.5 h on 8×H100; here 4 GPUs).

**6. When a run ends**, whether finished, timed out or crashed:
```bash
bash collect_results.sh runs/<run dir>                   # builds runs/<run dir>_return/ (also compares the config with the original run)
DRY=1 bash upload_results.sh runs/<run dir>_return       # must end with: [dry run] all checks passed, nothing written
bash upload_results.sh runs/<run dir>_return             # must end with: [hf] all repos uploaded and verified
bash report_status.sh
```
- If the upload is interrupted, rerun the last line with `UPLOAD_RESUME=1`.
- Keep `runs/` until the project contact confirms.

## Is the training healthy?
Compare with the original run, which had the same configuration with 8 GPUs:

| | First steps | Afterwards |
|---|---|---|
| Train `reward` | about 0.38–0.41 at steps 1–2 | above 0.5 by step 10, about 0.6–0.7 from step 30 on |
| `eval_reward` (every 10 steps) | – | between 0.46 and 0.51 in the original run |

- Anything between 0.42 and 0.56 for `eval_reward` is normal seed or hardware noise.
- **Report immediately** (without killing the job) if:
  - `eval_reward` drops below 0.40;
  - the train reward collapses;
  - NaN appears;
  - or no step progress for more than 2 hours while the job is RUNNING.

## Known pitfalls
| Symptom | What to do |
|---|---|
| `apptainer: command not found` | `module load apptainer` (check `module avail`); otherwise ask the human |
| Disk quota / "no space" during image pull | set `APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR` to the big filesystem (step 0) |
| `FATAL: this node is x86_64` | You are on an x86 login node; run setup and training on aarch64 GPU nodes |
| `download_kit.sh` reports FAILED files | Network or proxy problem: retry later. **Never** substitute other package versions |
| setup stops with an error, or no `ALL CHECKS PASS` | Report the last 50 lines and stop. Do not change anything |
| 401/403 from HuggingFace (Llama download) | Token missing or wrong; ask the human for the token from the project contact |
| `FATAL: ... requeued` / job requeued by the system | Do not resubmit; report |
| Job crashes early | Report `runs/<dir>/launcher_stdout.log` tail and `train.log` tail; do not resubmit |
| Site max walltime < 60 h | Stop before submitting and ask: an interrupted run needs a special resume procedure |
| Fewer than 4 GPUs per node | Stop and ask; the launcher needs one node with 4 GPUs |
| Compute nodes without internet | Pull the image on the login node (step 3). Run `bash prefetch_hf.sh collab` on a node with internet, then add `HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1` to `--export` |

## When to ask the human (only these)
1. Site scheduler flags (account / partition / QoS) and which large filesystem to use, if you cannot find them.
2. The HF token and wandb key from the project contact, only if `~/.corl_hf_token` / `~/.corl_wandb_key` are missing.
3. A stop condition above happened: < 4 GPUs per node, walltime < 60 h, no apptainer, setup failed, job failed.
   Report first with `bash report_status.sh`, then ask.
