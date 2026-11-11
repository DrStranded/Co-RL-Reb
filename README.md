# ARM64（aarch64）复现包：Co-GRPO / TTRL 3B 实验

用途：在 ARM64 GPU 集群（如 GH200）上，用**和原始 run 完全相同的代码 commit、命令和软件版本**重跑实验，只换随机种子。
启动脚本里的训练命令与在 x86 集群上冒烟通过的启动脚本**逐字节相同**。

**要跑的**：TTRL · Llama-3.2-3B-Instruct，seed 43 和 seed 44，各一次（`EXP=ttrl_llama3b`）。
另外两个实验（`ttrl_qwen3b`、`corlplus_qrephr3b`）也能用这个文件夹跑，没被要求就不用跑。

做不到"一模一样"的地方，**停下来问发包给你的人（下称"联系人"）**，不要自己调参数凑结果。

---

## 这个文件夹里有什么
| 文件 | 内容 |
|---|---|
| `download_kit.sh` | 补全文件：按 commit SHA 从公开的 [trl-projects](https://github.com/williamium3000/trl-projects) 仓库取原始代码（git 按 hash 保证一字不差）；从 PyPI、download.pytorch.org、GitHub 发布页下载安装包，逐个核对 sha256 |
| `kit_files_collab.tsv`、`kit_files_anvil.tsv` | 下载清单，每行是：放到哪里、sha256、官方地址 |
| `pins_collab.txt`、`pins_anvil.txt` | 两套环境的完整包清单（全部钉死版本） |
| `setup_env_arm64.sh` | 在 ARM GPU 节点上离线建环境并自检 |
| `run_exp_arm64.sh` | 启动脚本，训练命令与 x86 版逐字节相同 |
| `compare_config.py`、`ref_*.json` | 把实际生效的训练参数和原始 run 逐字段比对；`ref_*.json` 是三个原始 run 的参数记录（已去掉路径等无关字段） |
| `expected_pip_check_collab.txt` | collab 环境 `pip check` 的预期输出（17 行，原环境里就有，不用修） |
| `collect_results.sh`、`upload_results.sh` | 跑完后打包（含自动参数比对），再上传到联系人的 HuggingFace（私有仓）和 wandb |
| `prefetch_hf.sh` | 只在计算节点不能上网时用：提前把模型和数据下到缓存 |
| `SHA256SUMS` | 本文件夹所有文件的校验和 |

两套环境：
- **collab**：Python 3.11.14，torch 2.9.1+cu129、vLLM 0.14.0、transformers 4.57.1、DeepSpeed 0.18.6、accelerate 1.12.0、datasets 4.6.1、numpy 1.26.3。TTRL Llama 用这一套；
- **anvil**：Python 3.12.13，torch 2.9.0+cu128、vLLM 0.11.2 等，只有 `ttrl_qwen3b` 用。

---

## 0. 先准备好这些
- **硬件和软件**：ARM64（aarch64）GPU 节点，**单节点至少 4 张卡**（每节点只有 1 张卡的话先联系联系人）；**apptainer** 1.1 以上；`git`、`curl`。
- **网络**：能访问 github.com、pypi.org、files.pythonhosted.org、download.pytorch.org、huggingface.co，以及 Docker Hub（拉一次镜像）。计算节点不能上网见 §7。
- **磁盘**（都在同一个文件系统上）：
  - 下载约 5G，collab 环境约 11G，镜像约 6G；apptainer 缓存另需约 25G；inode 要几十万个；
  - 每个跑完的 TTRL Llama run 保留全部 checkpoint，约 **97G**，两个 seed 约 200G。
- **路径**：不能有空格或冒号（脚本会检查）。建好环境后不要挪动这个文件夹。
- **时间**：原始 run 在 8×H100 上用了 23.5 小时；这里 4 卡、梯度累积翻倍，预计约 2 天。单个作业最长不到 60 小时的话，先告诉联系人。
- **联系人单独发给你的 HF token 和 wandb key**：不要放进这个文件夹，不要提交到任何仓库，做完就删。
  - HF token 存到 `~/.cache/huggingface/token`，`chmod 600`；
  - wandb key 在上传前 `export WANDB_API_KEY=...`。

## 1. 拿到并补全
```
git clone https://github.com/DrStranded/Co-RL-Reb.git /大容量目录/arm64_repro   # 或者：tar -xzf arm64_repro.tar.gz -C /大容量目录/
cd /大容量目录/arm64_repro                                                      # 路径里不能有空格或冒号
sha256sum -c --quiet SHA256SUMS        # 没有任何输出 = 全部一致
bash download_kit.sh collab            # 最后必须打印 ALL CODE AND 332 PACKAGE FILES PRESENT AND VERIFIED
```

## 2. 建环境（必须在 ARM GPU 节点上，最后一步要在 GPU 上自检）
```
export APPTAINER_CACHEDIR=/大容量目录/apptainer_cache APPTAINER_TMPDIR=/大容量目录/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
srun -N1 --gres=gpu:1 -c 16 --mem=64G -t 02:00:00 bash setup_env_arm64.sh collab 2>&1 | tee setup_collab.log
```
- 用 sbatch 也行：在这个文件夹里提交，并加 `--export=ALL,KIT=$PWD`。
- 镜像会**按 digest** 自动拉取：nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 的 ARM64 版，`sha256:edb9bb6c7f45c624cb86bd1103e47471a609679a2b96ffb1bad59929f697a41e`。
- 计算节点不能上网的话，先在登录节点拉好：
  ```
  mkdir -p images
  apptainer pull --arch arm64 images/cuda-12.9.1-cudnn-devel-arm64.sif docker://nvidia/cuda@sha256:edb9bb6c7f45c624cb86bd1103e47471a609679a2b96ffb1bad59929f697a41e
  ```
- 环境必须建在默认位置 `envs/`（不要设 `ENVROOT`），启动脚本只认这个位置。

**通过标准**：日志最后打印 `ALL CHECKS PASS`。自检逐项包括：
- 每个包、torch 三件套和构建工具都是钉死的版本；
- TRL 来自原始 commit 目录；
- 实际加载的 cuDNN 是 9.10.2、NCCL 是 2.27.5；
- 三个 C 扩展（crcmod、netifaces、pycosat）真的编出来了；
- GPU 上实跑 flash-attn，并和 PyTorch 自带的注意力对比；
- latex2sympy2 实际解析一次；
- 能 import vLLM。

`envs/collab/pip_check.txt` 应该和 `expected_pip_check_collab.txt` 那 17 行一致，**不要去修**。

## 3. 冒烟（可以跳过）

**联系人已经在 x86 集群上用同一条命令（训练命令逐字节相同）替你冒烟过 `ttrl_llama3b`**：
- 参数比对正好是下面列出的 4 处不同，"只在新 run 里"的字段为 0；
- 前两步 reward 0.380 / 0.388、长度 705.9 / 730.3，和原始 run 同一水平。

所以建好环境（§2 打印 `ALL CHECKS PASS`）后，可以直接正式跑（§4）。想自己冒烟的话：
```
SMOKE=1 EXP=ttrl_llama3b SEED=42 sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 02:00:00 --export=ALL,KIT=$PWD run_exp_arm64.sh
envs/collab/venv/bin/python compare_config.py runs/<新生成的 SMOKE_ttrl_llama3b_seed42_... 目录> json:ref_ttrl_llama3b_is9g1zwx.json
```
比对输出的三块，预期如下：
1. **两边都有、值不同的字段**：只能是这 4 个：
   - `gradient_accumulation_steps` 和 `steps_per_generation`：192 → 384（原始 8 卡，这里 4 卡）；
   - `max_steps`：-1 → 2；
   - `save_total_limit`：3 → none。
2. **只在原始记录里的字段**：一批模型配置字段（`architectures`、`vocab_size` 之类），正常。
3. **只在新 run 里的字段**：应为空。

冒烟前两步的指标应和原始 run 同一 seed 的第 1、2 步在同一水平：
- 第 1 步：reward 0.4049，mean_length 753.4，entropy 1.615；
- 第 2 步：reward 0.3841，mean_length 725.7，entropy 1.414。

这两组数也在 `ref_ttrl_llama3b_is9g1zwx.json` 的 `first_steps` 里。

跳过冒烟也没关系：§6 打包时会自动做同样的比对，结果随上传一起交给联系人核对。

## 4. 正式跑
```
sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 60:00:00 --export=ALL,KIT=$PWD,EXP=ttrl_llama3b,SEED=43 run_exp_arm64.sh
sbatch -N1 --gres=gpu:4 --exclusive --mem=0 --no-requeue -t 60:00:00 --export=ALL,KIT=$PWD,EXP=ttrl_llama3b,SEED=44 run_exp_arm64.sh
```
- 在这个文件夹里提交，并且要传 `KIT=$PWD`（sbatch 执行的是脚本的副本）。
- `--no-requeue`：作业被系统重排会悄悄从第 0 步重来，脚本也会拒绝以重排作业的身份运行。
- 每 10 步有一次 MATH-500 验证分（`eval_reward`）。明显异常（比如掉到 0.3 以下，或出现 NaN）就停下来告诉联系人。

## 5. 中断了怎么办
节点挂了、超时、报错，都一样：
1. **不要重新提交**，不要删任何东西；
2. 把 `runs/<run>/launcher_stdout.log`、`train.log` 末尾和 Slurm 的输出文件发给联系人。

checkpoint 只存了模型（没有优化器状态），续训要用专门的步骤。

## 6. 跑完后：打包、上传
```
bash collect_results.sh runs/<run>
export WANDB_API_KEY=<联系人给你的 key>
DRY=1 bash upload_results.sh runs/<run>_return        # 演练：检查所有凭证，打印计划，什么都不写
bash upload_results.sh runs/<run>_return              # 真上传（在能上网的节点上跑）
```
- **打包**：生成 `runs/<run>_return/`，约 12G（checkpoint 用硬链接，不额外占空间）。打包失败不会留下半成品。内容：
  - `model/best/`：训练中验证分最高的 checkpoint，打平取较早的一步；
  - `model/endpoint/`：最后一个 checkpoint；
  - `train.log`、`trainer_state.json`、`selection.txt`；
  - `compare_config.txt`：与原始 run 的自动参数比对；
  - `first_steps.txt`：前两步指标；
  - wandb 离线日志、`env_check.txt`、`launcher_stdout.log`、`SHA256SUMS`。
- **上传前的检查**，任何一项不过就什么都不写：
  - wandb key 必须属于联系人的 entity；
  - HF token 必须有写权限；
  - 冒烟 run 一律拒绝；
  - 已存在的仓不覆盖。
- **上传顺序**：先把 wandb 离线日志 sync 上去，再在 HF 建**私有**仓（名字形如 `repro-ttrl-llama32-3b-instruct-seed43-arm-full`）传 `best/`、`endpoint/` 和日志，最后逐个文件核对大小。
- **上传断了**：用 `UPLOAD_RESUME=1 bash upload_results.sh runs/<run>_return` 接着传。
- **用哪个 Python**：ARM 节点上自动用建好的环境（在容器里跑）；在 x86 节点上传先 `pip install --user huggingface_hub wandb`（Python ≥ 3.9）。
- 上传完把脚本打印的仓名发给联系人，原始 run 目录先别删。

## 7. 计算节点不能上网时
在能上网的 ARM 节点上、建好环境后跑一次 `bash prefetch_hf.sh collab`。提交训练时加上 `HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1`（放进 `--export=ALL,...` 里）。启动脚本只在这两个变量被设置时才传进容器。这条路径没有实测过，第一次用时先冒烟。

---

## 和原始 run 一样的
- 代码 commit；
- 命令行：只改了
  - 采样 seed：data_seed 固定 42，题目顺序不变；
  - 输出路径；
  - wandb 离线；
  - 保留全部 checkpoint；
  - 卡数换算：每卡 batch 不变，梯度累积等比放大，启动前自动核对；
- ZeRO-3、不 offload；`sequence_mask`、3584、验证温度 0.6；
- 训练数据：`q1716523669/MATH-Level345`，自原始 run 以来未修改；
- 主要包的版本号。

## 和原始 run 不一样、无法消除的
- **硬件**：原始 run 是 H100（TTRL Llama 是 8×H100），这里是 ARM 节点、4 卡。数值上会有正常的细微差别。
- **Python 补丁版本**：原始机器是 3.11.2，这里是 3.11.14（与已验证的 x86 环境相同）。
- **少一个包**：collab 清单少一个与训练无关的 `iotop`。
- **容器里的 CUDA 编译器**：已验证的 x86 环境是 12.8，这里是 12.9.1。只在 vLLM/Triton 现场编译时用到。
- **cuDNN/NCCL**：torch 通过 RPATH 加载 pip 钉死的 cuDNN 9.10.2 / NCCL 2.27.5，优先于镜像自带的版本；自检会核对实际加载的版本。
- **torchvision / torchaudio 文件名**：ARM64 上不带 `+cuXXX` 后缀，来自同一个 PyTorch 源，版本号相同。
- **flash-attn**：同一份 2.8.3 源码编出来的社区预编译包。

## 已经做过的检查
- **下载清单**：清单里的每个文件，官方源给出的 sha256 都和验证过的那份一致（collab 332 个、anvil 247 个，0 个不符）。在 x86 上模拟完整下载一遍，结果与完整包逐个相同。
- **离线安装和编译**：
  - 用 pip 自己的解析器按 ARM64 + glibc 2.35 做离线安装演练，全部解析成功；
  - 所有源码包在 x86 上用同版本 Python 和构建工具实际编过；
  - 没有任何二进制需要比 Ubuntu 22.04 更新的 glibc。
- **启动脚本**：训练命令与 x86 版逐字节相同；外层的各项检查（sbatch 定位、卡数、重排、SMOKE 取值、路径、token 不进命令行、离线开关）都用替身命令逐项实测过。
- **打包、上传、比对工具**：在 x86 集群的真实 run 上实测通过。上传只做过演练，没有真传。

## 尚未验证
还没在 ARM 机器上实际跑过。只能在 ARM GPU 节点上验证的是：
- ARM wheel 实际装上并 import；
- DeepSpeed 的 CUDA 分支编译；
- GPU 上的 flash-attn 和 vLLM；
- 驱动版本和内存页大小：setup 第一行会打印。

第一次跑 `setup_env_arm64.sh`，以它的自检结果为准。
