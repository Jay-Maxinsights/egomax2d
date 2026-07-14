# H100 训练环境完整报告

- **生成时间**: 2026-06-30
- **GPU**: NVIDIA H100 80GB HBM3
- **镜像**: `egoposeformer:h100`（基于 `Dockerfile.a100`，CUDA 12.1 完全兼容 H100）

---

## 一、环境信息

| 项目 | 版本 / 状态 |
|------|-------------|
| Docker | 29.2.1（需 `sudo`，用户 `ubuntu` 不在 docker 组） |
| GPU | NVIDIA H100 80GB HBM3 |
| CPU 核数 | 26 |
| 宿主机内存 | 221 GB，`/dev/shm` 111 GB（`--ipc=host` 模式下容器共享） |
| nvidia-container-toolkit | 已装，CDI 配置 `/var/run/cdi/nvidia.yaml` 存在 |
| Docker runtime | 默认 `runc`，通过 `--gpus all` 走 nvidia hook |

---

## 二、镜像构建结果

- **构建状态**: 成功（exit 0）
- **DISK USAGE**（解压后实际磁盘占用）: **27.5GB**
- **CONTENT SIZE**（压缩去重后）: **9.32GB**
- **Image ID**: `cce922a54d14`

### 容器内验证

```
torch:            2.1.0
cuda available:   True
device:           NVIDIA H100 80GB HBM3
cuda version:     12.1
lightning:        2.1.0
mmcv:             2.1.0
```

### 镜像容量拆解

> 27.5GB 中约 **95%（≈26GB）来自 NVIDIA 官方 `-devel` 基础镜像**，
> 自己 Dockerfile 加的只有约 **1.5GB（5%）**。

| 来源 | 大小 | 内容 |
|----|-----:|------|
| `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` 基础层 | ~26GB | PyTorch + conda + CUDA toolkit + cuDNN devel |
| `RUN apt-get install` | 392MB | git / ffmpeg / libgl1 / libglib2.0 / zip 等 |
| `pip install requirements.txt` | 736MB | lightning、opencv、numba、matplotlib、pandas 等 |
| `pip install mmcv==2.1.0` | 330MB | mmcv 预编译包（cu121/torch2.1） |
| 代码 + 零碎包 | ~8MB | jsonargparse + `pip install -e .` |

---

## 三、容器启动脚本

**文件**: `run_h100_container.sh`

```bash
sudo docker run --gpus all --ipc=host --shm-size=32g \
  --ulimit nofile=65536:65536 \
  --ulimit core=0 \
  -v "$(pwd)":/workspace/egoposeformer \
  -v "$(pwd)/data":/workspace/egoposeformer/data \
  -v "$(pwd)/work_dirs":/workspace/egoposeformer/work_dirs \
  -it egoposeformer:h100 bash
```

关键参数说明：
- `--ipc=host`: 容器和宿主机共享 /dev/shm（111 GB），DataLoader 多进程必须
- `--shm-size=32g`: 仅在 ipc=host 不生效时的兜底（实际以 ipc=host 为准）
- `--ulimit nofile=65536:65536`: 解决 batch=256 DataLoader collate 产生大量 fd 问题（默认 soft=1024 会 "Too many open files"）
- `--ulimit core=0`: 禁止生成 core dump，防止崩溃时 6GB × N 个文件撑满磁盘

---

## 四、数据路径

| 位置 | 说明 |
|------|------|
| `./data/EgoBody3M_256/{train,validation,test}/` | 本地 ext4，约 161 GB，zip 包格式 |
| `/mnt/nas-q/LuDong/EgoBody3M_256` | NAS 原始路径，当前机器未挂载 |

**数据完整性**: test 集 4 个序列缺 `images_256.zip`（含 148625961194245 等），不影响训练/验证，manifest 自动排除。

---

## 五、Manifest 缓存（level-0 过滤）

**脚本**: `scripts/build_level0_manifest.py`（默认路径 `./data/EgoBody3M_256`）

训练只使用 `annotation_level=0` 的帧，manifest 预扫一次后缓存为 JSON，避免每次训练重扫 zip。

| split | 文件 | 序列数 | 帧数 |
|-------|------|--------|------|
| train | `.manifest_level0.json` | 1709 | 2,409,633 |
| validation | `.manifest_level0.json` | 94 | 170,632 |
| test | `.manifest_level0.json` | 71 | 130,184 |

---

## 六、训练配置

**文件**: `configs/egobody3m_r18_heatmap.yaml`

| 参数 | 值 | 说明 |
|------|----|------|
| `batch_size` | 256 | VRAM 约 32GB（~40%），IO 可追上 |
| `lr` | 0.001 | linear scaling: 256/640 × 0.0025 |
| `workers` | 8 | 16 → shm 过大，降回 8 |
| `precision` | `bf16-mixed` | H100 原生 bf16，优于 fp16 |
| `data_root` | `./data/EgoBody3M_256` | 直接指向数据目录（无软链） |
| `annotation_levels` | `[0]` | 只用 level-0 数据 |

**文件**: `pose_estimation/pl_wrappers/heatmap.py`

| 参数 | 值 | 说明 |
|------|----|------|
| `prefetch_factor` | 2 | train + val 两处均设置 |
| `persistent_workers` | 删除 | 原 True，导致 shm 累积 94GB → OOM Killer |

**in-flight 数据量**: 8 workers × prefetch 2 × batch(256) ≈ **46 GB** < shm 111 GB

---

## 七、踩过的坑

### 1. bus error / shm 容量不足

**配置**: workers=16 × prefetch=4 → 64 batches in-flight × ~2.89GB = 185 GB > shm 111 GB

**报错**: `RuntimeError: DataLoader worker process died unexpectedly` 和 bus error

**修复**: workers=8，prefetch=2 → 46 GB in-flight

### 2. Too many open files

**原因**: DataLoader collate 每个 batch 创建数百个 shm fd，容器默认 soft ulimit nofile=1024

**修复**: 启动脚本加 `--ulimit nofile=65536:65536`

### 3. core dump 撑满磁盘

**原因**: 两次崩溃生成 11 个 × 6GB core 文件 = 66GB，磁盘告急

**修复**: `sudo rm core.*` 删除；启动脚本加 `--ulimit core=0`

### 4. OOM Killer 杀死主进程（最终根因）

**现象**: 训练约 50 steps 后 process 输出 `Killed`，dmesg 显示：
```
shmem-rss: 98,586,196 kB  ← /dev/shm 占用 ~94GB
```

**根因**: `persistent_workers=True` 使 worker 进程跨 epoch 存活，PyTorch 的 shm tensor 引用在每个 step 累积，不被释放。50 steps × ~1.9GB/step = 94GB → OOM Killer 触发。

**修复**: 从 `train_dataloader()` 和 `val_dataloader()` 中删除 `persistent_workers=True`

---

## 八、run.py 已有优化（无需改动）

```python
torch.set_float32_matmul_precision("medium")  # H100 bf16 matmul 最快档
```

---

## 九、训练命令

进入容器后：

```bash
python run.py fit --config configs/egobody3m_r18_heatmap.yaml
```

---

## 十、待完成

- [ ] SSH 配置 GitHub push（`ssh-keygen` → 公钥加 GitHub → 改 remote 为 `git@github.com:ludong-max/egoposeformer.git`）
- [ ] 监控 GPU 利用率（去掉 `persistent_workers` 后是否仍有 IO 瓶颈）
- [ ] 如 GPU 利用率 < 50%，考虑增加 workers 或预解压图片到本地
