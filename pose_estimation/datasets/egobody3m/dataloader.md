# EgoBody3M Dataloader 设计说明

## 文件结构

```
pose_estimation/datasets/egobody3m/
├── egobody3m_heatmap.py   # 阶段一：Heatmap 训练用
├── egobody3m_3dpose.py    # 阶段二：3D Pose 训练用
└── __init__.py
```

---

## 一、索引构建（两个 Dataset 共用）

### `_build_sequence_index(split_dir)`

**目的**：把磁盘上 N 个 zip 文件展开成 `[(seq_id, frame_idx), ...]` 的平铺样本列表，供 `__getitem__` 按绝对索引访问。

**流程**：

```
split_dir/
├── 1003162681118387.images.zip     ← 扫描所有 .images.zip → 得到 seq_id 列表
├── 1003162681118387.metadata.zip   ← 打开对应 .metadata.zip → 数 .json 文件数 = 帧数 N
├── ...

→ 展开为 [(seq_id_0, 0), (seq_id_0, 1), ..., (seq_id_0, N-1),
           (seq_id_1, 0), ...]
```

**Manifest 缓存**：第一次调用时逐一打开 metadata zip 统计帧数（train 约 1833 个，耗时约 1 分钟）。统计完毕后将结果写入 `split_dir/.manifest.json`，后续直接读取，启动时间降至秒级。

```json
// .manifest.json 格式
{
  "1003162681118387": 1716,
  "1005857773939983": 1684,
  ...
}
```

---

## 二、Zip 文件读取（两个 Dataset 共用）

### `_open_zip(path)` — 线程局部 ZipFile 缓存

DataLoader 使用多 worker 并行时，每个 worker 是独立进程。如果每次 `__getitem__` 都 open/close zip，IO 开销很大。

解决方案：使用 `threading.local()` 在每个 worker 进程内缓存已打开的 `ZipFile` 对象，整个训练期间不关闭。

```
worker-0: _zip_cache.d = { "seqA.images.zip": <ZipFile>, "seqA.metadata.zip": <ZipFile>, ... }
worker-1: _zip_cache.d = { ... }   ← 独立副本，互不干扰
```

读取文件内容：
```python
raw_bytes = _open_zip(zip_path).read("seq_id/frame0000_cam1.jpg")
```

---

## 三、图像读取与预处理

### `_read_ego_img(img_zip_path, seq_id, frame_idx, cam_id)`

**读取路径**：`{seq_id}/frame{NNNN}_cam{1或2}.jpg`（JPEG 单色图）

**预处理流程**：

```
原始 JPEG (1280 × 1024, 单色)
    │
    ▼ copyMakeBorder(top=128, bottom=128)   # 上下补零，使高度 1024 → 1280
    │
    ▼ resize(1280×1280 → 256×256, INTER_LINEAR)
    │
    ▼ stack × 3 channels                   # 灰度复制为 3 通道（与 UnrealEgo 格式对齐）
    │
    ▼ / 255.0 → float32 [0, 1]
    │
    → [3, 256, 256]
```

**为什么先 padding 再 resize**：

cam1/cam2 原始分辨率是 1280×1024（非正方形，宽高比 1.25）。若直接 resize 到 256×256 会对水平和垂直方向使用不同缩放比，造成图像内容变形，同时 heatmap 坐标也需要用两套不同的缩放系数。

Padding 后统一为 1280×1280，缩放比固定为 256/1280 = 0.2，图像和坐标变换完全一致。

---

## 四、Heatmap GT 生成（`egobody3m_heatmap.py`）

### 数据来源

从 metadata JSON 的 `projected_joint_positions_cam{1或2}_px` 字段直接读取，格式为：

```python
[[u_px, v_px, depth_cm], ...]   # shape: [26, 3]
```

这是数据集提供的已投影坐标（已按各相机鱼眼模型计算），无需自行做相机投影运算。

### 可见性过滤

```python
if depth <= 0:
    continue   # 关节在相机后方，跳过
```

### 坐标变换

原始投影坐标 → heatmap 坐标的完整变换链：

```
(u, v) in 1280×1024 pixel space
    │
    ├── u_hm = u              × (64 / 1280)
    └── v_hm = (v + 128)      × (64 / 1280)   ← v 先加 padding 偏移量再统一缩放
```

等价展开：
```
u_hm = u × (256/1280) × (64/256) = u × (64/1280)
v_hm = (v + 128) × (256/1280) × (64/256) = (v + 128) × (64/1280)
```

与图像预处理的坐标系完全自洽（图像补了 128px 零行，heatmap 坐标也偏移 128px）。

### Gaussian 生成

```python
hm[y0:y1, x0:x1] = exp(-(dx² + dy²) / (2 × σ²))
```

- `σ = 2.0`（默认），单位：heatmap 像素
- 峰值 = 1.0，范围 = 3σ = 6 像素
- 关节中心超出 64×64 边界时该通道全为零

### 输出

| key | shape | dtype |
|-----|-------|-------|
| `img` | `[2, 3, 256, 256]` | float32 |
| `heatmap_gt` | `[2, 26, 64, 64]` | float32 |

---

## 五、3D Pose GT 生成（`egobody3m_3dpose.py`）

### 骨盆计算

EgoBody3M 没有单独的 pelvis 关节，取左右髋关节（joint 14 = 右髋，joint 20 = 左髋）的中点作为骨盆：

```python
pelvis_world = joints_world[[14, 20]].mean(axis=0)   # [3], 单位 cm
```

### GT Pose（骨盆相对坐标）

```python
gt_pose = joints_world - pelvis_world   # [26, 3], 单位 cm
```

模型预测的是骨盆相对坐标，与 UnrealEgo 的 `gt_local_pose` 定义一致。

### origin_3d（骨盆在相机系的位置）

模型的 Transformer 需要将 3D 关节点投影到 2D 图像特征上做 cross-attention，投影公式为：

```python
cam_3d = local_3d + origin_3d   # 骨盆相对坐标 + 骨盆在相机系坐标 = 关节在相机系坐标
```

因此 `origin_3d` 需要是骨盆在每个相机坐标系下的位置。

**当前实现（近似）**：

cam1 和 cam2 都安装在头显正前方，用头显坐标系近似相机坐标系：

```python
T_H_W = inv(world_from_headset_xf_cm)          # headset_from_world [4, 4]
pelvis_headset = T_H_W[:3,:3] @ pelvis_world + T_H_W[:3,3]   # [3]

origin_3d = [pelvis_headset, pelvis_headset]    # [2, 1, 3]，两路相机共用同一近似值
```

**局限性**：cam1 和 cam2 相对于头显中心有各自的固定安装偏移（外参），当前实现将这个偏移忽略。若后续拿到 cam1/cam2 的相机-头显外参，在此处替换即可，其余逻辑不变。

### 输出

| key | shape | dtype | 说明 |
|-----|-------|-------|------|
| `img` | `[2, 3, 256, 256]` | float32 | cam1 (front-left) + cam2 (front-right) |
| `gt_pose` | `[26, 3]` | float32 | 骨盆相对，单位 cm |
| `origin_3d` | `[2, 1, 3]` | float32 | 骨盆在（近似）相机系，单位 cm |
| `dataset_idx` | int | — | 样本绝对索引 |

---

## 六、与模型的接口对应

### Heatmap 阶段（`EgoPoseFormerHeatmap`）

```python
# egoposeformer_heatmap.py forward_train
def forward_train(self, img, heatmap_gt):
    # img:        [B, V, 3, 256, 256]   V=2
    # heatmap_gt: [B, V, 26, 64, 64]
```

DataLoader batch 后自动在 dim=0 拼接，V 维度来自 Dataset 返回的 shape。

### 3D Pose 阶段（`EgoPoseFormer`）

```python
# egoposeformer.py forward_train
def forward_train(self, img, origin_3d, gt_pose):
    # img:       [B, 2, 3, 256, 256]
    # origin_3d: [B, 2, 1, 3]
    # gt_pose:   [B, 26, 3]
```

---

## 七、数值规格汇总

| 参数 | 值 |
|------|----|
| 使用相机 | cam1（前向左）、cam2（前向右） |
| 原始分辨率 | 1280 × 1024（单色） |
| Padding 后 | 1280 × 1280 |
| 模型输入 | 256 × 256 × 3 |
| Heatmap 大小 | 64 × 64 |
| 关节数 | 26 |
| 骨盆关节 | joint 14（右髋）和 joint 20（左髋）的中点 |
| 坐标单位 | cm |
| Gaussian σ | 2.0 heatmap px（默认） |
