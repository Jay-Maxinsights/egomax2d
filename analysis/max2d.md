# EgoMax2D 数据集统计与读取计划

- Generated: 2026-07-10
- 数据路径: `data/EgoMax2D/`
- 统计脚本: 本文档所有数字由遍历全部 82 个 session 的 `estimations.toon` / `calibration.json` / `images/` 得出（PyYAML 解析，全量非抽样）。

---

## 1. 总体规模

| 项目 | 数值 |
|---|---|
| Session 数（ULID 命名目录） | 82 |
| 相机 | 2 路（`head-front-left`, `head-front-right`，头戴前向双目） |
| 每路总帧数 | 179,377 |
| 总图片数（双路） | 358,754 张 JPG |
| 图片分辨率 | 2592 × 1944（全部一致） |
| 帧率 | 30 fps（全部一致） |
| 总时长 | ≈ 1.66 小时（单路口径） |
| 单 session 长度 | 302 ~ 4,388 帧（10 s ~ 146 s） |
| 磁盘占用 | 255 GB（平均 ≈ 745 KB/张） |

结构完全规整：82 个 session 全部具备双相机目录、`estimations.toon`、`calibration.json`、`calibration.mcap`，且**左右相机帧数一一相等、toon 帧数与图片数完全一致，无一缺漏**。

## 2. 目录与文件格式

```
data/EgoMax2D/
└── 01KW3RYY4Z67HKAVYMAG8QG9WH/        # session（ULID）
    ├── images/
    │   ├── head-front-left/frame_XXXXXXXX.jpg    # 8 位帧号，连续
    │   └── head-front-right/frame_XXXXXXXX.jpg
    ├── estimations.toon               # 2D 关键点人工标注（YAML 文本，~0.6–3 MB）
    ├── calibration.json               # 标定（内嵌 YAML 文本）
    └── calibration.mcap               # 同一标定的 mcap 封装
```

### 2.1 图片命名（✅ 已对齐为 0 基）

原始数据各 session 的起始帧号**不一致**：68 个从 `frame_00000000` 开始，14 个从 1（4 个）/ 2（8 个）/ 43 / 173 开始。帧号在 session 内部连续无空洞，左右相机起始一致，且图片张数 == toon 帧数 == metadata 声明帧数。

**2026-07-10 已用 `scripts/max2D_Id_align.py --apply` 将 14 个非 0 起始 session 的 58,308 个文件整体平移重命名为 0 基**（复验通过：82×2 路全部 start=0、连续、总张数 358,754 不变）。对齐后 `frame_%08d.jpg` 与 toon 帧 key `'%06d'` 直接同号对应，无需再排序对齐。该脚本幂等，数据重新拉取后可再跑一次。

**不要**用 toon 里的 `original_frame_id` 拼文件名：同一 session 左右相机的偏移都不一样（如 L 为 3..527、R 为 1..525），它只是裁剪前原始流的帧号。

### 2.2 estimations.toon（实为 YAML）

顶层两个 key：`head_front_left` / `head_front_right`（下划线，与图片目录的连字符不同）。每路：

```yaml
head_front_left:
  metadata:
    image_size: [1944, 2592]        # [H, W]
    number_of_frames: 525
    fps: 30.0
    support_keypoints: [left_elbow, right_elbow, left_shoulder, right_shoulder, pelvis]
    frame_image_extension: .jpg
  frames:
    '000000':
      left_elbow:
        pixel_coords: [901.968, 1821.743]   # [x, y]，x∈[0,2592), y∈[0,1944)
        confidence_score: 0.8348
        status: null
      ...
      original_frame_id: 3                  # 原始流帧号，勿用于拼文件名
      original_timestamp: 1782538275391027200   # 纳秒
```

**5 个关键点**：left/right_elbow、left/right_shoulder、pelvis。

**标注结构：每 3 帧一个人工关键帧 + 中间插值**（关键帧间隔分布 gap=3 占 46.3 万次绝对主导；帧级关键帧率 33.2%）。各条目按 `pixel_coords × confidence_score × status` 交叉分类：

| 类别 | 判定条件 | 条数 | 可用性 |
|---|---|---|---|
| 一次标注（人工关键帧） | 有坐标, conf>0, status=null | 382,841 | ✅ 可用，weight 1.0（conf 为标注辅助/可见性分，非检测器置信度） |
| 插值 | 有坐标, conf=0, status=null | 869,159 | ⚠️ 暂不使用（质量未抽验；坐标带长小数尾巴可辨认） |
| 人工二次修改 | 有坐标, `manual_annotated` | 51,942 | ✅ 可用，质量最高（约 10.6% 落在无一次标注的帧上） |
| 遮挡但有坐标 | 有坐标, `occluded` | 8,228 | ✅ 可用 |
| 出画（关键帧） | 无坐标, conf>0, `out_of_frame` | 36,956 | ❌ 按 coords-is-None 过滤，**不能看 conf**（conf 是残留低分） |
| 出画（插值段） | 无坐标, conf=0, `out_of_frame` | 444,581 | ❌ 不可用 |
| 其余（`not_in_frame` 等） | — | 63 | 噪声级别 |

**时间同步**：左右相机同帧 index 的 timestamp 差普遍 < 0.2 ms，个别 session 首帧差约 24 ms（±0.7 帧），双目可按 index 直接配对。

### 2.3 每关键点标签量（双路合计，每点 358,754 帧次）

"人工可用" = 一次标注(conf>0 有坐标) + 二次修改(manual) + occluded 带坐标：

| 关键点 | 一次标注 | 二次修改(manual) | occluded带坐标 | **人工可用合计** | 插值 | 无坐标 |
|---|---|---|---|---|---|---|
| left_elbow     | 28.1% | 1.7% | 5,048 | **112,017 (31.2%)** | 59.6% |  9.2% |
| right_elbow    | 28.4% | 1.8% | 2,323 | **110,939 (30.9%)** | 60.5% |  8.5% |
| left_shoulder  |  7.9% | 4.0% |   314 | **43,210 (12.0%)**  | 23.9% | 64.0% |
| right_shoulder | 13.7% | 4.0% |    42 | **63,493 (17.7%)**  | 35.4% | 47.0% |
| pelvis         | 28.6% | 2.9% |   501 | **113,352 (31.6%)** | 62.9% |  5.5% |

**二次修改（manual_annotated）分布**：81/82 个 session 含二次修改（仅 `01KWB4RBYS2W0JDMRWSXGHX20P` 无），覆盖稀疏不均——按含 manual 的帧占比算各 session 从 0.2% 到 33%，多数 5%~20%。二次修改集中在肩部（各 ~14.5k 条 vs 肘部 ~6.5k 条），即一次标注最难标准/最易缺失的点。

要点：
- **肘部和 pelvis 人工标注量大**（每点约 11 万条，>90% 帧有坐标）；
- **肩部大量 out_of_frame**（左肩 64% 无坐标）——头戴前向相机的固有视野限制，与 EgoBody3M 上肩部难点一致；左右肩不对称（64% vs 47%）可能与相机安装/佩戴偏转有关；
- 一次标注的 conf 字段是标注辅助/可见性分（肩部均值 ~0.5、肘部 ~0.84），**不代表人工标注不可靠**，不必据此降权。

### 2.4 标定（calibration.json）

`calibrations[0].calibration_text` 是内嵌 YAML。要点汇总：

| 项目 | 数值 |
|---|---|
| 相机模型 | **Double Sphere 鱼眼**（`model_type: ds`，params = `[fx, fy, cx, cy, xi, alpha]`） |
| 内参覆盖 | 5 路相机 `videoFL / videoL / videoC / videoFR / videoR` |
| 外参覆盖 | 仅 `videoFL`、`videoFR`，`transform_ref_current` = `[tx, ty, tz, qx, qy, qz, qw]`，参考系 `imu` |
| 双目基线 | FL–FR 6.5 ~ 7.8 cm（各设备略有差异） |
| 标定组数 | 82 个 session 共 **20 组不同标定**（多台设备/多次标定） |
| 加载方式 | **必须按 session 各自加载**，不能全局共用 |

videoFL/FR 分辨率 2592×1944（与图片一致），videoL/C/R 为 1920×1080（本数据集未提供其图像）。

相机名与图片目录对应：`head-front-left ↔ videoFL`，`head-front-right ↔ videoFR`（分辨率与外参均吻合）。`calibration.mcap` 是同一标定的 mcap 封装，可忽略，以 JSON 为准。

## 3. 与 EgoBody3M / 本仓库的关系

- 相机即 EgoBody3M 的前向双目对（cam1=videoFL, cam2=videoFR），但这里是**原始全分辨率 2592×1944**（EgoBody3M 中 FL/FR 为 1024×1280 crop/缩放版本），且**只有 2D 人工标注（关键帧）、无 3D GT、无 headset 6DoF 轨迹**（IMU 位姿未随数据提供）。
- 关键点可映射到 EgoBody3M 26 关节骨架（见 `analysis/skeleton.md`）：
  - left/right_shoulder → joint 2 (`b_l_arm`) / 6 (`b_r_arm`)（盂肱关节）；
  - left/right_elbow → joint 3 (`b_l_forearm`) / 7 (`b_r_forearm`)；
  - pelvis → joint 14/20（左右髋）中点，与 `egobody3m_3dpose.py` 的 `_PELVIS_IDS` 定义一致。
- 定位：**in-the-wild 2D 人工标注数据**，适合给 V2/V3 做 2D 重投影损失的域适应/半监督微调，而非独立训练集（缺 3D GT）。

## 4. 数据读取计划

### 4.1 预处理与索引（一次性离线）

1. **建 meta 缓存**（对齐现有 `meta_preload` 做法）：遍历 82 session，解析 toon + calibration，落成 npz/npy：
   - `keypoints[N, V=2, K=5, 2]`（原图像素坐标）、`conf[N, V, K]`、`source[N, V, K]`（0=null / 1=插值 / 2=一次标注 / 3=manual二次修改 / 4=occluded带坐标）、`timestamps[N, V]`；
   - `image_index`：session 内**排序后**的文件名列表（解决起始帧号不一坑）；
   - 每 session 的 DS 内参 + FL/FR 外参（imu 系）。
   - toon 是纯文本 YAML，全量解析一次约几分钟，务必缓存，训练时零 YAML 解析。
2. **图片降采样缓存**（对齐 `EgoBody3M_256` 做法）：2592×1944 → 256×256 离线转存，255 GB 可压到 ~10 GB 量级；2D 坐标按同一变换缩放。分辨率/畸变与 EgoBody3M 输入不完全同源，建议先做一版直接 resize，若分布差异明显再考虑用 DS 模型重映射到与 EgoBody3M cam1/2 一致的成像几何。

### 4.2 Dataset 类（`pose_estimation/datasets/egomax2d/`）

仿照 `EgoBody3M3DPoseDataset` / `..._seq.py`：

- **`EgoMax2DDataset`**（逐帧）：输出 `img [V=2, 3, 256, 256]`、`kp2d [V, K, 2]`（归一化到输入分辨率）、`kp2d_weight [V, K]`、`session_id/frame_idx`；
- **`EgoMax2DSeqDataset`**（窗口，供 V3 时序分支）：`img [T, V, 3, 256, 256]` + 对应 2D 目标；session 内帧连续、30 fps，天然支持滑窗；无 6DoF headset pose 输入，V3 该输入需置零/mask（或后续用 VIO 补）。

### 4.3 监督策略（关键设计点）

- **权重**：关键帧标签均为人工标注，一次标注、二次修改、occluded 带坐标统一给 **weight = 1.0**（occluded 可单独打可见性标记）；插值帧是纯几何插值，建议**降权（~0.2）或起步不用**。可用量：帧级关键帧 119,007 帧（33.2%，双目口径），肘/pelvis 每点约 11 万条人工坐标；
- **损失**：V3 输出 3D → 用各 session 的 DS 模型投影回 FL/FR 两视图 → 与 2D 人工标注做加权 L1/Huber（仅 5 个可映射关节）；肩部缺失率高，损失里按 mask 处理即可；注意 conf>0 但坐标为 null 的 out_of_frame 条目（36,956 条）必须按不可用过滤，不能只看 conf；
- **划分**：按 session 划 train/val（如 74/8），避免同 session 帧泄漏；20 组标定尽量在两侧都有覆盖。manual 修改帧质量最高，可额外留作伪 GT 验证集。

### 4.4 待确认事项

- 插值帧坐标质量未人工抽验，用前建议可视化抽查若干 session（快速动作段插值可能漂移）；
- pelvis 2D 定义（标注规范中的 pelvis 像素点）与我们 "左右髋中点投影" 是否一致，需在重投影损失上线前用 EgoBody3M 交叉验证一版；
- 无 headset 6DoF：V3 的 head pose 辅助输入在该数据上的替代方案（置零 / VIO / 仅用不依赖该输入的分支微调）。


## 5. TODO 状态

| # | 事项 | 状态 |
|---|---|---|
| ① | 数据清理：14 个非 0 起始 session 的帧号对齐 | ✅ 已完成（2026-07-10，`scripts/max2D_Id_align.py --apply` 全部对齐为 0 基，复验通过，详见 §2.1） |
| ② | "抖动严重"——插值帧（source=1）质量问题 | ⏳ 未处理，用前需可视化抽验若干 session（快速动作段插值可能漂移） |