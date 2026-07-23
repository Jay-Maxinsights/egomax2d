# Design — Modules, interfaces & config

## Module layout (`inference/egomax2d_pipeline/`)

```
__init__.py
main.py            # ENTRANCE: parse config -> factories build components ->
                   #   run_sequential (with monitor brackets) -> write result; argparse CLI
model.py           # load_model (+ optional forward helper); imports MODEL_CFG from configs.constant
pipeline.py        # class Pipeline: preprocess / inference / decode
monitor.py         # class ResourceMonitor  (see resource-monitor.md)
configs/
  __init__.py
  schema.py        # dataclasses: InputConfig, ProcessorConfig, OrchestratorConfig,
                   #   MonitorConfig, OutputConfig, PipelineConfig
  constant.py      # IMG_SIZE, HM_SIZE, HM_SCALE, CONF_THRESH, CAMS, MODEL_CFG,
                   #   _ENCODER_CFG, DEFAULT_CKPT   (lifted verbatim from _dev.py)
  utils.py         # PipelineConfig.from_yaml(path); merge_cli(cfg, args); load_calibration()
io/
  __init__.py
  reader.py        # FrameBatch, InputSource(ABC), FolderInputSource
  writer.py        # InferenceResult, save_pt
```

Placement rules:
- `configs/` imports nothing heavy (no torch) → testable without a GPU.
- Factories live in `main.py` (they import the heavy component modules), keeping that
  dependency out of `configs/`.
- `MODEL_CFG` stays a **constant** in `configs/constant.py` (must match the checkpoint
  architecture); config only carries the `ckpt` path.
- Prefer stable lower-level `pose_estimation.*` imports. Where required to avoid a
  second numerical implementation, the new package may import helpers from the legacy
  inference scripts. The reverse dependency is forbidden: neither
  `inference_heatmap_egomax2d.py` nor `inference_heatmap_egomax2d_dev.py` may import or
  re-export anything from `inference.egomax2d_pipeline`.

See `../reference.md` for the exact existing symbols each module reuses.

## `io/reader.py`

```python
@dataclass
class FrameBatch:
    indices: list[int]                 # frame indices, ascending
    left:  list[str | np.ndarray]      # aligned with indices (paths now; arrays later)
    right: list[str | np.ndarray]

class InputSource(ABC):
    @abstractmethod
    def calibration(self) -> dict: ...            # {"videoFL": {...}, "videoFR": {...}}
    @abstractmethod
    def __iter__(self) -> Iterator[FrameBatch]: ...

class FolderInputSource(InputSource):
    def __init__(self, left_dir: str, right_dir: str, calibration,
                 batch_size: int, step: int = 1, max_frames: int = 0):
        ...
    # calibration: a dir path (uses load_session_calib), a calibration.json path, or a
    #   pre-parsed dict. Normalize in __init__ to the {"videoFL":.., "videoFR":..} dict.
    # __iter__: parse frame_%08d.jpg in both folders, take the sorted intersection,
    #   apply step/max_frames, group into FrameBatch of batch_size (last may be partial).
    #   left/right hold the JPEG paths (fed to preprocess_gpu).
```

The batch is the unit of work everywhere — there is deliberately no per-frame object.

## `pipeline.py`

```python
class Pipeline:
    def __init__(self, calibration: dict, ckpt: str, device: str = "cuda",
                 rotate: str = "right", preprocess: str = "gpu",
                 conf_thresh: float = CONF_THRESH):
        # build remaps ONCE:
        #   left  = build_remap(dict(_EGOBODY3M_DS_PARAMS[1]), calibration["videoFL"], rotate)
        #   right = build_remap(dict(_EGOBODY3M_DS_PARAMS[2]), calibration["videoFR"], rotate)
        # self.model = load_model(ckpt).to(device)

    def preprocess(self, batch: FrameBatch) -> Tensor:   # (B, 2, 3, 256, 256) on device
        # build flat [f0_L, f0_R, f1_L, f1_R, ...] paths + aligned maps, then
        #   preprocess_gpu(...)  (or batch_preprocess(...) if preprocess == "cpu");
        #   reshape to (B, 2, 3, 256, 256).

    def inference(self, x: Tensor) -> np.ndarray:        # (B*2, 26, 64, 64)
        # feats = model.forward_backbone(x)
        # hm = model.conv_heatmap(feats.view(B*2, *feats.shape[2:]))
        # return hm.cpu().numpy()

    def decode(self, hm: np.ndarray, batch: FrameBatch, result: "InferenceResult") -> None:
        # for i, idx in enumerate(batch.indices):
        #     left  = decode_heatmap(hm[i*2 + 0])
        #     right = decode_heatmap(hm[i*2 + 1])
        #     result.add(idx, left, right)
```

Batch indexing: `hm[i*2]` = left, `hm[i*2+1]` = right for `batch.indices[i]`.

## `io/writer.py`

```python
@dataclass
class InferenceResult:
    frames: dict[int, dict] = field(default_factory=dict)   # the data-contracts.md shape

    def add(self, idx, left, right):
        self.frames[idx] = {
            "left":  {"joints": left[0],  "confidences": left[1]},
            "right": {"joints": right[0], "confidences": right[1]},
        }

    def to_pt_dict(self) -> dict:
        return self.frames

def save_pt(result: InferenceResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)       # batch_main never did this
    torch.save(result.to_pt_dict(), path)
```

## Config system (`configs/schema.py`, `configs/utils.py`)

### Parse, then build

```
YAML / CLI ──parser (configs/utils)──▶ PipelineConfig (plain values)
           ──factories (main.py)──▶ FolderInputSource, Pipeline, ResourceMonitor, writer
```

- **Parser** — `PipelineConfig.from_yaml(path)` + `merge_cli(cfg, args)` (CLI overrides
  win). Produces nested dataclasses of plain values (no torch, no GPU) → testable in
  isolation.
- **Factories** — `build_input_source(cfg.input)`, `build_pipeline(cfg.processor,
  calibration)`, `build_monitor(cfg.monitor)`, `build_writer(cfg.output)`. Each is a
  small `if type == ...: return X(...)` with a `raise ValueError` default — the single
  place a future variant plugs in.

### Schema (`configs/schema.py`)

```python
@dataclass
class InputConfig:        type: str = "folder"; left_dir = ""; right_dir = ""; calibration = ""; batch_size: int = 32; step: int = 1; max_frames: int = 0
@dataclass
class ProcessorConfig:    ckpt: str = DEFAULT_CKPT; device = "cuda"; rotate = "right"; preprocess = "gpu"; conf_thresh: float = CONF_THRESH
@dataclass
class OrchestratorConfig: type: str = "sequential"
@dataclass
class MonitorConfig:      enabled: bool = True; warmup: int = 10; report_json: str | None = None
@dataclass
class OutputConfig:       format: str = "pt"; path: str = ""
@dataclass
class PipelineConfig:
    input: InputConfig; processor: ProcessorConfig; orchestrator: OrchestratorConfig
    monitor: MonitorConfig; output: OutputConfig
    @classmethod
    def from_yaml(cls, path) -> "PipelineConfig": ...   # yaml.safe_load -> nested dataclasses
```

### YAML shape

```yaml
input:
  type: folder            # folder | magicap (future)
  left_dir:  <path>
  right_dir: <path>
  calibration: <path>     # dir | calibration.json path | inline dict
  batch_size: 32
processor:
  ckpt:   <path>
  device: cuda
  rotate: right           # right | left | none
  preprocess: gpu         # gpu (nvjpeg) | cpu (cv2)
  conf_thresh: 0.01
orchestrator:
  type: sequential        # sequential | taskmanager (future)
monitor:
  enabled: true
  warmup: 10              # batches excluded from timing stats
  report_json: null       # optional path to dump the report
output:
  format: pt              # pt | toon (future)
  path: <path>
```

### CLI (`main.py`)

`--config <yaml>` plus overrides `--left-dir --right-dir --calibration --ckpt --out
--batch-size --device` (merged with CLI winning). Future features are a config value +
one factory branch: `input.type: magicap`, `orchestrator.type: taskmanager`,
`output.format: toon`.

### `load_calibration` (`configs/utils.py`)

`load_session_calib` takes a **directory** and reads `calibration.json` inside it. The
episode input passes a calibration *path*, so add a thin
`load_calibration(path_or_dir_or_dict) -> dict` that dispatches: dict → passthrough; dir
→ `load_session_calib`; `*.json` file → read + parse the same
`calibrations[0].calibration_text` YAML (keep parsing identical to `load_session_calib`).
