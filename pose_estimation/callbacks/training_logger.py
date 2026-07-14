import os
import time
from datetime import datetime, timedelta

from pytorch_lightning import Callback


class TimestampedTrainingLogger(Callback):
    """Writes a timestamped plain-text log recording batch loss during training.

    Log file: <log_dir>/train_YYYYMMDD_HHMMSS.log
    Each line: step, epoch, batch, loss, lr, elapsed, ETA
    """

    def __init__(self, log_dir: str = "./work_dirs", log_every_n_steps: int = 50):
        self.log_dir = log_dir
        self.log_every_n_steps = log_every_n_steps
        self._file = None
        self._start_time = None

    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.log_dir, exist_ok=True)
        self._start_time = time.time()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"train_{ts}.log")
        self._file = open(log_path, "w", buffering=1)

        total_steps = trainer.estimated_stepping_batches
        header = (
            f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Log file : {log_path}\n"
            f"Total steps: {total_steps}  |  Max epochs: {trainer.max_epochs}\n"
            + "-" * 90 + "\n"
            + f"{'step':>8}  {'epoch':>5}  {'batch':>7}  {'loss':>10}  {'lr':>12}  {'elapsed':>10}  {'ETA':>18}\n"
            + "-" * 90 + "\n"
        )
        self._file.write(header)
        print(f"\n[TrainingLogger] Logging to {log_path}")
        print(f"[TrainingLogger] Total steps: {total_steps}  Max epochs: {trainer.max_epochs}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        loss = outputs["loss"].item() if isinstance(outputs, dict) else float(outputs)
        lr = trainer.optimizers[0].param_groups[0]["lr"]

        elapsed = time.time() - self._start_time
        total_steps = trainer.estimated_stepping_batches
        done = trainer.global_step
        eta_secs = (elapsed / done * (total_steps - done)) if done > 0 else 0

        elapsed_str = str(timedelta(seconds=int(elapsed)))
        eta_str = str(timedelta(seconds=int(eta_secs)))

        line = (
            f"{done:>8}  "
            f"{trainer.current_epoch:>5}  "
            f"{batch_idx:>7}  "
            f"{loss:>10.4f}  "
            f"{lr:>12.6f}  "
            f"{elapsed_str:>10}  "
            f"ETA {eta_str:>14}"
        )
        self._file.write(line + "\n")
        print(f"[LOG] {line}", flush=True)

    def on_train_end(self, trainer, pl_module):
        if self._file:
            total = str(timedelta(seconds=int(time.time() - self._start_time)))
            self._file.write("-" * 90 + "\n")
            self._file.write(
                f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                f"Total time: {total}\n"
            )
            self._file.close()
