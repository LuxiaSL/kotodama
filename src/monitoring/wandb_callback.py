"""
Wandb integration for luxia-base training.

Logs training metrics (loss, LR, throughput) and geometric health
metrics (from GeometricMonitor) to Weights & Biases. Also maintains
a local JSONL log for offline analysis.

Usage in training loop::

    wb = WandbLogger(config)
    # In training loop:
    wb.log_step(step, {"loss": loss, "lr": lr, ...})
    wb.log_geo(step, geo_metrics)
    # At end:
    wb.finish()

Adapted from patterns in ~/projects/4p-sc/voice_wandb_callback.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Wandb + JSONL dual logger for training metrics.

    All metrics are namespaced:
    - ``train/*`` — loss, z_loss, grad_norm, tokens/sec
    - ``optim/*`` — learning rates, optimizer stats
    - ``geo/*`` — geometric health metrics (Tier 1 + Tier 2)
    - ``perf/*`` — GPU memory, step time, throughput
    - ``data/*`` — tokens consumed, epoch, data position

    Uses ``commit=False`` on ``wandb.log()`` so that all metrics
    for a step are grouped under one global step counter.
    """

    def __init__(
        self,
        project: str = "luxia-base",
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        log_dir: str = "logs",
        enabled: bool = True,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.enabled = enabled
        self._wandb_run = None
        self._jsonl_path: Optional[Path] = None
        self._jsonl_file = None

        if not enabled:
            logger.info("Wandb logging disabled")
            return

        # Sliding windows for slope computation
        self._loss_history: deque[tuple[int, float]] = deque(maxlen=100)
        self._rankme_history: deque[tuple[int, float]] = deque(maxlen=50)

        # Set up JSONL local log
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = log_dir_path / "metrics.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a")
        logger.info("JSONL metrics log: %s", self._jsonl_path)

        # Initialize wandb
        try:
            import wandb

            # Check for API key
            api_key = os.environ.get("WANDB_API_KEY", "")
            if not api_key:
                logger.warning(
                    "WANDB_API_KEY not set — wandb will run in offline mode. "
                    "Set it from ~/projects/4p-sc/run_v2.1.sh or wandb login."
                )

            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or ["pretraining", "luxia-base"],
                resume="allow",
            )
            logger.info("Wandb initialized: %s", self._wandb_run.url or "offline")
        except ImportError:
            logger.warning("wandb not installed — logging to JSONL only")
            self._wandb_run = None
        except Exception as e:
            logger.warning("wandb init failed (%s) — logging to JSONL only", e)
            self._wandb_run = None

    def log_step(
        self,
        step: int,
        loss: float,
        z_loss: float,
        grad_norm: float,
        muon_lr: float,
        adamw_lr: float,
        tokens_per_sec: float,
        tokens_consumed: int,
        gpu_mem_gb: float,
        step_time_s: float,
    ) -> None:
        """Log core training metrics for one optimizer step."""
        # Track loss history for slope computation
        self._loss_history.append((step, loss))

        metrics = {
            "train/loss": loss,
            "train/z_loss": z_loss,
            "train/perplexity": math.exp(min(loss, 20.0)),  # cap to avoid overflow
            "train/grad_norm": grad_norm,
            "optim/muon_lr": muon_lr,
            "optim/adamw_lr": adamw_lr,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/gpu_mem_gb": gpu_mem_gb,
            "perf/step_time_s": step_time_s,
            "data/tokens_consumed": tokens_consumed,
            "data/tokens_consumed_B": tokens_consumed / 1e9,
        }

        # Loss slope over recent window (detect plateaus / divergence)
        if len(self._loss_history) >= 10:
            slope = _linear_slope(list(self._loss_history))
            metrics["train/loss_slope"] = slope

        self._log(step, metrics)

    def log_geo(self, step: int, geo_metrics: dict[str, float]) -> None:
        """Log geometric health metrics (already namespaced with geo/ prefix)."""
        if not geo_metrics:
            return

        # Track RankMe slope
        rankme = geo_metrics.get("geo/rankme_last")
        if rankme is not None:
            self._rankme_history.append((step, rankme))
            if len(self._rankme_history) >= 5:
                slope = _linear_slope(list(self._rankme_history))
                geo_metrics["geo/rankme_slope"] = slope

        self._log(step, geo_metrics)

    def log_custom(self, step: int, metrics: dict[str, Any]) -> None:
        """Log arbitrary metrics."""
        self._log(step, metrics)

    def _log(self, step: int, metrics: dict[str, Any]) -> None:
        """Write metrics to both wandb and JSONL."""
        # Add step to metrics for JSONL
        record = {"step": step, "timestamp": time.time(), **metrics}

        # JSONL (always, even if wandb is down)
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._jsonl_file.flush()
            except Exception as e:
                logger.debug("JSONL write failed: %s", e)

        # Wandb
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step, commit=False)
            except Exception as e:
                logger.debug("wandb log failed: %s", e)

    def commit(self, step: int) -> None:
        """
        Commit the current step's metrics to wandb.

        Call this once per step after all log_step/log_geo calls
        to advance wandb's step counter.
        """
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log({}, step=step, commit=True)
            except Exception:
                pass

    def finish(self) -> None:
        """Clean up wandb run and close JSONL file."""
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except Exception:
                pass

        logger.info("WandbLogger finished")


def _linear_slope(points: list[tuple[int, float]]) -> float:
    """
    Compute the slope of a linear regression through (x, y) points.

    Used for detecting trends in loss, RankMe, etc. over sliding windows.
    Positive slope = increasing, negative = decreasing, near-zero = plateau.
    """
    n = len(points)
    if n < 2:
        return 0.0
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    den = sum((x - x_mean) ** 2 for x in x_vals)
    if den < 1e-10:
        return 0.0
    return num / den
