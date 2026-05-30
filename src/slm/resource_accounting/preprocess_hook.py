from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

from .config import ResourceConfig
from .reporter import Reporter


class PreprocessResourceHook:
    """Times each preprocessing pipeline stage and reports resource usage.

    Designed to wrap slm.preprocess.pipeline.runner — call
    wrap_stage_runner() to get an instrumented version of your run function,
    or use the stage_timer() context manager around individual stages.

    Usage with pipeline/runner.py:

        hook = PreprocessResourceHook(resource_cfg, checkpoint_dir="artifacts/checkpoints")
        with hook.stage_timer("snapshot"):
            runner.run_stage("snapshot", ...)

        # or wrap the whole runner:
        instrumented_run = hook.wrap_stage_runner(runner.run)
        instrumented_run(graph, config)

        # at the end write the summary
        hook.finish()

    Metrics collected per stage:
      - wall_time_sec
      - rows_processed (if stage returns a count or manifest has it)
      - peak_gpu_mem_gb (sampled after stage completes)
      - throughput_rows_per_sec (derived)
    """

    def __init__(
        self,
        resource_cfg: ResourceConfig | None = None,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        self.resource_cfg = resource_cfg or ResourceConfig()
        self.reporter = Reporter(
            log_to_wandb=self.resource_cfg.log_to_wandb,
            write_json_summary=self.resource_cfg.write_json_summary,
            checkpoint_dir=checkpoint_dir,
        )

        # List of per-stage result dicts accumulated during the pipeline run.
        self._stage_results: list[dict[str, Any]] = []

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def stage_timer(
        self,
        stage_name: str,
        *,
        rows_processed: int | None = None,
    ) -> Generator[None, None, None]:
        """Time a single pipeline stage and record results.

        with hook.stage_timer("minihash", rows_processed=manifest.num_rows):
            runner.run_stage("minihash", ...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._record_stage(stage_name, elapsed, rows_processed=rows_processed)

    # ── Wrapper ───────────────────────────────────────────────────────────────

    def wrap_stage_runner(
        self, run_fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Return a wrapped version of a runner function that times each call.

        The wrapped function passes all args/kwargs through unchanged.
        It expects run_fn to accept a stage_name kwarg or first positional
        arg — adjust to match your pipeline/runner.py signature.

        TODO: adapt the stage_name extraction to match runner.py's actual
        function signature once you've settled on it.
        """
        def instrumented(*args: Any, **kwargs: Any) -> Any:
            # TODO: extract stage_name from args/kwargs to match runner.py
            stage_name = kwargs.get("stage_name", args[0] if args else "unknown")
            rows = kwargs.get("rows_processed", None)

            with self.stage_timer(stage_name, rows_processed=rows):
                return run_fn(*args, **kwargs)

        return instrumented

    # ── Internal ──────────────────────────────────────────────────────────────

    def _record_stage(
        self,
        stage_name: str,
        wall_time_sec: float,
        *,
        rows_processed: int | None,
    ) -> None:
        """Collect metrics for one stage and forward to reporter."""
        peak_mem_gb = self._sample_gpu_mem()

        throughput = (
            rows_processed / wall_time_sec
            if rows_processed is not None and wall_time_sec > 0
            else None
        )

        result: dict[str, Any] = {
            "stage": stage_name,
            "wall_time_sec": wall_time_sec,
            "rows_processed": rows_processed,
            "peak_gpu_mem_gb": peak_mem_gb,
            "throughput_rows_per_sec": throughput,
        }

        self._stage_results.append(result)

        # Mirror to reporter's actual samples so they appear in the JSON summary.
        self.reporter._actual_samples.append(result)

        print(
            f"[resource_accounting] stage={stage_name!r}  "
            f"time={wall_time_sec:.1f}s  "
            + (f"rows={rows_processed:,}  " if rows_processed else "")
            + (f"mem={peak_mem_gb:.2f}GB  " if peak_mem_gb else "")
            + (f"tput={throughput:.0f} rows/s" if throughput else "")
        )

        if self.resource_cfg.log_to_wandb:
            self._wandb_log_stage(result)

    def _sample_gpu_mem(self) -> float | None:
        """Sample current GPU memory after a stage completes."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_reserved() / 1024 ** 3
        except ImportError:
            pass
        return None

    def _wandb_log_stage(self, result: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        stage = result["stage"]
        payload: dict[str, Any] = {}

        if result["wall_time_sec"] is not None:
            payload[f"resource/preprocess/{stage}/wall_time_sec"] = result["wall_time_sec"]
        if result["rows_processed"] is not None:
            payload[f"resource/preprocess/{stage}/rows_processed"] = result["rows_processed"]
        if result["peak_gpu_mem_gb"] is not None:
            payload[f"resource/preprocess/{stage}/peak_gpu_mem_gb"] = result["peak_gpu_mem_gb"]
        if result["throughput_rows_per_sec"] is not None:
            payload[f"resource/preprocess/{stage}/throughput_rows_per_sec"] = result["throughput_rows_per_sec"]

        if payload:
            wandb.log(payload)

    # ── Finish ────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        """Call after all pipeline stages are done to print the stage summary."""
        if not self._stage_results:
            return

        total_time = sum(r["wall_time_sec"] for r in self._stage_results)
        total_rows = sum(
            r["rows_processed"] for r in self._stage_results
            if r["rows_processed"] is not None
        )

        print(
            f"\n[resource_accounting] preprocessing complete — "
            f"{len(self._stage_results)} stages, "
            f"{total_time:.1f}s total, "
            f"{total_rows:,} total rows"
        )