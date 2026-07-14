from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ATTN_KEY_RE = re.compile(r"^attention_diagnostics/layer_(\d+)/(.+)$")


def _metric_value(value: Any) -> Any:
    try:
        import math

        if isinstance(value, float) and not math.isfinite(value):
            return None
    except Exception:
        pass
    return value


def _run_path(entity: str | None, project: str) -> str:
    return f"{entity}/{project}" if entity else project


def _choose_change(
    *,
    run: Any,
    change_tag_prefix: str,
    change_summary_key: str | None,
) -> str:
    summary = dict(run.summary)

    if change_summary_key:
        value = summary.get(change_summary_key)
        if value is not None:
            return f"{change_tag_prefix}{value:g}" if isinstance(value, float) else str(value)

    tags = list(getattr(run, "tags", []) or [])
    matches = [tag for tag in tags if str(tag).startswith(change_tag_prefix)]
    if matches:
        # Some scripts include both a rounded and exact tag. Prefer the readable one.
        return sorted(matches, key=len)[0]

    return run.name or run.id


def _diagnostic_rows_from_history(
    *,
    run: Any,
    change: str,
    include_all_steps: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    latest_by_layer: dict[int, dict[str, Any]] = {}
    any_history = False

    for hist in run.scan_history(page_size=1000):
        diag_items: dict[int, dict[str, Any]] = defaultdict(dict)
        for key, value in hist.items():
            match = ATTN_KEY_RE.match(str(key))
            if not match:
                continue
            layer = int(match.group(1))
            metric = match.group(2)
            diag_items[layer][metric] = _metric_value(value)

        if not diag_items:
            continue

        any_history = True
        step = hist.get("_step", hist.get("train/step"))
        for layer, metrics in diag_items.items():
            row = {
                "change": change,
                "layer": layer,
                "step": step,
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
            }
            row.update(metrics)
            if include_all_steps:
                rows.append(row)
            else:
                latest_by_layer[layer] = row

    if include_all_steps:
        return rows
    if any_history:
        return [latest_by_layer[layer] for layer in sorted(latest_by_layer)]

    return []


def _diagnostic_rows_from_summary(*, run: Any, change: str) -> list[dict[str, Any]]:
    by_layer: dict[int, dict[str, Any]] = defaultdict(dict)
    summary = dict(run.summary)

    for key, value in summary.items():
        match = ATTN_KEY_RE.match(str(key))
        if not match:
            continue
        layer = int(match.group(1))
        metric = match.group(2)
        by_layer[layer][metric] = _metric_value(value)

    rows = []
    for layer in sorted(by_layer):
        row = {
            "change": change,
            "layer": layer,
            "step": summary.get("_step"),
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
        }
        row.update(by_layer[layer])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export W&B attention_diagnostics metrics to an Excel file."
    )
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "transformer_instability"),
    )
    parser.add_argument(
        "--tag",
        default="qk-init-step0",
        help="Only export runs containing this W&B tag.",
    )
    parser.add_argument(
        "--change-tag-prefix",
        default="qk-init-m-",
        help="Tag prefix used for the first column, e.g. qk-init-m-.",
    )
    parser.add_argument(
        "--change-summary-key",
        default="config/qk_gain_init",
        help="Prefer this W&B summary key for the change value when present.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Export every logged step instead of the latest row per run/layer.",
    )
    parser.add_argument(
        "--out",
        default="artifacts/attention_diagnostics_qk_init_step0.xlsx",
    )
    args = parser.parse_args()

    import pandas as pd
    import wandb

    api = wandb.Api()
    runs = api.runs(_run_path(args.entity, args.project), filters={"tags": args.tag})

    rows: list[dict[str, Any]] = []
    for run in runs:
        change = _choose_change(
            run=run,
            change_tag_prefix=args.change_tag_prefix,
            change_summary_key=args.change_summary_key,
        )

        run_rows = _diagnostic_rows_from_history(
            run=run,
            change=change,
            include_all_steps=args.all_steps,
        )
        if not run_rows:
            run_rows = _diagnostic_rows_from_summary(run=run, change=change)
        rows.extend(run_rows)

    if not rows:
        raise SystemExit(
            f"No attention_diagnostics rows found for tag={args.tag!r} "
            f"in {_run_path(args.entity, args.project)!r}."
        )

    df = pd.DataFrame(rows)

    leading = ["change", "layer", "step", "run_name", "run_id", "state"]
    metric_cols = sorted(col for col in df.columns if col not in leading)
    df = df[[col for col in leading if col in df.columns] + metric_cols]
    df = df.sort_values(["change", "layer", "step"], kind="stable")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
