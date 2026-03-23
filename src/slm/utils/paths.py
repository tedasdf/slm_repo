import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class ContextResult:
    run_id: str
    run_dir: str
    config_snapshot_path: str
    manifest_path: str
    started_at: str


def _read_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def start_run(
    config_path: str,
    process_type: str,
    dataset_version_id: int,
    artifacts_root: str = "artifacts",
    extras: Optional[Dict[str, Any]] = None,
) -> ContextResult:
    started_dt = datetime.now()
    run_id = started_dt.strftime("%Y%m%d_%H%M%S")
    started_at = started_dt.isoformat()

    run_dir = os.path.join(artifacts_root, process_type, run_id)
    os.makedirs(run_dir, exist_ok=True)

    config_snapshot_path = os.path.join(run_dir, "config.snapshot.yaml")
    shutil.copy2(config_path, config_snapshot_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {
        "run_id": run_id,
        "process_type": process_type,
        "dataset_version_id": dataset_version_id,
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "status": "running",
        "config_snapshot_path": "config.snapshot.yaml",
        "extras": extras or {},
    }
    _write_manifest(manifest_path, manifest)

    return ContextResult(
        run_id=run_id,
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        manifest_path=manifest_path,
        started_at=started_at,
    )


def update_manifest(manifest_path: str, updates: Dict[str, Any]) -> None:
    manifest = _read_manifest(manifest_path)
    manifest.update(updates)
    _write_manifest(manifest_path, manifest)


def finish_run(
    manifest_path: str,
    status: str = "completed",
    extras: Optional[Dict[str, Any]] = None,
    **fields: Any,
) -> None:
    manifest = _read_manifest(manifest_path)

    finished_dt = datetime.now()
    finished_at = finished_dt.isoformat()

    started_at_str = manifest["started_at"]
    started_dt = datetime.fromisoformat(started_at_str)
    elapsed_seconds = (finished_dt - started_dt).total_seconds()

    manifest["finished_at"] = finished_at
    manifest["elapsed_seconds"] = elapsed_seconds
    manifest["status"] = status

    if extras is not None:
        manifest["extras"] = {**manifest.get("extras", {}), **extras}

    manifest.update(fields)
    _write_manifest(manifest_path, manifest)