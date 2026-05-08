from pathlib import Path

import yaml


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text())


def test_baseline_cosine_scheduler_has_tmax():
    cfg = load_yaml("configs/train/baseline.yaml")
    scheduler = cfg.get("scheduler", {})
    assert scheduler.get("scheduler_type") != "cosine" or scheduler.get("t_max") is not None


def test_baseline_data_source_matches_file_pattern():
    cfg = load_yaml("configs/train/baseline.yaml")
    data = cfg["data"]
    train_paths = data.get("train_paths") or ""
    if train_paths.endswith(".json.gz") or "*.json" in train_paths:
        assert data["source_type"] in {"json", "jsonl"}
