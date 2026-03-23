from pathlib import Path

def resolve_config_paths(config_path: str) -> list[Path]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Expected a .yaml or .yml file, got: {path}")
        return [path]

    if path.is_dir():
        yaml_files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}
        )
        if not yaml_files:
            raise ValueError(f"No YAML files found in directory: {path}")
        return yaml_files

    raise ValueError(f"Unsupported config path: {path}")
