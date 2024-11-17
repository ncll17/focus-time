from pathlib import Path
import yaml
import pickle
from typing import Dict, Any


def validate_config(config: Dict[str, Any]) -> None:
    """Validate that all required paths exist in config."""
    required_paths = [
        "data.raw_data_path",
        "data.app_mappings_path",
        "data.exploded_df_path",
        "data.sequences_path",
        "data.vocab_path",
    ]

    for path_key in required_paths:
        # Navigate nested dict using path_key
        current = config
        try:
            for key in path_key.split("."):
                current = current[key]
        except KeyError:
            raise KeyError(f"Missing required config key: {path_key}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config


def safe_load_pickle(path: Path):
    """Safely load pickle file if it exists."""
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return None


def safe_save_pickle(data: Any, path: Path) -> None:
    """Safely save data to pickle file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)
