from pathlib import Path
import yaml
import pickle
from typing import Dict, Any
from torch import nn


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


def load_model(
    model_path: str | Path, config: dict, model_class: nn.Module, cfg: dict, device: str
) -> tuple[nn.Module, dict]:
    """
    Load a saved model and its vocabulary mapping.

    Args:
        model_path: Path to the saved model checkpoint
        config: Optional model configuration dictionary

    Returns:
        tuple: (loaded_model, app_to_idx_mapping)
    """
    checkpoint = torch.load(model_path)
    app_to_idx = checkpoint["app_to_idx"]

    # Get vocabulary size from the saved state dict
    vocab_size = len(app_to_idx)

    # Initialize model with same configuration
    model = model_class(
        vocab_size=vocab_size + 1,
        d_model=cfg.get("model", {}).get("d_model", 64),
        nhead=cfg.get("model", {}).get("nhead", 4),
        seq_length=cfg.get("model", {}).get("seq_length", 64),
        n_layers=cfg.get("model", {}).get("num_encoder_layers", 3),
    )

    # Load the saved state
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval()  # Set to evaluation mode

    logger.info(f"Model loaded from {model_path}")

    return model, checkpoint["app_to_idx"]
