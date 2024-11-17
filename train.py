from loguru import logger
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data.load_and_preprocess import (
    load_raw_data,
    create_exploded_df,
    create_sequences,
    create_vocab,
)
from data.datasets import AppSequenceDataset, PreloadedDataset
from models.transformer_models import ShallowTransformerWithAttention
from training.iterations import train_epoch, evaluate
from utils import load_config, safe_load_pickle, safe_save_pickle, load_model


def setup_data(cfg):
    """Setup and preprocess all required data."""
    logger.info("Converting path strings to Path objects")
    data_paths = {k: Path(v) for k, v in cfg["data"].items()}

    # Load or create exploded_df
    exploded_df = safe_load_pickle(data_paths["exploded_df_path"])
    if exploded_df is None:
        logger.info("Loading raw data and creating exploded_df")
        df_day_point, _ = load_raw_data(
            data_paths["raw_data_path"], data_paths["app_mappings_path"]
        )
        exploded_df = create_exploded_df(df_day_point)
        safe_save_pickle(exploded_df, data_paths["exploded_df_path"])

    # Load or create sequences
    sequences = safe_load_pickle(data_paths["sequences_path"])
    if sequences is None:
        logger.info("Creating sequences from exploded_df")
        seq_length = cfg.get("model", {}).get("seq_length", 64)
        sequences = create_sequences(exploded_df, seq_length)
        safe_save_pickle(sequences, data_paths["sequences_path"])

    # Create vocabulary
    app_to_idx = safe_load_pickle(data_paths["vocab_path"])
    if app_to_idx is None:
        app_to_idx = create_vocab(sequences)
        safe_save_pickle(app_to_idx, data_paths["vocab_path"])

    return sequences, app_to_idx


def create_datasets(sequences, app_to_idx, cfg, device):
    """Create and split datasets."""
    logger.info("Creating datasets and splitting into train/val sets")

    seq_length = len(sequences[0]["apps"])  # Get sequence length from data
    dataset = AppSequenceDataset(
        sequences,
        app_to_idx,
        sequence_length=seq_length,
        mask_prob=cfg.get("training", {}).get("mask_prob", 0.15),
    )

    df_sequences = pd.DataFrame(sequences)
    train_emp_ids, val_emp_ids = train_test_split(
        df_sequences["employeeId"].unique(),
        test_size=cfg.get("training", {}).get("test_size", 0.2),
        random_state=cfg.get("training", {}).get("random_seed", 42),
    )

    train_idx = df_sequences.merge(
        pd.Series(train_emp_ids, name="employeeId"), on="employeeId", how="inner"
    ).index.tolist()
    val_idx = df_sequences.merge(
        pd.Series(val_emp_ids, name="employeeId"), on="employeeId", how="inner"
    ).index.tolist()

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    if cfg.get("training", {}).get("preload_dataset", False):
        logger.info("Preloading dataset to device")
        train_dataset = PreloadedDataset(train_dataset, device)
        val_dataset = PreloadedDataset(val_dataset, device)

    return train_dataset, val_dataset, seq_length


def create_data_loaders(train_dataset, val_dataset, batch_size):
    """Create training and validation data loaders."""
    logger.info("Creating data loaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def train(cfg):
    """Main training function."""
    # Setup device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.get("training", {}).get("device") == "cuda"
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Setup data
    sequences, app_to_idx = setup_data(cfg)
    train_dataset, val_dataset, seq_length = create_datasets(
        sequences, app_to_idx, cfg, device
    )
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, cfg.get("model", {}).get("batch_size")
    )

    # Initialize model
    logger.info("Initializing model")
    if cfg.get("training", {}).get("pretrained_model_path", None):
        model, app_to_idx = load_model(
            cfg.get("training", {}).get("pretrained_model_path"),
            cfg,
            ShallowTransformerWithAttention,
            cfg,
            device,
        )
    else:
        model = ShallowTransformerWithAttention(
            vocab_size=len(app_to_idx) + 1,  # +1 for MASK
            d_model=cfg.get("model", {}).get("d_model", 64),
            nhead=cfg.get("model", {}).get("nhead", 4),
            seq_length=seq_length,
            n_layers=cfg.get("model", {}).get("num_encoder_layers", 3),
        ).to(device)

    # Setup training components
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(), float(cfg.get("training", {}).get("lr", 1e-4))
    )

    # Setup TensorBoard
    log_dir = Path("runs") / f"transformer_{seq_length}_{cfg['training']['lr']}"
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")

    # Training loop
    logger.info("Starting training loop")
    n_epochs = cfg.get("training", {}).get("num_epochs", 10)

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, accuracy = evaluate(model, val_loader, criterion, device)

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("Accuracy/validation", accuracy, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param.data, epoch)

        logger.info(f"Epoch {epoch+1}/{n_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")

    writer.close()

    # Save model if specified
    if cfg.get("training", {}).get("save_model_path", None):
        logger.info("Saving model checkpoint")
        model_path = Path(cfg.get("training", {}).get("save_model_path"))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "app_to_idx": app_to_idx,
            },
            model_path,
        )


if __name__ == "__main__":
    logger.info("Loading and validating configuration")
    cfg_path = Path("config/train/default.yaml")
    cfg = load_config(cfg_path)
    train(cfg)
