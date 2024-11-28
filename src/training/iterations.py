import torch
from tqdm import tqdm
import torch.nn.functional as F


def train_epoch(
    model, dataloader, optimizer, criterion, device, output_attentions=False
):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader, desc="Training"):
        app_ids = batch["app_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Check if the model expects durations (time-aware model)
        if hasattr(model, "duration_projection"):
            durations = batch["durations"].to(device)
            outputs = model(
                app_ids, durations, attention_mask, output_attentions=output_attentions
            )
        else:
            outputs = model(
                app_ids, attention_mask, output_attentions=output_attentions
            )

        if output_attentions:
            outputs, attention_weights = outputs

        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != -100
        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
        total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, output_attentions=False):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            app_ids = batch["app_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Check if the model expects durations (time-aware model)
            if hasattr(model, "duration_projection"):
                durations = batch["durations"].to(device)
                outputs = model(
                    app_ids,
                    durations,
                    attention_mask,
                    output_attentions=output_attentions,
                )
            else:
                outputs = model(
                    app_ids, attention_mask, output_attentions=output_attentions
                )

            if output_attentions:
                outputs, attention_weights = outputs

            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            mask = labels != -100
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy
