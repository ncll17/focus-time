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
        # Move input tensors to the appropriate device (GPU/CPU)
        app_ids = batch["app_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Collect all extra features if they exist
        extra_inputs = {}
        for feature_name in [
            "durations", "mouseClicks", "mouseScroll", "keystrokes", "mic", "camera", "app_quality"
        ]:
            if feature_name in batch:
                extra_inputs[feature_name] = batch[feature_name].to(device)

        optimizer.zero_grad()

        # Forward pass: passing extra inputs dynamically based on model configuration
        outputs = model(
            app_ids,
            attention_mask,
            **extra_inputs,
            output_attentions=output_attentions
        )

        if output_attentions:
            outputs, attention_weights = outputs

        # Calculate loss
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = labels != -100  # Use mask to ignore padding tokens during accuracy calculation
        correct_predictions += (predictions[mask] == labels[mask]).sum().item()
        total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, accuracy



def evaluate(model, dataloader, criterion, device, output_attentions=False):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move input tensors to the appropriate device (GPU/CPU)
            app_ids = batch["app_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Collect all extra features if they exist
            extra_inputs = {}
            for feature_name in [
                "durations", "mouseClicks", "mouseScroll", "keystrokes", "mic", "camera", "app_quality"
            ]:
                if feature_name in batch:
                    extra_inputs[feature_name] = batch[feature_name].to(device)

            # Forward pass: passing extra inputs dynamically based on model configuration
            outputs = model(
                app_ids,
                attention_mask,
                **extra_inputs,
                output_attentions=output_attentions
            )

            if output_attentions:
                outputs, attention_weights = outputs

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            mask = labels != -100  # Use mask to ignore padding tokens during accuracy calculation
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, accuracy

