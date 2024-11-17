import torch
from tqdm import tqdm
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, criterion, device, output_attentions=False):
    """Train model for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        app_ids = batch["app_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(app_ids, attention_mask, output_attentions=output_attentions)
        if output_attentions:
            logits = outputs[0]
        else:
            logits = outputs
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, output_attentions=False):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            app_ids = batch["app_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(app_ids, attention_mask, output_attentions)
            if output_attentions:
                attentions = outputs[1]
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

            total_loss += loss.item()

    return total_loss / len(loader), correct_predictions / total_predictions
