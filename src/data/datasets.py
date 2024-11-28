import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm


class AppSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        app_to_idx,
        sequence_length=64,
        mask_prob=0.15,
        include_time=False,
    ):
        self.sequences = sequences
        self.app_to_idx = app_to_idx
        self.sequence_length = sequence_length
        self.mask_prob = mask_prob
        self.mask_token = len(app_to_idx)
        self.include_time = include_time

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        app_ids = [
            self.app_to_idx.get(app, self.app_to_idx["<UNK>"]) for app in seq["apps"]
        ][: self.sequence_length]

        # Basic masking logic remains the same
        masked_app_ids = app_ids.copy()
        labels = [-100] * len(app_ids)

        for i in range(len(app_ids)):
            if random.random() < self.mask_prob:
                labels[i] = app_ids[i]
                masked_app_ids[i] = self.mask_token

        # Handle padding
        if len(app_ids) < self.sequence_length:
            padding_length = self.sequence_length - len(app_ids)
            masked_app_ids = (
                masked_app_ids + [self.app_to_idx["<PAD>"]] * padding_length
            )
            labels = labels + [-100] * padding_length

        # Base output dictionary
        output = {
            "app_ids": torch.tensor(masked_app_ids),
            "attention_mask": torch.tensor(
                [1] * len(app_ids) + [0] * (self.sequence_length - len(app_ids))
            ),
            "labels": torch.tensor(labels),
        }

        # Add time-related features if requested
        if self.include_time:
            durations = seq["durations"][: self.sequence_length]
            if len(durations) < self.sequence_length:
                durations = durations + [0] * (self.sequence_length - len(durations))
            output["durations"] = torch.tensor(durations)

        return output


class PreloadedDataset(Dataset):
    def __init__(self, original_dataset, device):
        self.data = []
        for i in tqdm(range(len(original_dataset))):
            batch = original_dataset[i]
            self.data.append({k: v.to(device) for k, v in batch.items()})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
