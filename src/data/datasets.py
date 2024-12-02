import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import random

class AppSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        app_to_idx,
        sequence_length=64,
        mask_prob=0.15,
        extra_inputs=None,
    ):
        self.sequences = sequences
        self.app_to_idx = app_to_idx
        self.sequence_length = sequence_length
        self.mask_prob = mask_prob
        self.mask_token = len(app_to_idx)
        self.extra_inputs = extra_inputs if extra_inputs else {}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Convert apps to indices
        app_ids = [
            self.app_to_idx.get(app, self.app_to_idx["<UNK>"]) for app in seq["apps"]
        ][: self.sequence_length]

        # Basic masking logic
        masked_app_ids = app_ids.copy()
        labels = [-100] * len(app_ids)

        for i in range(len(app_ids)):
            if random.random() < self.mask_prob:
                labels[i] = app_ids[i]
                masked_app_ids[i] = self.mask_token

        # Handle padding for app_ids and labels
        if len(app_ids) < self.sequence_length:
            padding_length = self.sequence_length - len(app_ids)
            masked_app_ids = (
                masked_app_ids + [self.app_to_idx["<PAD>"]] * padding_length
            )
            labels = labels + [-100] * padding_length

        # Base output dictionary
        output = {
            "app_ids": torch.tensor(masked_app_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * len(app_ids) + [0] * (self.sequence_length - len(app_ids)),
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # Add extra inputs if specified in the configuration
        if self.extra_inputs.get("time", False):
            durations = seq["durations"][: self.sequence_length]
            if len(durations) < self.sequence_length:
                durations = durations + [0] * (self.sequence_length - len(durations))
            output["durations"] = torch.tensor(durations, dtype=torch.float)

        if self.extra_inputs.get("mouseClicks", False):
            mouse_clicks = seq["mouseClicks"][: self.sequence_length]
            if len(mouse_clicks) < self.sequence_length:
                mouse_clicks = mouse_clicks + [0] * (self.sequence_length - len(mouse_clicks))
            output["mouseClicks"] = torch.tensor(mouse_clicks, dtype=torch.float)

        if self.extra_inputs.get("mouseScroll", False):
            mouse_scroll = seq["mouseScroll"][: self.sequence_length]
            if len(mouse_scroll) < self.sequence_length:
                mouse_scroll = mouse_scroll + [0] * (self.sequence_length - len(mouse_scroll))
            output["mouseScroll"] = torch.tensor(mouse_scroll, dtype=torch.float)

        if self.extra_inputs.get("keystrokes", False):
            keystrokes = seq["keystrokes"][: self.sequence_length]
            if len(keystrokes) < self.sequence_length:
                keystrokes = keystrokes + [0] * (self.sequence_length - len(keystrokes))
            output["keystrokes"] = torch.tensor(keystrokes, dtype=torch.float)

        if self.extra_inputs.get("mic", False):
            mic = seq["mic"][: self.sequence_length]
            if len(mic) < self.sequence_length:
                mic = mic + [0] * (self.sequence_length - len(mic))
            output["mic"] = torch.tensor(mic, dtype=torch.bool)

        if self.extra_inputs.get("camera", False):
            camera = seq["camera"][: self.sequence_length]
            if len(camera) < self.sequence_length:
                camera = camera + [0] * (self.sequence_length - len(camera))
            output["camera"] = torch.tensor(camera, dtype=torch.bool)

        # Add app_quality if specified in the configuration
        if self.extra_inputs.get("app_quality", False):
            app_quality = seq["app_quality"][: self.sequence_length]
            if len(app_quality) < self.sequence_length:
                app_quality = app_quality + [5] * (self.sequence_length - len(app_quality))  # Default quality is 5
            output["app_quality"] = torch.tensor(app_quality, dtype=torch.float)

        return output

