import torch
from torch import nn
import math


class ShallowTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, seq_length=64):
        super().__init__()

        self.app_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Parameter(torch.randn(1, seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.app_predictor = nn.Linear(d_model, vocab_size)

    def forward(self, app_ids, attention_mask):
        x = self.app_embeddings(app_ids)
        x = x + self.pos_embeddings[:, : x.size(1), :]
        attention_mask = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        return self.app_predictor(x)


class ShallowTransformerWithAttention(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, seq_length=64, n_layers=3):
        super().__init__()
        self.app_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Parameter(torch.randn(1, seq_length, d_model))
        self.nhead = nhead

        # Create custom attention layers to capture weights
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=128,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.app_predictor = nn.Linear(d_model, vocab_size)

    def forward(self, app_ids, attention_mask, output_attentions=False):
        # Get app embeddings
        x = self.app_embeddings(app_ids)  # [batch_size, seq_len, d_model]

        # Add positional embeddings
        x = x + self.pos_embeddings[:, : x.size(1), :]

        # Create attention mask for transformer
        attention_mask = attention_mask == 0

        attention_weights = []

        # Pass through transformer layers
        for layer in self.encoder_layers:
            if output_attentions:
                # Multi-head attention computation
                batch_size, seq_len, d_model = x.shape
                num_heads = self.nhead  # Use stored number of heads
                head_dim = d_model // num_heads

                # Reshape input for attention computation
                q = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                k = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

                # Compute attention scores
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

                # Reshape attention mask
                if attention_mask is not None:
                    # Expand mask for multiple heads
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    expanded_mask = expanded_mask.expand(
                        batch_size, num_heads, seq_len, seq_len
                    )
                    scores = scores.masked_fill(expanded_mask, float("-inf"))

                # Apply softmax
                attn_weights = torch.softmax(scores, dim=-1)
                attention_weights.append(attn_weights)

            # Forward pass through layer
            x = layer(x, src_key_padding_mask=attention_mask)

        # Get logits
        logits = self.app_predictor(x)

        if output_attentions:
            return logits, attention_weights
        return logits

class ShallowTransformerTimeWithAttention(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, seq_length=64, n_layers=3, cfg=None):
        super().__init__()
        
        # Configuration for extra inputs
        self.cfg = cfg if cfg else {}
        self.extra_inputs = self.cfg.get("extra_inputs", {})

        # Embedding layers for app IDs and other features
        self.app_embeddings = nn.Embedding(vocab_size, d_model)
        self.duration_projection = nn.Linear(1, d_model)

        # Additional feature projections based on cfg
        self.mouse_clicks_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("mouseClicks", False) else None
        )
        self.mouse_scroll_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("mouseScroll", False) else None
        )
        self.keystrokes_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("keystrokes", False) else None
        )
        self.mic_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("mic", False) else None
        )
        self.camera_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("camera", False) else None
        )
        self.app_quality_projection = (
            nn.Linear(1, d_model) if self.extra_inputs.get("app_quality", False) else None
        )

        # Calculating the size of the final concatenated embedding
        feature_count = 2  # App Embeddings + Duration Embedding
        if self.mouse_clicks_projection:
            feature_count += 1
        if self.mouse_scroll_projection:
            feature_count += 1
        if self.keystrokes_projection:
            feature_count += 1
        if self.mic_projection:
            feature_count += 1
        if self.camera_projection:
            feature_count += 1
        if self.app_quality_projection:
            feature_count += 1

        self.combined_d_model = feature_count * d_model

        # Positional embeddings of appropriate size
        self.pos_embeddings = nn.Parameter(torch.randn(1, seq_length, self.combined_d_model))
        self.nhead = nhead

        # Transformer layers with updated d_model size
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.combined_d_model,  # Updated for concatenated features
                    nhead=nhead,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        # Linear layer to project back to vocabulary size from combined embedding size
        self.app_predictor = nn.Linear(self.combined_d_model, vocab_size)

    def forward(self, app_ids, durations, attention_mask, **kwargs):
        # Get app embeddings
        x_apps = self.app_embeddings(app_ids)  # [batch_size, seq_len, d_model]

        # Project durations to d_model size
        x_durations = self.duration_projection(
            durations.unsqueeze(-1)
        )  # [batch_size, seq_len, d_model]

        # Collect all features to concatenate
        features = [x_apps, x_durations]

        # Add mouseClicks projection if specified in cfg
        if self.extra_inputs.get("mouseClicks", False):
            x_mouse_clicks = self.mouse_clicks_projection(
                kwargs["mouseClicks"].unsqueeze(-1)
            )  # [batch_size, seq_len, d_model]
            features.append(x_mouse_clicks)

        # Add mouseScroll projection if specified in cfg
        if self.extra_inputs.get("mouseScroll", False):
            x_mouse_scroll = self.mouse_scroll_projection(
                kwargs["mouseScroll"].unsqueeze(-1)
            )  # [batch_size, seq_len, d_model]
            features.append(x_mouse_scroll)

        # Add keystrokes projection if specified in cfg
        if self.extra_inputs.get("keystrokes", False):
            x_keystrokes = self.keystrokes_projection(
                kwargs["keystrokes"].unsqueeze(-1)
            )  # [batch_size, seq_len, d_model]
            features.append(x_keystrokes)

        # Add mic projection if specified in cfg
        if self.extra_inputs.get("mic", False):
            x_mic = self.mic_projection(kwargs["mic"].unsqueeze(-1))  # [batch_size, seq_len, d_model]
            features.append(x_mic)

        # Add camera projection if specified in cfg
        if self.extra_inputs.get("camera", False):
            x_camera = self.camera_projection(kwargs["camera"].unsqueeze(-1))  # [batch_size, seq_len, d_model]
            features.append(x_camera)

        # Add app_quality projection if specified in cfg
        if self.extra_inputs.get("app_quality", False):
            x_app_quality = self.app_quality_projection(kwargs["app_quality"].unsqueeze(-1))  # [batch_size, seq_len, d_model]
            features.append(x_app_quality)

        # Concatenate along the feature dimension
        x = torch.cat(features, dim=-1)  # [batch_size, seq_len, combined_d_model]

        # Add positional embeddings
        x = x + self.pos_embeddings[:, :x.size(1), :]  # Broadcasting for position encoding

        # Create attention mask for transformer
        attention_mask = attention_mask == 0

        attention_weights = []

        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=attention_mask)

        # Get logits for predicting app IDs
        logits = self.app_predictor(x)

        return logits

