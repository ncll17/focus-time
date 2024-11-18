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
