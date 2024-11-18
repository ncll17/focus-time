from torch import nn
from models.transformer_models import ShallowTransformer_with_attn


# Load model with compatibility wrapper
class ModelCompatibilityWrapper(nn.Module):
    def __init__(self, old_state_dict, vocab_size, d_model=64, nhead=4, seq_length=64):
        super().__init__()
        self.model = ShallowTransformer_with_attn(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead, seq_length=seq_length
        )

        # Map state dict keys
        new_state_dict = {}
        for old_key, param in old_state_dict.items():
            if "transformer.layers." in old_key:
                # Map transformer layer keys
                new_key = old_key.replace("transformer.layers.", "encoder_layers.")
                new_state_dict[new_key] = param
            else:
                # Keep other keys the same
                new_state_dict[old_key] = param

        # Load state dict
        self.model.load_state_dict(new_state_dict)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
