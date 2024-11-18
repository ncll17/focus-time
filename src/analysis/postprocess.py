import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from data.datasets import AppSequenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def extract_app_embeddings(model, app_to_idx, device):
    """Extract embeddings for all apps in the vocabulary."""
    model.eval()
    app_indices = torch.tensor(list(app_to_idx.values()), device=device)

    with torch.no_grad():
        embeddings = model.app_embeddings(app_indices)
        embeddings = embeddings.cpu().numpy()

    app_embeddings = {
        app: embeddings[i]
        for app, i in app_to_idx.items()
        if app not in ["<PAD>", "<UNK>"]
    }

    return app_embeddings


def find_similar_apps(app_name, embeddings_df, n=5):
    """Find n most similar apps to the given app based on embedding similarity."""
    if app_name not in embeddings_df.index:
        return []

    similarities = cosine_similarity(
        embeddings_df.loc[app_name].values.reshape(1, -1), embeddings_df.values
    )[0]

    most_similar = pd.Series(similarities, index=embeddings_df.index).sort_values(
        ascending=False
    )[1 : n + 1]

    return most_similar


def visualize_embeddings(embeddings_df):
    """Create t-SNE visualization of app embeddings."""
    pca = PCA(n_components=min(50, len(embeddings_df)))
    embeddings_pca = pca.fit_transform(embeddings_df)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_pca)

    viz_df = pd.DataFrame(embeddings_2d, columns=["x", "y"], index=embeddings_df.index)

    plt.figure(figsize=(15, 10))
    plt.scatter(viz_df["x"], viz_df["y"], alpha=0.5)

    for idx, row in viz_df.iloc[::5].iterrows():
        plt.annotate(idx, (row["x"], row["y"]))

    plt.title("App Embeddings Visualization")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()

    return plt.gcf()


def predict_sequence(model, sequence, app_to_idx, device, top_k=3):
    """Make predictions for a single sequence.

    Args:
        model: Trained transformer model
        sequence: Dictionary containing sequence data
        app_to_idx: Vocabulary mapping
        device: torch device
        top_k: Number of top predictions to return

    Returns:
        List of dictionaries containing predictions for each position
    """
    model.eval()

    # Create dataset instance for the sequence
    dataset = AppSequenceDataset([sequence], app_to_idx)
    batch = dataset[0]

    # Move tensors to device
    app_ids = batch["app_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        # Get model outputs
        outputs = model(app_ids, attention_mask)

        # Calculate probabilities
        probs = torch.softmax(outputs.squeeze(), dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # Convert to numpy for easier handling
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

    # Create reverse mapping
    idx_to_app = {v: k for k, v in app_to_idx.items()}

    # Format results
    results = []
    actual_apps = sequence["apps"]

    for pos in range(len(actual_apps)):
        predictions = [
            {"app": idx_to_app[idx], "probability": float(prob)}
            for idx, prob in zip(top_indices[pos], top_probs[pos])
            if idx < len(idx_to_app)  # Filter out special tokens
        ]

        results.append(
            {
                "position": pos,
                "actual_app": actual_apps[pos],
                "predictions": predictions,
            }
        )

    return results


def batch_predict(model, sequences, app_to_idx, device, batch_size=32):
    """Make predictions for multiple sequences.

    Args:
        model: Trained transformer model
        sequences: List of sequence dictionaries
        app_to_idx: Vocabulary mapping
        device: torch device
        batch_size: Batch size for inference

    Returns:
        List of prediction results for each sequence
    """
    model.eval()

    # Create dataset and dataloader
    dataset = AppSequenceDataset(sequences, app_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Making predictions"):
            # Move batch to device
            app_ids = batch["app_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get predictions
            outputs = model(app_ids, attention_mask)
            probs = torch.softmax(outputs, dim=-1)

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=3, dim=-1)

            # Store predictions
            all_predictions.extend(zip(top_indices.cpu(), top_probs.cpu()))

    return all_predictions


def predict_sequence_with_attention(model, sequence, app_to_idx, device, top_k=3):
    """Make predictions and return attention weights for a single sequence.

    Args:
        model: Trained transformer model
        sequence: Dictionary containing sequence data
        app_to_idx: Vocabulary mapping
        device: torch device
        top_k: Number of top predictions to return

    Returns:
        Tuple containing:
        - List of prediction dictionaries
        - List of attention weights for each layer
    """
    model.eval()

    # Create dataset instance for the sequence
    dataset = AppSequenceDataset([sequence], app_to_idx)
    batch = dataset[0]

    # Debug prints
    print(f"Sequence length: {len(sequence['apps'])}")
    print(f"Attention mask sum: {batch['attention_mask'].sum()}")
    print(f"App IDs shape: {batch['app_ids'].shape}")

    # Move tensors to device
    app_ids = batch["app_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        # Get model outputs with attention weights
        outputs, attention_weights = model(
            app_ids, attention_mask, output_attentions=True
        )

        # Calculate probabilities
        probs = torch.softmax(outputs.squeeze(), dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        # Convert to numpy for easier handling
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Process attention weights
        processed_attention = [
            layer_attention.cpu().numpy() for layer_attention in attention_weights
        ]

    # Create reverse mapping
    idx_to_app = {v: k for k, v in app_to_idx.items()}

    # Format prediction results
    results = []
    actual_apps = sequence["apps"]

    for pos in range(len(actual_apps)):
        predictions = [
            {"app": idx_to_app[idx], "probability": float(prob)}
            for idx, prob in zip(top_indices[pos], top_probs[pos])
            if idx < len(idx_to_app)
        ]

        results.append(
            {
                "position": pos,
                "actual_app": actual_apps[pos],
                "predictions": predictions,
            }
        )

    return results, processed_attention


def analyze_attention_patterns(attention_weights, sequence, app_to_idx):
    """Analyze attention patterns from the model.

    Args:
        attention_weights: List of attention weights from each layer
        sequence: Original sequence data
        app_to_idx: Vocabulary mapping

    Returns:
        Dictionary containing attention analysis
    """
    idx_to_app = {v: k for k, v in app_to_idx.items()}
    apps = sequence["apps"]

    analysis = {
        "layer_wise_patterns": [],
        "average_attention": None,
        "most_attended_positions": [],
    }

    for layer_idx, layer_attention in enumerate(attention_weights):
        # Average across batch and heads
        avg_attention = layer_attention.mean(axis=(0, 1))

        # Find most attended positions for each position
        most_attended = []
        for pos in range(len(apps)):
            attention_scores = avg_attention[pos]
            top_attended_pos = np.argsort(attention_scores)[-3:][::-1]  # Top 3

            most_attended.append(
                {
                    "position": pos,
                    "current_app": apps[pos],
                    "attended_to": [
                        {
                            "position": int(att_pos),
                            "app": apps[att_pos],
                            "score": float(attention_scores[att_pos]),
                        }
                        for att_pos in top_attended_pos
                    ],
                }
            )

        analysis["layer_wise_patterns"].append(
            {"layer": layer_idx, "attention_patterns": most_attended}
        )

    # Calculate average attention across all layers
    all_layer_attention = np.mean(
        [layer.mean(axis=(0, 1)) for layer in attention_weights], axis=0
    )
    analysis["average_attention"] = all_layer_attention.tolist()

    # Find global most attended positions
    global_top_positions = np.argsort(np.mean(all_layer_attention, axis=0))[-5:][::-1]
    analysis["most_attended_positions"] = [
        {
            "position": int(pos),
            "app": apps[pos],
            "average_attention_score": float(np.mean(all_layer_attention[:, pos])),
        }
        for pos in global_top_positions
    ]

    return analysis


def visualize_attention(model, sequence, app_to_idx, device="cpu"):
    """Visualize attention patterns for a sequence."""
    model.eval()

    # Convert sequence to indices
    app_ids = [app_to_idx.get(app, app_to_idx["<UNK>"]) for app in sequence["apps"]]
    sequence_length = len(app_ids)
    attention_mask = [1] * sequence_length

    # Convert to tensors
    app_ids = torch.tensor(app_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    # Get predictions and attention weights
    with torch.no_grad():
        _, attention_weights = model(app_ids, attention_mask, output_attentions=True)

        # Debug print
        print(f"Number of attention layers: {len(attention_weights)}")
        if attention_weights:
            print(f"Shape of first layer attention: {attention_weights[0].shape}")

    # Format attention weights for BertViz
    formatted_weights = []
    for layer_attn in attention_weights:
        # Add batch dimension back if needed
        if layer_attn.dim() == 3:
            layer_attn = layer_attn.unsqueeze(0)
        # Ensure shape is [1, num_heads, seq_len, seq_len]
        formatted_weights.append(layer_attn)

    # Stack layers along a new dimension
    stacked_weights = torch.stack(formatted_weights)

    # Truncate sequence if needed
    tokens = sequence["apps"][:sequence_length]

    # Debug prints
    print(f"\nFormatted attention weights shape: {stacked_weights.shape}")
    print(f"Number of tokens: {len(tokens)}")

    return tokens, stacked_weights


# Try visualization


# Create a custom visualization function that works with our attention format
def plot_attention_heads(attention_weights, tokens):
    """Plot attention weights for each layer and head."""
    num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape

    # Create a figure with subplots for each layer and head
    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(20, 5 * num_layers), squeeze=False
    )

    # Plot each attention head
    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]

            # Get attention weights for this head
            attn = attention_weights[layer, 0, head].cpu().numpy()

            # Create heatmap
            im = ax.imshow(attn, cmap="viridis")

            # Add colorbar
            plt.colorbar(im, ax=ax)

            # Set title
            ax.set_title(f"Layer {layer+1}, Head {head+1}")

            # Set labels
            if head == 0:
                ax.set_ylabel("Query tokens")
            if layer == num_layers - 1:
                ax.set_xlabel("Key tokens")

            # Add token labels (show every nth token to avoid overcrowding)
            n = max(1, len(tokens) // 10)
            ax.set_xticks(range(0, len(tokens), n))
            ax.set_yticks(range(0, len(tokens), n))
            ax.set_xticklabels(tokens[::n], rotation=45, ha="right")
            ax.set_yticklabels(tokens[::n])

    plt.tight_layout()
    return fig
