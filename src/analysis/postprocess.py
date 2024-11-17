import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
