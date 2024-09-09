from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import os

def plot_tsne(activations, labels, save_dir, file_name, perplexity=30, lr=1000.0, iters=1000,):
    tsne = TSNE(n_components=2, learning_rate=lr, n_iter=iters, perplexity=perplexity, random_state=0)
    tsne_results = tsne.fit_transform(activations)

    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = {label: idx for idx, label in enumerate(unique_labels)}
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colors[label] for label in labels], cmap='viridis', alpha=0.5)
    
    # Create custom legend
    legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(colors[label])), label=label) for label in unique_labels]
    plt.legend(handles=legend_handles, title="Datasets")
    
    plt.title('t-SNE of Model Activations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # check dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save plot
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    plt.close()  # Close figure to free memory

    print(f"t-SNE plot saved to {file_path}")

