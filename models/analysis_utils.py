from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import os

def plot_tsne(activations, labels, save_dir, file_name):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(activations.numpy())
    
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = {label: idx for idx, label in enumerate(unique_labels)}
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[colors[label] for label in labels], cmap='viridis')
    
    # # Create a colorbar with the unique class names
    # cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    # cbar.ax.set_yticklabels(unique_labels)

    # Create custom legend
    legend_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(colors[label])), label=label) for label in unique_labels]
    plt.legend(handles=legend_handles, title="Datasets")
    
    plt.title('t-SNE of Model Activations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory

    print(f"t-SNE plot saved to {file_path}")

    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    # plt.colorbar(scatter)
    # plt.title('t-SNE of Model Activations')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.show()