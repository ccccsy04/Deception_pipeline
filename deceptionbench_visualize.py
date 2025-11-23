import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def collect_hidden_states(hidden_dir, layer_idx):
    mesa_vecs, outer_vecs, mesa_ids, outer_ids = [], [], [], []
    files = os.listdir(hidden_dir)
    mesa_files = [f for f in files if f.endswith('_mesa.pt')]
    outer_files = [f for f in files if f.endswith('_outer.pt')]
    mesa_ids_set = set(f.split('_')[0] for f in mesa_files)
    outer_ids_set = set(f.split('_')[0] for f in outer_files)
    common_ids = sorted(mesa_ids_set & outer_ids_set)
    for idx in common_ids:
        mesa_path = os.path.join(hidden_dir, f"{idx}_mesa.pt")
        outer_path = os.path.join(hidden_dir, f"{idx}_outer.pt")
        mesa_h = torch.load(mesa_path)
        outer_h = torch.load(outer_path)
        mesa_vecs.append(mesa_h[layer_idx].to(torch.float32).numpy())
        outer_vecs.append(outer_h[layer_idx].to(torch.float32).numpy())
        mesa_ids.append(idx)
        outer_ids.append(idx)
    return np.array(mesa_vecs), np.array(outer_vecs), mesa_ids, outer_ids

def plot_and_save(X, Y, method, save_path):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c='blue', label='mesa', alpha=0.7)
    plt.scatter(Y[:,0], Y[:,1], c='red', label='outer', alpha=0.7)
    plt.legend()
    plt.title(f"{method} visualization of hidden states")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    for layer in args.layer:
        mesa_vecs, outer_vecs, mesa_ids, outer_ids = collect_hidden_states(args.hidden_dir, layer)
        all_vecs = np.concatenate([mesa_vecs, outer_vecs], axis=0)
        labels = np.array([0]*len(mesa_vecs) + [1]*len(outer_vecs))

        # t-SNE
        perplexity = max(2, min(30, len(all_vecs)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        all_tsne = tsne.fit_transform(all_vecs)
        plot_and_save(all_tsne[labels==0], all_tsne[labels==1], "t-SNE", os.path.join(args.save_dir, f"tsne_layer{layer}.png"))

        # PCA
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_vecs)
        plot_and_save(all_pca[labels==0], all_pca[labels==1], "PCA", os.path.join(args.save_dir, f"pca_layer{layer}.png"))

        print(f"Layer {layer} 's t-SNE and PCA plots saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir", type=str, required=True, help="hidden_state pt dir")
    parser.add_argument("--save_dir", type=str, required=True, help="pic saving dir")
    parser.add_argument("--layer", type=int, nargs='+', default=[-1], help="layers to be extracted")
    args = parser.parse_args()
    main(args)