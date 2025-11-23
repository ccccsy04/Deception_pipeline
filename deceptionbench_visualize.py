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

def collect_all_groups(exp_dir, control_dir, layer_idx):
    exp_mesa, exp_outer, _, _ = collect_hidden_states(exp_dir, layer_idx)
    ctrl_mesa, ctrl_outer, _, _ = collect_hidden_states(control_dir, layer_idx)
    return exp_mesa, exp_outer, ctrl_mesa, ctrl_outer

def plot_and_save(Xs, labels, colors, method, save_path):
    plt.figure(figsize=(8,6))
    for X, label, color in zip(Xs, labels, colors):
        plt.scatter(X[:,0], X[:,1], label=label, alpha=0.7, c=color)
    plt.legend()
    plt.title(f"{method} visualization of hidden states (exp/control)")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    for layer in args.layer:
        exp_mesa, exp_outer, ctrl_mesa, ctrl_outer = collect_all_groups(args.hidden_dir_exp, args.hidden_dir_control, layer)
        all_vecs = np.concatenate([exp_mesa, exp_outer, ctrl_mesa, ctrl_outer], axis=0)
        group_sizes = [len(exp_mesa), len(exp_outer), len(ctrl_mesa), len(ctrl_outer)]
        # t-SNE
        perplexity = max(2, min(30, len(all_vecs)//3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        all_tsne = tsne.fit_transform(all_vecs)
        idxs = np.cumsum([0]+group_sizes)
        plot_and_save([
            all_tsne[idxs[0]:idxs[1]],
            all_tsne[idxs[1]:idxs[2]],
            all_tsne[idxs[2]:idxs[3]],
            all_tsne[idxs[3]:idxs[4]]
        ], ["exp_mesa", "exp_outer", "ctrl_mesa", "ctrl_outer"], ["blue", "red", "green", "orange"], "t-SNE", os.path.join(args.save_dir, f"tsne_layer{layer}.png"))
        # PCA
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_vecs)
        plot_and_save([
            all_pca[idxs[0]:idxs[1]],
            all_pca[idxs[1]:idxs[2]],
            all_pca[idxs[2]:idxs[3]],
            all_pca[idxs[3]:idxs[4]]
        ], ["exp_mesa", "exp_outer", "ctrl_mesa", "ctrl_outer"], ["blue", "red", "green", "orange"], "PCA", os.path.join(args.save_dir, f"pca_layer{layer}.png"))
        print(f"Layer {layer} 's t-SNE and PCA plots saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dir_exp", type=str, required=True, help="deceptive cases hidden_state pt dir")
    parser.add_argument("--hidden_dir_control", type=str, required=True, help="non-deceptive cases hidden_state pt dir")
    parser.add_argument("--save_dir", type=str, required=True, help="pic saving dir")
    parser.add_argument("--layer", type=int, nargs='+', default=[-1], help="layers to be extracted")
    args = parser.parse_args()
    main(args)