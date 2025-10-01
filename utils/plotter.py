import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os

def plot_all(hist, save_folder="assets/plots", show_plots=False):
    if not hist:
        print("No history to plot.")
        return

    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    steps = [h["step"] for h in hist]
    mimic_scores = [h["mimic"] for h in hist]
    prob_scores = [h["prob"] for h in hist]
    combined_scores = [h["combined"] for h in hist]
    alignments = [h.get("alignment", 0.0) for h in hist]
    thetas = np.array([h["theta"] for h in hist])  # shape: (steps, dims)

    # --- 2D: Leakage scores ---
    plt.figure()
    plt.plot(steps, mimic_scores, label="Mimic Leakage")
    plt.plot(steps, prob_scores, label="Prob Leakage")
    plt.plot(steps, combined_scores, label="Combined Leakage")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Leakage Scores vs Steps")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "leakage_scores.png"))
    if show_plots:
        plt.show()
    plt.close()

    # --- 2D: Alignment ---
    plt.figure()
    plt.plot(steps, alignments, marker="o", label="Alignment")
    plt.xlabel("Step")
    plt.ylabel("Alignment Score")
    plt.title("Alignment vs Steps")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "alignment.png"))
    if show_plots:
        plt.show()
    plt.close()

    # --- 2D: Theta evolution ---
    plt.figure()
    for dim in range(thetas.shape[1]):
        plt.plot(steps, thetas[:, dim], label=f"Theta[{dim}]")
    plt.xlabel("Step")
    plt.ylabel("Theta Value")
    plt.title("Theta Evolution")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "theta_evolution.png"))
    if show_plots:
        plt.show()
    plt.close()

    # --- 3D: Leakage vs Alignment vs Step ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(steps, combined_scores, alignments, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Combined Leakage")
    ax.set_zlabel("Alignment")
    plt.title("Leakage vs Alignment vs Step")
    plt.savefig(os.path.join(save_folder, "leakage_vs_alignment_3d.png"))
    if show_plots:
        plt.show()
    plt.close()

    # --- 3D: Theta trajectory (first 3 dims or PCA) ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if thetas.shape[1] >= 3:
        ax.plot(thetas[:, 0], thetas[:, 1], thetas[:, 2], marker="o")
        ax.set_xlabel("Theta[0]")
        ax.set_ylabel("Theta[1]")
        ax.set_zlabel("Theta[2]")
        plt.title("Theta Trajectory (first 3 dims)")
    else:
        # If less than 3 dims, project with PCA
        pca = PCA(n_components=3)
        thetas_3d = pca.fit_transform(thetas)
        ax.plot(thetas_3d[:, 0], thetas_3d[:, 1], thetas_3d[:, 2], marker="o")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.title("Theta Trajectory (PCA projection)")
    plt.savefig(os.path.join(save_folder, "theta_trajectory_3d.png"))
    if show_plots:
        plt.show()
    plt.close()

    print(f"All plots saved to folder: {save_folder}")
