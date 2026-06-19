import json
import os

def add_useful_graphs(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    updated = False
    
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "color_features = np.hstack" in source:
                new_source = [
                    "# Flatten a* and b* channels for PCA\n",
                    "a_flat = a_channel.astype(np.float32).reshape(-1, 1)\n",
                    "b_flat = b_channel.astype(np.float32).reshape(-1, 1)\n",
                    "color_features = np.hstack((a_flat, b_flat))\n",
                    "\n",
                    "# Compute PCA using OpenCV's PCACompute2\n",
                    "mean, eigenvectors, eigenvalues = cv2.PCACompute2(color_features, mean=None)\n",
                    "\n",
                    "# Project onto the first principal component (PC1)\n",
                    "projected = cv2.PCAProject(color_features, mean, eigenvectors)\n",
                    "pc1 = projected[:, 0].reshape(a_channel.shape)\n",
                    "\n",
                    "# Polarity correction: ensure green (low values in a*) corresponds to high values in PC1\n",
                    "correlation = np.corrcoef(pc1.ravel(), a_channel.ravel())[0, 1]\n",
                    "if correlation > 0:\n",
                    "    pc1 = -pc1\n",
                    "\n",
                    "# Scale to 0-255 range\n",
                    "pc1_min = pc1.min()\n",
                    "pc1_max = pc1.max()\n",
                    "pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))\n",
                    "\n",
                    "# Visualize PC1, its histogram, and the 2D Color Space PCA comparison\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(18, 16))\n",
                    "\n",
                    "# 1. PC1 Grayscale Image\n",
                    "axes[0, 0].imshow(pc1_scaled, cmap='gray')\n",
                    "axes[0, 0].set_title(\"Polarity-Corrected PC1 Channel\")\n",
                    "axes[0, 0].axis('off')\n",
                    "\n",
                    "# 2. Histogram of PC1\n",
                    "axes[0, 1].hist(pc1_scaled.ravel(), bins=256, color='gray', range=[0, 256])\n",
                    "axes[0, 1].set_title(\"Histogram of PC1 Values (Bimodal Distribution)\")\n",
                    "axes[0, 1].set_xlabel(\"Pixel Intensity (PC1)\")\n",
                    "axes[0, 1].set_ylabel(\"Count\")\n",
                    "\n",
                    "# 3. 2D Color Scatter Plot (Without PCA - True Colors)\n",
                    "sample_rate = 200  # Sample pixels for clean and fast rendering\n",
                    "a_sample = a_channel.ravel()[::sample_rate]\n",
                    "b_sample = b_channel.ravel()[::sample_rate]\n",
                    "rgb_sample = img_rgb.reshape(-1, 3)[::sample_rate] / 255.0  # Normalize to [0,1]\n",
                    "\n",
                    "axes[1, 0].scatter(a_sample, b_sample, c=rgb_sample, alpha=0.6, edgecolors='none')\n",
                    "axes[1, 0].set_title(\"2D Color Scatter Plot (Without PCA - True Colors)\")\n",
                    "axes[1, 0].set_xlabel(\"a* Channel (Green to Red)\")\n",
                    "axes[1, 0].set_ylabel(\"b* Channel (Blue to Yellow)\")\n",
                    "axes[1, 0].axvline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[1, 0].axhline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[1, 0].grid(True, linestyle=':', alpha=0.5)\n",
                    "\n",
                    "# 4. 2D Color Scatter Plot (With PCA - Color-Mapped by PC1 Projection)\n",
                    "pc1_sample = pc1_scaled.ravel()[::sample_rate]\n",
                    "scatter = axes[1, 1].scatter(a_sample, b_sample, c=pc1_sample, cmap='viridis', alpha=0.4, edgecolors='none')\n",
                    "fig.colorbar(scatter, ax=axes[1, 1], label='Projected PC1 Intensity')\n",
                    "\n",
                    "mean_a, mean_b = mean[0, 0], mean[0, 1]\n",
                    "axes[1, 1].scatter(mean_a, mean_b, color='red', s=120, zorder=5, label='Mean Color Centroid')\n",
                    "\n",
                    "scale_pc1 = 2.5 * np.sqrt(eigenvalues[0, 0])\n",
                    "scale_pc2 = 2.5 * np.sqrt(eigenvalues[1, 0])\n",
                    "\n",
                    "# Draw arrows for PC1 and PC2\n",
                    "axes[1, 1].arrow(mean_a, mean_b, eigenvectors[0, 0] * scale_pc1, eigenvectors[0, 1] * scale_pc1,\n",
                    "                 color='red', width=1.5, head_width=5, label='PC1 Direction', zorder=6)\n",
                    "axes[1, 1].arrow(mean_a, mean_b, eigenvectors[1, 0] * scale_pc2, eigenvectors[1, 1] * scale_pc2,\n",
                    "                 color='blue', width=1.5, head_width=5, label='PC2 Direction', zorder=6)\n",
                    "\n",
                    "axes[1, 1].set_title(\"2D Color Scatter Plot (With PCA - Projected)\")\n",
                    "axes[1, 1].set_xlabel(\"a* Channel (Green to Red)\")\n",
                    "axes[1, 1].set_ylabel(\"b* Channel (Blue to Yellow)\")\n",
                    "axes[1, 1].axvline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[1, 1].axhline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[1, 1].grid(True, linestyle=':', alpha=0.5)\n",
                    "axes[1, 1].legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
                cell["source"] = new_source
                updated = True
                print(f"Updated cell {i} in {path}")
                
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Saved updates to {path}")
    else:
        print(f"No updates required for {path}")

def main():
    add_useful_graphs("leaf_segmentation_tutorial_clean.ipynb")
    add_useful_graphs("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
