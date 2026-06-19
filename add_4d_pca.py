import json
import os

def apply_4d_pca(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    updated = False
    
    # Update the markdown cell for Steps 3 & 4 to describe the 4D PCA
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if "Steps 3 & 4: CIELAB-PCA Projection" in src or "CIELAB-PCA Projection & Polarity" in src:
                new_md = [
                    "### Steps 3 & 4: 4D Color PCA Projection & Polarity Correction\n",
                    "\n",
                    "Rather than using simple, fixed color indexes, we use **Principal Component Analysis (PCA)** on a 4-dimensional color feature space:\n",
                    "1. **L\\* Channel**: Lightness (CIELAB)\n",
                    "2. **a\\* Channel**: Green-to-Red color variation (CIELAB)\n",
                    "3. **b\\* Channel**: Blue-to-Yellow color variation (CIELAB)\n",
                    "4. **Saturation Channel**: Chroma intensity (HSV)\n",
                    "\n",
                    "Stacking these four channels creates a 4D feature representation per pixel. Running PCA on this space finds the axis of maximum variation (**PC1**), which dynamically combines color, brightness, and saturation to isolate the leaves from the background (gray pipes, grids, and shadows).\n",
                    "\n",
                    "We also apply **polarity correction**: we calculate the correlation between the projected PC1 values and the original a* channel values. If they are positively correlated, we flip PC1 (multiply by -1) to guarantee that green vegetation (which has low values in a*) always corresponds to the highest values in the scaled PC1 image."
                ]
                cell["source"] = new_md
                print(f"Updated Markdown cell {i} in {path}")
                
    # Update the code cell for Steps 3 & 4
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "color_features = np.hstack" in src:
                new_source = [
                    "# Flatten L*, a*, b* and Saturation channels for 4D PCA\n",
                    "l_flat = l_channel.astype(np.float32).reshape(-1, 1)\n",
                    "a_flat = a_channel.astype(np.float32).reshape(-1, 1)\n",
                    "b_flat = b_channel.astype(np.float32).reshape(-1, 1)\n",
                    "s_flat = s_channel.astype(np.float32).reshape(-1, 1)\n",
                    "color_features = np.hstack((l_flat, a_flat, b_flat, s_flat))\n",
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
                    "# Calculate Explained Variance Ratio\n",
                    "total_variance = np.sum(eigenvalues)\n",
                    "var_ratios = (eigenvalues[:, 0] / total_variance) * 100\n",
                    "\n",
                    "# Visualize PC1, its histogram, 2D PCA comparison, and 4D Explained Variance\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(18, 16))\n",
                    "\n",
                    "# 1. PC1 Grayscale Image\n",
                    "axes[0, 0].imshow(pc1_scaled, cmap='gray')\n",
                    "axes[0, 0].set_title(\"4D Polarity-Corrected PC1 Channel (L*, a*, b*, S)\")\n",
                    "axes[0, 0].axis('off')\n",
                    "\n",
                    "# 2. Histogram of PC1\n",
                    "axes[0, 1].hist(pc1_scaled.ravel(), bins=256, color='gray', range=[0, 256])\n",
                    "axes[0, 1].set_title(\"Histogram of 4D PC1 Values\")\n",
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
                    "# 4. Explained Variance Scree Plot for all 4 components\n",
                    "pc_labels = [f'PC{i+1}' for i in range(len(var_ratios))]\n",
                    "bars = axes[1, 1].bar(pc_labels, var_ratios, color=['red', 'blue', 'orange', 'purple'], alpha=0.7)\n",
                    "axes[1, 1].set_ylabel('Percentage of Explained Variance (%)')\n",
                    "axes[1, 1].set_title('4D PCA Scree Plot (Information Captured per Component)')\n",
                    "axes[1, 1].set_ylim(0, 110)\n",
                    "for bar in bars:\n",
                    "    yval = bar.get_height()\n",
                    "    axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f\"{yval:.2f}%\", ha='center', va='bottom', fontweight='bold')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
                cell["source"] = new_source
                updated = True
                print(f"Updated Code cell {i} in {path}")
                
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Saved updates to {path}")
    else:
        print(f"No updates required for {path}")

def main():
    apply_4d_pca("leaf_segmentation_tutorial_clean.ipynb")
    apply_4d_pca("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
