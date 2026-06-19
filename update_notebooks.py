import json
import os

def update_notebook(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    updated_color_pca = False
    updated_spatial_pca = False
    
    # 1. Update Color PCA Cell
    # We look for a cell containing "a_flat = a_channel.astype"
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "a_flat = a_channel.astype" in source and "# Visualize PC1, its histogram, and the 2D Color Space PCA" not in source:
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
                    "# Visualize PC1, its histogram, and the 2D Color Space PCA\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(22, 7))\n",
                    "\n",
                    "# 1. PC1 Grayscale Image\n",
                    "axes[0].imshow(pc1_scaled, cmap='gray')\n",
                    "axes[0].set_title(\"Polarity-Corrected PC1 Channel\")\n",
                    "axes[0].axis('off')\n",
                    "\n",
                    "# 2. Histogram of PC1\n",
                    "axes[1].hist(pc1_scaled.ravel(), bins=256, color='gray', range=[0, 256])\n",
                    "axes[1].set_title(\"Histogram of PC1 Values\")\n",
                    "axes[1].set_xlabel(\"Pixel Intensity\")\n",
                    "axes[1].set_ylabel(\"Count\")\n",
                    "\n",
                    "# 3. 2D PCA Scatter Plot (a* vs b*)\n",
                    "sample_rate = 200  # Sample pixels for clean and fast rendering\n",
                    "a_sample = a_channel.ravel()[::sample_rate]\n",
                    "b_sample = b_channel.ravel()[::sample_rate]\n",
                    "axes[2].scatter(a_sample, b_sample, alpha=0.15, c='forestgreen', edgecolors='none', label='Pixels (Sampled)')\n",
                    "\n",
                    "mean_a, mean_b = mean[0, 0], mean[0, 1]\n",
                    "axes[2].scatter(mean_a, mean_b, color='red', s=120, zorder=5, label='Mean Color Centroid')\n",
                    "\n",
                    "scale_pc1 = 2.5 * np.sqrt(eigenvalues[0, 0])\n",
                    "scale_pc2 = 2.5 * np.sqrt(eigenvalues[1, 0])\n",
                    "\n",
                    "# Draw arrows for PC1 and PC2\n",
                    "axes[2].arrow(mean_a, mean_b, eigenvectors[0, 0] * scale_pc1, eigenvectors[0, 1] * scale_pc1,\n",
                    "              color='red', width=1.5, head_width=5, label='PC1 (Primary Variation)', zorder=6)\n",
                    "axes[2].arrow(mean_a, mean_b, eigenvectors[1, 0] * scale_pc2, eigenvectors[1, 1] * scale_pc2,\n",
                    "              color='blue', width=1.5, head_width=5, label='PC2', zorder=6)\n",
                    "\n",
                    "axes[2].set_title(\"PCA of CIELAB Color Space (a* vs b*)\")\n",
                    "axes[2].set_xlabel(\"a* Channel (Green to Red)\")\n",
                    "axes[2].set_ylabel(\"b* Channel (Blue to Yellow)\")\n",
                    "axes[2].axvline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[2].axhline(128, color='gray', linestyle='--', alpha=0.4)\n",
                    "axes[2].grid(True, linestyle=':', alpha=0.5)\n",
                    "axes[2].legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
                cell["source"] = new_source
                updated_color_pca = True
                print(f"Updated Color PCA cell (Index {i}) in {path}")
                
    # 2. Update Spatial PCA Cell
    # We look for a cell containing "def analyze_plant_shape(contour):"
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "def analyze_plant_shape(contour):" in source and "Spatial PCA of Plant Shapes (3x4 Grid Layout)" not in source:
                new_source = [
                    "def analyze_plant_shape(contour):\n",
                    "    # Extract coordinates of all pixels inside the contour\n",
                    "    mask = np.zeros(cleaned.shape, dtype=np.uint8)\n",
                    "    cv2.drawContours(mask, [contour], -1, 255, -1)\n",
                    "    pts = np.column_stack(np.where(mask == 255))\n",
                    "    pts = pts[:, [1, 0]].astype(np.float32) # Swap to x, y format\n",
                    "    \n",
                    "    if len(pts) < 5:\n",
                    "        return None, None, None, None, None\n",
                    "    \n",
                    "    # Run 2D PCA on coordinates\n",
                    "    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)\n",
                    "    \n",
                    "    center = (mean[0, 0], mean[0, 1])\n",
                    "    sd1 = np.sqrt(max(0, eigenvalues[0, 0]))\n",
                    "    sd2 = np.sqrt(max(0, eigenvalues[1, 0]))\n",
                    "    \n",
                    "    length = 2.0 * sd1\n",
                    "    width = 2.0 * sd2\n",
                    "    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180.0 / np.pi\n",
                    "    \n",
                    "    return center, eigenvectors, length, width, angle\n",
                    "\n",
                    "# Draw the PCA axes on the original image\n",
                    "img_pca = img_rgb.copy()\n",
                    "\n",
                    "print(\"Plant ID | Centroid  | Length (px) | Width (px) | Area (px) | Orientation Angle\")\n",
                    "print(\"--------------------------------------------------------------------------------\")\n",
                    "for record in all_sorted_plants:\n",
                    "    cnt = record['contour']\n",
                    "    center, eigenvectors, length, width, angle = analyze_plant_shape(cnt)\n",
                    "    \n",
                    "    if center is None:\n",
                    "        continue\n",
                    "        \n",
                    "    pid = record['id']\n",
                    "    area = record['area']\n",
                    "    print(f\"ID: {pid:<4} | ({int(center[0]):>3}, {int(center[1]):>3}) | {length:>11.1f} | {width:>10.1f} | {area:>9.1f} | {angle:>6.1f}\u00b0\")\n",
                    "    \n",
                    "    cx, cy = int(center[0]), int(center[1])\n",
                    "    # Major axis endpoints (+/- 2 std dev)\n",
                    "    p1_major = (int(cx + eigenvectors[0, 0] * length), int(cy + eigenvectors[0, 1] * length))\n",
                    "    p2_major = (int(cx - eigenvectors[0, 0] * length), int(cy - eigenvectors[0, 1] * length))\n",
                    "    # Minor axis endpoints (+/- 2 std dev)\n",
                    "    p1_minor = (int(cx + eigenvectors[1, 0] * width), int(cy + eigenvectors[1, 1] * width))\n",
                    "    p2_minor = (int(cx - eigenvectors[1, 0] * width), int(cy - eigenvectors[1, 1] * width))\n",
                    "    \n",
                    "    # Draw center\n",
                    "    cv2.circle(img_pca, (cx, cy), 5, (255, 0, 0), -1)\n",
                    "    # Draw axes (Red = Major, Blue = Minor)\n",
                    "    cv2.line(img_pca, p1_major, p2_major, (255, 0, 0), 2)\n",
                    "    cv2.line(img_pca, p1_minor, p2_minor, (0, 0, 255), 2)\n",
                    "    # Label plant ID\n",
                    "    cv2.putText(img_pca, f\"ID: {pid}\", (cx - 30, cy - 20), \n",
                    "                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)\n",
                    "\n",
                    "plt.figure(figsize=(12, 10))\n",
                    "plt.imshow(img_pca)\n",
                    "plt.title(\"Hydroponic Grid Growth Analysis with PCA Axes (Red = Major, Blue = Minor)\")\n",
                    "plt.axis('off')\n",
                    "plt.show()\n",
                    "\n",
                    "# --- Plot Spatial PCA coordinate graphs for each plant in a 3x4 grid ---\n",
                    "fig, axes = plt.subplots(3, 4, figsize=(16, 12))\n",
                    "\n",
                    "# Initialize grid with placeholder text\n",
                    "for r in range(3):\n",
                    "    for c in range(4):\n",
                    "        axes[r, c].text(0.5, 0.5, \"No Plant\", ha='center', va='center', color='gray')\n",
                    "        axes[r, c].axis('off')\n",
                    "\n",
                    "for record in all_sorted_plants:\n",
                    "    cnt = record['contour']\n",
                    "    center, eigenvectors, length, width, angle = analyze_plant_shape(cnt)\n",
                    "    if center is None:\n",
                    "        continue\n",
                    "        \n",
                    "    r = record['row'] - 1\n",
                    "    c = record['col'] - 1\n",
                    "    \n",
                    "    mask = np.zeros(cleaned.shape, dtype=np.uint8)\n",
                    "    cv2.drawContours(mask, [cnt], -1, 255, -1)\n",
                    "    pts = np.column_stack(np.where(mask == 255))\n",
                    "    pts = pts[:, [1, 0]].astype(np.float32) # Swap to x, y format\n",
                    "    \n",
                    "    ax = axes[r, c]\n",
                    "    ax.clear()\n",
                    "    ax.scatter(pts[:, 0], pts[:, 1], s=1, color='forestgreen', alpha=0.3)\n",
                    "    ax.scatter(center[0], center[1], color='gold', s=40, edgecolors='black', zorder=5)\n",
                    "    \n",
                    "    cx, cy = center\n",
                    "    ax.plot([cx - eigenvectors[0, 0] * length, cx + eigenvectors[0, 0] * length],\n",
                    "            [cy - eigenvectors[0, 1] * length, cy + eigenvectors[0, 1] * length],\n",
                    "            color='red', linewidth=2, label='Major' if (r == 0 and c == 0) else \"\")\n",
                    "    ax.plot([cx - eigenvectors[1, 0] * width, cx + eigenvectors[1, 0] * width],\n",
                    "            [cy - eigenvectors[1, 1] * width, cy + eigenvectors[1, 1] * width],\n",
                    "            color='blue', linewidth=2, label='Minor' if (r == 0 and c == 0) else \"\")\n",
                    "            \n",
                    "    ax.set_title(f\"Plant ID: {record['id']}\")\n",
                    "    ax.invert_yaxis()\n",
                    "    ax.axis('equal')\n",
                    "    ax.grid(True, linestyle=':', alpha=0.5)\n",
                    "\n",
                    "plt.suptitle(\"Spatial PCA of Plant Shapes (3x4 Grid Layout)\", fontsize=16, y=0.98)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
                cell["source"] = new_source
                updated_spatial_pca = True
                print(f"Updated Spatial PCA cell (Index {i}) in {path}")
                
    if updated_color_pca or updated_spatial_pca:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Saved updates to {path}")
    else:
        print(f"No updates required for {path}")

def main():
    update_notebook("leaf_segmentation_tutorial_clean.ipynb")
    update_notebook("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
