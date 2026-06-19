import json
import os

def append_alternatives(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    
    # Check if already inserted
    already_appended = False
    for cell in cells:
        src = "".join(cell.get("source", []))
        if "Alternative Technique 1" in src:
            already_appended = True
            break
            
    if already_appended:
        print(f"Alternative techniques already appended in {path}.")
        return

    # Create cells for Adaptive Thresholding
    md_adaptive = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7: Alternative Technique 1 - Adaptive Thresholding\n",
            "\n",
            "Unlike Otsu's method which finds a single global threshold, **Adaptive Thresholding** calculates thresholds for small local neighborhoods of pixels. This helps handle non-uniform lighting conditions across the hydroponic grid."
        ]
    }
    
    code_adaptive = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Apply Adaptive Gaussian Thresholding on the PC1 channel\n",
            "adaptive_mask = cv2.adaptiveThreshold(\n",
            "    pc1_scaled, \n",
            "    255, \n",
            "    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \n",
            "    cv2.THRESH_BINARY, \n",
            "    101,  # Block size (neighborhood size)\n",
            "    2     # Constant subtracted from the mean\n",
            ")\n",
            "\n",
            "# 2. Clean mask morphologically\n",
            "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
            "cleaned = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel)\n",
            "cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)\n",
            "\n",
            "# Mask the original image\n",
            "img_adaptive = cv2.bitwise_and(img_rgb, img_rgb, mask=cleaned)\n",
            "\n",
            "# 3. Plot mask and segmented result\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 8))\n",
            "axes[0].imshow(cleaned, cmap='gray')\n",
            "axes[0].set_title(\"Adaptive Threshold Mask\")\n",
            "axes[0].axis('off')\n",
            "\n",
            "axes[1].imshow(img_adaptive)\n",
            "axes[1].set_title(\"Adaptive Threshold Segmentation\")\n",
            "axes[1].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Create cells for K-Means Clustering
    md_kmeans = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 8: Alternative Technique 2 - K-Means Clustering\n",
            "\n",
            "**K-Means Clustering** is an unsupervised algorithm that groups pixels into $K$ clusters based on color proximity. Here, we run K-Means with $K=2$ on our 4D color features (L*, a*, b*, S) to partition the image into \"plants\" and \"background\" without manually setting any threshold."
        ]
    }
    
    code_kmeans = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Standardize features for K-Means (important for distance-based clustering)\n",
            "features_std = (color_features - color_features.mean(axis=0)) / (color_features.std(axis=0) + 1e-8)\n",
            "Z = features_std.astype(np.float32)\n",
            "\n",
            "# 2. Define criteria and run K-Means (K=2)\n",
            "criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)\n",
            "K = 2\n",
            "ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)\n",
            "\n",
            "# Reshape label back to 2D mask\n",
            "kmeans_mask = label.reshape(a_channel.shape)\n",
            "\n",
            "# Identify which cluster corresponds to the plant\n",
            "# In our 4D space, the second channel is a* (index 1). Green leaves have lower a* values.\n",
            "plant_cluster = np.argmin(center[:, 1])\n",
            "kmeans_mask = np.uint8((kmeans_mask == plant_cluster) * 255)\n",
            "\n",
            "# 3. Clean mask morphologically\n",
            "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
            "cleaned = cv2.morphologyEx(kmeans_mask, cv2.MORPH_OPEN, kernel)\n",
            "cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)\n",
            "\n",
            "# Mask the original image\n",
            "img_kmeans = cv2.bitwise_and(img_rgb, img_rgb, mask=cleaned)\n",
            "\n",
            "# 4. Plot mask and segmented result\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 8))\n",
            "axes[0].imshow(cleaned, cmap='gray')\n",
            "axes[0].set_title(\"K-Means (K=2) Mask\")\n",
            "axes[0].axis('off')\n",
            "\n",
            "axes[1].imshow(img_kmeans)\n",
            "axes[1].set_title(\"K-Means Segmentation\")\n",
            "axes[1].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Create cells for Watershed
    md_watershed = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 9: Alternative Technique 3 - Watershed Segmentation\n",
            "\n",
            "The **Watershed Algorithm** treats the image as a topographic surface. We use the distance transform of our binary mask to identify \"sure foreground\" markers (peaks/centers of plants) and \"sure background\" markers, and then flood the surface to find precise boundaries. This is especially useful for separating touching leaves."
        ]
    }
    
    code_watershed = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if 'cleaned' not in locals():\n",
            "    raise NameError(\"Segmentation mask 'cleaned' not found. Please execute one of the thresholding steps first.\")\n",
            "\n",
            "# 1. We use the cleaned Otsu mask as the starting point\n",
            "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
            "\n",
            "# Finding sure background area (dilate the mask to make sure background is clear)\n",
            "sure_bg = cv2.dilate(cleaned, kernel, iterations=3)\n",
            "\n",
            "# 2. Finding sure foreground area (using distance transform)\n",
            "dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)\n",
            "_, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)\n",
            "sure_fg = np.uint8(sure_fg)\n",
            "\n",
            "# 3. Finding unknown region (subtracting sure foreground from sure background)\n",
            "unknown = cv2.subtract(sure_bg, sure_fg)\n",
            "\n",
            "# 4. Marker labeling\n",
            "_, markers = cv2.connectedComponents(sure_fg)\n",
            "\n",
            "# Add one to all labels so that sure background is not 0, but 1\n",
            "markers = markers + 1\n",
            "\n",
            "# Mark the unknown region with 0\n",
            "markers[unknown == 255] = 0\n",
            "\n",
            "# 5. Run Watershed on the RGB image\n",
            "markers_ws = cv2.watershed(img_rgb, markers.copy())\n",
            "\n",
            "# Create the final watershed segmentation mask (all labeled regions > 1)\n",
            "watershed_mask = np.zeros_like(cleaned)\n",
            "watershed_mask[markers_ws > 1] = 255\n",
            "\n",
            "# Clean mask morphologically\n",
            "watershed_cleaned = cv2.morphologyEx(watershed_mask, cv2.MORPH_OPEN, kernel)\n",
            "watershed_cleaned = cv2.morphologyEx(watershed_cleaned, cv2.MORPH_CLOSE, kernel)\n",
            "\n",
            "# Mask the original image\n",
            "img_watershed = cv2.bitwise_and(img_rgb, img_rgb, mask=watershed_cleaned)\n",
            "\n",
            "# 6. Plot mask and segmented result\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 8))\n",
            "axes[0].imshow(watershed_cleaned, cmap='gray')\n",
            "axes[0].set_title(\"Watershed Mask\")\n",
            "axes[0].axis('off')\n",
            "\n",
            "axes[1].imshow(img_watershed)\n",
            "axes[1].set_title(\"Watershed Segmentation\")\n",
            "axes[1].axis('off')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    
    # Append the cells to the notebook
    cells.append(md_adaptive)
    cells.append(code_adaptive)
    cells.append(md_kmeans)
    cells.append(code_kmeans)
    cells.append(md_watershed)
    cells.append(code_watershed)
    
    # Also update the title block to include the new steps
    intro_source = cells[0].get("source", [])
    intro_str = "".join(intro_source)
    if "Step 7:" not in intro_str:
        new_intro = intro_source.copy()
        new_intro.append("\n7. **Adaptive Thresholding (Alternative)**: Compare local adaptive thresholding against global Otsu.")
        new_intro.append("\n8. **K-Means Clustering (Alternative)**: Compare color-space clustering segmentation.")
        new_intro.append("\n9. **Watershed Segmentation (Alternative)**: Compare marker-controlled region flooding.")
        cells[0]["source"] = new_intro

    nb["cells"] = cells
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully appended alternative techniques to {path}")

def main():
    append_alternatives("leaf_segmentation_tutorial_clean.ipynb")
    append_alternatives("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
