import json
import os

def update_final_notebook():
    path = "final.ipynb"
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    # Cell 5 is the main comparison loop
    # Let's replace its source code with the color-only PCA version
    new_source = [
        "# Define how to extract color features (excluding lightness) for each color space\n",
        "def get_rgb_features(rgb):\n",
        "    r, g, b = cv2.split(rgb)\n",
        "    total = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + 1e-8\n",
        "    r_norm = r.astype(np.float32) / total\n",
        "    g_norm = g.astype(np.float32) / total\n",
        "    b_norm = b.astype(np.float32) / total\n",
        "    return r_norm, g_norm, b_norm\n",
        "\n",
        "color_spaces = {\n",
        "    'RGB (Normalized)': lambda rgb: get_rgb_features(rgb),\n",
        "    'CIELAB (Color-Only)': lambda rgb: cv2.split(cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab))[1:], # a* and b* only\n",
        "    'HSV (Color-Only)': lambda rgb: cv2.split(cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV))[:2]     # H and S only\n",
        "}\n",
        "\n",
        "# CIELAB a* reference channel for polarity correction\n",
        "_, a_ref, _ = cv2.split(cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab))\n",
        "\n",
        "for space_name, extract_func in color_spaces.items():\n",
        "    print(f\"\\n\" + \"=\"*60)\n",
        "    print(f\"Processing Color Space: {space_name}\")\n",
        "    print(\"=\"*60)\n",
        "    \n",
        "    # 1. Split channels\n",
        "    channels = extract_func(img_rgb)\n",
        "    \n",
        "    # 2. Stack channels for PCA\n",
        "    flats = [ch.astype(np.float32).reshape(-1, 1) for ch in channels]\n",
        "    features = np.hstack(flats)\n",
        "    \n",
        "    # 3. Run PCA\n",
        "    mean, eigenvectors, eigenvalues = cv2.PCACompute2(features, mean=None)\n",
        "    projected = cv2.PCAProject(features, mean, eigenvectors)\n",
        "    pc1 = projected[:, 0].reshape(channels[0].shape)\n",
        "    \n",
        "    # 4. Polarity Correction using CIELAB a* (negative values represent green)\n",
        "    correlation = np.corrcoef(pc1.ravel(), a_ref.ravel())[0, 1]\n",
        "    if correlation > 0:\n",
        "        pc1 = -pc1\n",
        "        \n",
        "    # Scale to 0-255\n",
        "    pc1_min = pc1.min()\n",
        "    pc1_max = pc1.max()\n",
        "    pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))\n",
        "    \n",
        "    # 5. Apply Otsu's Thresholding\n",
        "    ret, thresh = cv2.threshold(pc1_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
        "    print(f\"[{space_name}] Otsu threshold value: {ret}\")\n",
        "    \n",
        "    # 6. Morphological Cleaning\n",
        "    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
        "    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)\n",
        "    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)\n",
        "    \n",
        "    # 7. Post-processing: Geometric Contour Filtering\n",
        "    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
        "    filtered_mask = np.zeros_like(cleaned)\n",
        "    for cnt in contours:\n",
        "        area = cv2.contourArea(cnt)\n",
        "        if area < 100 or area > 150000:\n",
        "            continue\n",
        "        x, y, w, h = cv2.boundingRect(cnt)\n",
        "        aspect_ratio = float(w) / h\n",
        "        # Ignore background channel lines and extreme shapes\n",
        "        if w > 0.8 * img_rgb.shape[1] or aspect_ratio > 4.0 or aspect_ratio < 0.25:\n",
        "            continue\n",
        "        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)\n",
        "    cleaned = filtered_mask\n",
        "    \n",
        "    # Extract final filtered contours\n",
        "    final_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
        "    \n",
        "    # Draw contours on RGB image copy\n",
        "    img_contour = img_rgb.copy()\n",
        "    cv2.drawContours(img_contour, final_contours, -1, (0, 255, 0), 2)\n",
        "    \n",
        "    # Plot step results side-by-side\n",
        "    fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
        "    \n",
        "    axes[0].imshow(pc1_scaled, cmap='gray')\n",
        "    axes[0].set_title(f\"Polarity-Corrected PC1 ({space_name})\")\n",
        "    axes[0].axis('off')\n",
        "    \n",
        "    axes[1].imshow(cleaned, cmap='gray')\n",
        "    axes[1].set_title(f\"Filtered Binary Mask ({space_name})\")\n",
        "    axes[1].axis('off')\n",
        "    \n",
        "    axes[2].imshow(img_contour)\n",
        "    axes[2].set_title(f\"Green Contour Overlay ({space_name}) - Detected: {len(final_contours)}\")\n",
        "    axes[2].axis('off')\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()"
    ]
    
    nb["cells"][5]["source"] = new_source
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated final.ipynb.")

def update_test_notebook():
    path = "segmentation-test.ipynb"
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            source_str = "".join(source)
            if "    # Define the 3 color spaces to evaluate individually" in source_str:
                # Replace everything in this cell from defining the color spaces to the plotting code
                new_source = [
                    "    # Define color features (excluding lightness) for each color space\n",
                    "    def get_rgb_features(rgb):\n",
                    "        r, g, b = cv2.split(rgb)\n",
                    "        total = r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32) + 1e-8\n",
                    "        r_norm = r.astype(np.float32) / total\n",
                    "        g_norm = g.astype(np.float32) / total\n",
                    "        b_norm = b.astype(np.float32) / total\n",
                    "        return r_norm, g_norm, b_norm\n",
                    "\n",
                    "    color_spaces = {\n",
                    "        'RGB (Normalized)': lambda rgb: get_rgb_features(rgb),\n",
                    "        'CIELAB (Color-Only)': lambda rgb: cv2.split(cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2Lab))[1:], # a* and b* only\n",
                    "        'HSV (Color-Only)': lambda rgb: cv2.split(cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV))[:2]     # H and S only\n",
                    "    }\n",
                    "    \n",
                    "    for space_name, extract_func in color_spaces.items():\n",
                    "        print(f\"\\n\" + \"-\"*40)\n",
                    "        print(f\"Color Space: {space_name} (Image {idx+1})\")\n",
                    "        print(\"-\"*40)\n",
                    "        \n",
                    "        # Stack channels into feature representation\n",
                    "        channels = extract_func(img_rgb)\n",
                    "        flats = [ch.astype(np.float32).reshape(-1, 1) for ch in channels]\n",
                    "        features = np.hstack(flats)\n",
                    "        \n",
                    "        # Perform PCA\n",
                    "        mean, eigenvectors, eigenvalues = cv2.PCACompute2(features, mean=None)\n",
                    "        projected = cv2.PCAProject(features, mean, eigenvectors)\n",
                    "        pc1 = projected[:, 0].reshape(channels[0].shape)\n",
                    "        \n",
                    "        # Polarity Correction using CIELAB a* channel as baseline for greenness\n",
                    "        correlation = np.corrcoef(pc1.ravel(), a.ravel())[0, 1]\n",
                    "        if correlation > 0:\n",
                    "            pc1 = -pc1\n",
                    "            \n",
                    "        # Scale to 0-255\n",
                    "        pc1_min = pc1.min()\n",
                    "        pc1_max = pc1.max()\n",
                    "        pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))\n",
                    "        \n",
                    "        # Explained Variance Ratios\n",
                    "        total_variance = np.sum(eigenvalues)\n",
                    "        var_ratios = (eigenvalues[:, 0] / total_variance) * 100\n",
                    "        \n",
                    "        # Plot Explained Variance Bar Chart\n",
                    "        plt.figure(figsize=(6, 4))\n",
                    "        pc_labels = [f'PC{i+1}' for i in range(len(var_ratios))]\n",
                    "        bars = plt.bar(pc_labels, var_ratios, color=['royalblue', 'orange', 'forestgreen'][:len(var_ratios)], alpha=0.8)\n",
                    "        plt.ylabel('Percentage of Explained Variance (%)')\n",
                    "        plt.title(f'PCA Explained Variance ({space_name}) - Image {idx+1}')\n",
                    "        plt.ylim(0, 110)\n",
                    "        for bar in bars:\n",
                    "            yval = bar.get_height()\n",
                    "            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f\"{yval:.2f}%\", ha='center', va='bottom', fontweight='bold')\n",
                    "        plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "        plt.show()\n",
                    "        \n",
                    "        # Apply Otsu's thresholding\n",
                    "        ret, thresh = cv2.threshold(pc1_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
                    "        print(f\"[{space_name}] Otsu threshold value: {ret}\")\n",
                    "        \n",
                    "        # Morphological Cleanup\n",
                    "        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n",
                    "        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)\n",
                    "        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)\n",
                    "        \n",
                    "        # Post-processing: Filter out background/gulley lines geometrically\n",
                    "        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                    "        filtered_mask = np.zeros_like(cleaned)\n",
                    "        for cnt in contours:\n",
                    "            area = cv2.contourArea(cnt)\n",
                    "            if area < 100 or area > 150000:\n",
                    "                continue\n",
                    "            x, y, w, h = cv2.boundingRect(cnt)\n",
                    "            aspect_ratio = float(w) / h\n",
                    "            if w > 0.8 * img_bgr.shape[1] or aspect_ratio > 4.0 or aspect_ratio < 0.25:\n",
                    "                continue\n",
                    "            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)\n",
                    "        cleaned = filtered_mask\n",
                    "        \n",
                    "        # Find external contours\n",
                    "        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                    "        \n",
                    "        # Draw contours on RGB image copy\n",
                    "        img_contour = img_rgb.copy()\n",
                    "        cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 2)\n",
                    "        \n",
                    "        # Plot Results as Individual Images\n",
                    "        # 1. Polarity-Corrected PC1\n",
                    "        plt.figure(figsize=(6, 6))\n",
                    "        plt.imshow(pc1_scaled, cmap='gray')\n",
                    "        plt.title(f\"Polarity-Corrected PC1 ({space_name}) - Image {idx+1}\")\n",
                    "        plt.axis('off')\n",
                    "        plt.show()\n",
                    "        \n",
                    "        # 2. Otsu binary mask\n",
                    "        plt.figure(figsize=(6, 6))\n",
                    "        plt.imshow(cleaned, cmap='gray')\n",
                    "        plt.title(f\"Filtered Binary Mask ({space_name}) - Image {idx+1}\")\n",
                    "        plt.axis('off')\n",
                    "        plt.show()\n",
                    "        \n",
                    "        # 3. Contour overlay\n",
                    "        plt.figure(figsize=(6, 6))\n",
                    "        plt.imshow(img_contour)\n",
                    "        plt.title(f\"Green Contour Overlay ({space_name}) - Image {idx+1} - Detected: {len(contours)}\")\n",
                    "        plt.axis('off')\n",
                    "        plt.show()\n"
                ]
                cell["source"] = new_source
                updated = True
                print("Successfully updated segmentation-test.ipynb.")
                
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
    else:
        print("Could not find matching code cell in segmentation-test.ipynb.")

if __name__ == "__main__":
    update_final_notebook()
    update_test_notebook()
