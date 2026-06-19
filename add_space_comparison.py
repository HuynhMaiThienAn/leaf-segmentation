import json
import os

def insert_comparison(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    
    # Check if already inserted
    already_inserted = False
    for cell in cells:
        src = "".join(cell.get("source", []))
        if "RGB Color Space (Green vs. Red" in src:
            already_inserted = True
            break
            
    if already_inserted:
        print(f"Comparison plots already present in {path}.")
        return
        
    # Find Step 2 code cell
    insert_idx = -1
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "l_channel, a_channel, b_channel = cv2.split(img_lab)" in src:
                insert_idx = idx + 1
                break
                
    if insert_idx == -1:
        print(f"Error: Could not locate Step 2 code cell in {path}.")
        return
        
    new_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2D Color Space Analysis: CIELAB (a* vs. b*) vs. RGB (Green vs. Red) without PCA\n",
            "\n",
            "To understand why we project the color space using PCA, we can visualize the 2D pixel color distributions of both spaces *without* PCA. We sample the pixels and color each coordinate point by its actual RGB value:\n",
            "1. **CIELAB (a* vs. b*)**: Shows how greenness is decoupled from lightness. The vegetation forms a clear, isolated branch stretching to the left (negative a*) and upwards (positive b*).\n",
            "2. **RGB (Green vs. Red)**: Shows that vegetation stands out above the diagonal line $G = R$ (which represents gray/neutral colors like the grids and pipes)."
        ]
    }
    
    new_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Sample pixels for clean and fast rendering\n",
            "sample_rate = 200\n",
            "r_sample = img_rgb[:, :, 0].ravel()[::sample_rate]\n",
            "g_sample = img_rgb[:, :, 1].ravel()[::sample_rate]\n",
            "b_sample = img_rgb[:, :, 2].ravel()[::sample_rate]\n",
            "rgb_sample = img_rgb.reshape(-1, 3)[::sample_rate] / 255.0\n",
            "\n",
            "a_sample = a_channel.ravel()[::sample_rate]\n",
            "b_sample = b_channel.ravel()[::sample_rate]\n",
            "\n",
            "# 2. Plot the scatter plots side-by-side\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
            "\n",
            "# CIELAB a* vs b* Scatter\n",
            "axes[0].scatter(a_sample, b_sample, c=rgb_sample, alpha=0.6, edgecolors='none')\n",
            "axes[0].set_title(\"CIELAB Color Space (a* vs. b* - True Colors)\")\n",
            "axes[0].set_xlabel(\"a* Channel (Green to Red)\")\n",
            "axes[0].set_ylabel(\"b* Channel (Blue to Yellow)\")\n",
            "axes[0].axvline(128, color='gray', linestyle='--', alpha=0.4)\n",
            "axes[0].axhline(128, color='gray', linestyle='--', alpha=0.4)\n",
            "axes[0].grid(True, linestyle=':', alpha=0.5)\n",
            "\n",
            "# RGB Green vs Red Scatter\n",
            "axes[1].scatter(r_sample, g_sample, c=rgb_sample, alpha=0.6, edgecolors='none')\n",
            "axes[1].set_title(\"RGB Color Space (Green vs. Red - True Colors)\")\n",
            "axes[1].set_xlabel(\"Red Channel (0-255)\")\n",
            "axes[1].set_ylabel(\"Green Channel (0-255)\")\n",
            "axes[1].plot([0, 255], [0, 255], color='red', linestyle='--', alpha=0.5, label='R = G (Neutral)')\n",
            "axes[1].grid(True, linestyle=':', alpha=0.5)\n",
            "axes[1].legend()\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
    
    cells.insert(insert_idx, new_md_cell)
    cells.insert(insert_idx + 1, new_code_cell)
    
    nb["cells"] = cells
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully inserted comparison plots into {path}")

def main():
    insert_comparison("leaf_segmentation_tutorial_clean.ipynb")
    insert_comparison("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
