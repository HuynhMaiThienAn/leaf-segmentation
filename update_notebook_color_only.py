import json
import os

def update_to_color_only(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    
    # 1. Truncate cells to keep only the first 9 cells (indices 0 to 8)
    # This keeps Steps 1 to 6 and removes Steps 7 & 8 plus any empty trailing cells
    truncated_cells = cells[:9]
    
    # 2. Update the introduction cell (index 0)
    new_intro = [
        "# Leaf Color Segmentation Tutorial\n",
        "\n",
        "This notebook demonstrates a robust, step-by-step pipeline to segment plant leaves from background elements in a top-down hydroponic setup using the **CIELAB Color Space** and **Principal Component Analysis (PCA)** on color channels.\n",
        "\n",
        "### Pipeline Steps:\n",
        "1. **Load Image & Convert Color Space**: Convert to RGB for visualization and CIELAB for color processing.\n",
        "2. **Channel Separation, Saturation Filtering & Visualization**: Split CIELAB channels and extract the Saturation channel to isolate highly saturated vegetation.\n",
        "3. **CIELAB-PCA Projection**: Project a* and b* color channels onto the first principal component (PC1) to isolate vegetation.\n",
        "4. **Polarity Correction**: Ensure consistent positive values for green vegetation by checking correlation with the a* channel.\n",
        "5. **Otsu's Thresholding**: Create a binary mask from PC1.\n",
        "6. **Morphological Cleaning**: Apply opening and closing operations to clean background noise and fill leaf holes."
    ]
    
    truncated_cells[0]["source"] = new_intro
    nb["cells"] = truncated_cells
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully truncated {path} to focus on color segmentation only.")

def main():
    update_to_color_only("leaf_segmentation_tutorial_clean.ipynb")
    update_to_color_only("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
