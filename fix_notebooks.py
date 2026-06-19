import json
import os

def fix_notebook(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    updated_cells_count = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            source_str = "".join(source_lines)
            
            # Check if replacements are needed
            has_cleaned_mask = "cleaned_mask" in source_str
            has_adaptive_cleaned = "adaptive_cleaned" in source_str
            has_kmeans_cleaned = "kmeans_cleaned" in source_str
            has_watershed_start = "sure_bg = cv2.dilate(cleaned" in source_str and "if 'cleaned' not in locals():" not in source_str
            
            if has_cleaned_mask or has_adaptive_cleaned or has_kmeans_cleaned or has_watershed_start:
                new_lines = []
                # If it's the watershed cell, prepend the safety check
                if has_watershed_start:
                    new_lines.extend([
                        "if 'cleaned' not in locals():\n",
                        "    raise NameError(\"Segmentation mask 'cleaned' not found. Please execute one of the thresholding steps first.\")\n",
                        "\n"
                    ])
                
                for line in source_lines:
                    updated_line = line
                    # Perform replacements
                    updated_line = updated_line.replace("cleaned_mask", "cleaned")
                    updated_line = updated_line.replace("adaptive_cleaned", "cleaned")
                    updated_line = updated_line.replace("kmeans_cleaned", "cleaned")
                    new_lines.append(updated_line)
                
                cell["source"] = new_lines
                updated_cells_count += 1
                
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Successfully processed {file_path}. Updated {updated_cells_count} cells.")

if __name__ == "__main__":
    fix_notebook("leaf_segmentation_tutorial_clean.ipynb")
    fix_notebook("leaf_segmentation_tutorial.ipynb")
