import json

with open("leaf_segmentation_tutorial.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Clear outputs and execution counts
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open("leaf_segmentation_tutorial_clean.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook cleaned successfully!")
