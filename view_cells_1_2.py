import json

with open("leaf_segmentation_tutorial.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [1, 2]:
    cell = nb.get("cells", [])[i]
    print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
    print("".join(cell.get("source", [])))
