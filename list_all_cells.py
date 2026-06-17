import json

with open("leaf_segmentation_tutorial.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    source = cell.get("source", [])
    first_line = source[0].strip() if source else "<empty>"
    print(f"Cell {i:02d}: type={cell.get('cell_type'):<10} | first_line={first_line[:80]}")
