import json

with open("leaf_segmentation_tutorial_clean.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = "".join(cell.get("source", []))
    first_line = source.split("\n")[0] if source else ""
    # Print any line containing cleaned or cleaned_mask in code cells
    if cell_type == "code":
        lines = []
        for l_idx, line in enumerate(source.split("\n")):
            if "cleaned" in line or "cleaned_mask" in line:
                lines.append(f"   Line {l_idx+1}: {line.strip()}")
        if lines:
            print(f"Cell {i:02d}: {cell_type:<10} | {first_line[:70]}")
            for l in lines:
                print(l)
