import json

with open("leaf_segmentation_tutorial.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb.get("cells", [])):
    source = "".join(cell.get("source", []))
    if "Step 2:" in source:
        found = True
        print(f"--- Cell {i} (Markdown) ---")
        print(source)
    elif found:
        # Print subsequent cells until the next step
        if cell.get("cell_type") == "markdown" and "Step " in source:
            break
        print(f"--- Cell {i} ({cell.get('cell_type')}) ---")
        # For code cells, let's print the code and some info about outputs
        if cell.get("cell_type") == "code":
            print(source)
            print("Outputs count:", len(cell.get("outputs", [])))
