import json

def inspect(file_path, out_file):
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    with open(out_file, "w", encoding="utf-8") as out:
        for i, cell in enumerate(nb.get("cells", [])):
            cell_type = cell.get("cell_type")
            source = cell.get("source", [])
            first_line = source[0].strip() if source else "<empty>"
            out.write(f"Cell {i} ({cell_type}): {first_line}\n")
            if cell_type == "code":
                for j, line in enumerate(source[:5]):
                    out.write(f"  Line {j}: {line.rstrip()}\n")

inspect("leaf_segmentation_tutorial_clean.ipynb", "clean_nb_cells.txt")
inspect("leaf_segmentation_tutorial.ipynb", "tutorial_nb_cells.txt")

