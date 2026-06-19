import json

with open("leaf_segmentation_tutorial_clean.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

output = []
output.append(f"Total cells in clean notebook: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source_lines = cell.get("source", [])
    source_str = "".join(source_lines)
    
    # Check if this cell is markdown or code
    found_lines = []
    for l_idx, line in enumerate(source_lines):
        if "cleaned" in line or "cleaned_mask" in line:
            found_lines.append(f"  Line {l_idx+1}: {line.rstrip()}")
            
    if found_lines:
        output.append(f"Cell {i:02d} ({cell_type}):")
        output.extend(found_lines)

with open("scan_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
print("Scan completed successfully.")
