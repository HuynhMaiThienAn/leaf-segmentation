import json

with open("leaf_segmentation_tutorial_clean.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source_str = "".join(cell.get("source", []))
        if any(keyword in source_str for keyword in ["threshold", "adaptive", "kmeans", "watershed"]):
            print(f"--- Cell {i} ---")
            print("SOURCE:")
            print(source_str[:300] + "..." if len(source_str) > 300 else source_str)
            print("OUTPUTS:")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    print("".join(output.get("text", [])))
                elif output.get("output_type") == "execute_result":
                    print(output.get("data", {}).get("text/plain", ""))
                elif output.get("output_type") == "display_data":
                    print("[Image Displayed]")
