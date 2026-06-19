import json
import os

def main():
    path = "segmentation-test.ipynb"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            source_str = "".join(source)
            # Find the code cell performing the main contour drawing loop
            if "        # Find external contours" in source_str and "filtered_mask" not in source_str:
                new_source = []
                for line in source:
                    if "        # Find external contours" in line:
                        new_source.extend([
                            "        # Post-processing: Filter out background/gulley lines geometrically\n",
                            "        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                            "        filtered_mask = np.zeros_like(cleaned)\n",
                            "        for cnt in contours:\n",
                            "            area = cv2.contourArea(cnt)\n",
                            "            if area < 100 or area > 150000:\n",
                            "                continue\n",
                            "            x, y, w, h = cv2.boundingRect(cnt)\n",
                            "            aspect_ratio = float(w) / h\n",
                            "            if w > 0.8 * img_bgr.shape[1] or aspect_ratio > 4.0 or aspect_ratio < 0.25:\n",
                            "                continue\n",
                            "            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)\n",
                            "        cleaned = filtered_mask\n",
                            "        \n",
                            "        # Find external contours\n",
                            "        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
                        ])
                    elif "        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)" in line:
                        # Skip the original line since it is replaced by our block
                        continue
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                updated = True
                print("Successfully updated the contour drawing cell in segmentation-test.ipynb.")

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Saved updated notebook file.")
    else:
        print("No matching cell found in segmentation-test.ipynb, or it has already been updated.")

if __name__ == "__main__":
    main()
