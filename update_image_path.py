import json
import os

def update_image_path(path):
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist.")
        return
        
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    updated = False
    
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "Load sample image" in source and "img_path = 'leaf-image/veg_1/IMG_7993.JPG'" in source:
                new_source = [
                    "import cv2\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Load sample image\n",
                    "img_path = 'leaf-image/veg_3/IMG_8001.JPG'\n",
                    "img_bgr = cv2.imread(img_path)\n",
                    "if img_bgr is None:\n",
                    "    raise FileNotFoundError(f\"Could not load image at {img_path}\")\n",
                    "\n",
                    "img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
                    "img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)\n",
                    "\n",
                    "print(f\"Loaded image: {img_path}\")\n",
                    "print(f\"Image Resolution: {img_rgb.shape[1]}x{img_rgb.shape[0]}\")\n",
                    "\n",
                    "# Display original image\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "plt.imshow(img_rgb)\n",
                    "plt.title(\"Original RGB Image\")\n",
                    "plt.axis('off')\n",
                    "plt.show()"
                ]
                cell["source"] = new_source
                updated = True
                print(f"Updated sample image cell (Index {i}) in {path}")
                
    if updated:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Saved updates to {path}")
    else:
        print(f"No updates required for {path} (or path already updated)")

def main():
    update_image_path("leaf_segmentation_tutorial_clean.ipynb")
    update_image_path("leaf_segmentation_tutorial.ipynb")

if __name__ == "__main__":
    main()
