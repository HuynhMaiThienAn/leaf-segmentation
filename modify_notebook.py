import json

# Load the clean notebook
with open("leaf_segmentation_tutorial_clean.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new content for cell 3 (Markdown)
new_markdown_source = [
    "### Step 2: Channel Separation, Saturation Filtering & Visualization\n",
    "\n",
    "We split the CIELAB image into L*, a*, and b* channels to analyze lightness and color differences:\n",
    "- **L\\* Channel**: Represents lightness.\n",
    "- **a\\* Channel**: Represents green-to-red color variation (negative values are green, positive are red).\n",
    "- **b\\* Channel**: Represents blue-to-yellow color variation (negative values are blue, positive are yellow).\n",
    "\n",
    "In addition, we convert the original image to the **HSV color space** to extract the **Saturation channel**. Plants are highly saturated compared to the background elements (grow medium, white plastic grid, grey tubes, and shadows). Filtering by saturation helps isolate potential plant pixels before running PCA."
]

# Define the new content for cell 4 (Code)
new_code_source = [
    "# 1. Split CIELAB channels\n",
    "l_channel, a_channel, b_channel = cv2.split(img_lab)\n",
    "\n",
    "# 2. Extract Hue and Saturation channels from HSV space\n",
    "img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)\n",
    "h_channel, s_channel, _ = cv2.split(img_hsv)\n",
    "\n",
    "# 3. Filter by saturation using Otsu's thresholding\n",
    "sat_threshold, sat_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n",
    "\n",
    "# Mask the original RGB image using the saturation mask to show the filtered result\n",
    "img_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=sat_mask)\n",
    "\n",
    "# 4. Plot CIELAB channels, Saturation channel, Saturation mask, and Filtered result\n",
    "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n",
    "\n",
    "# Row 1: CIELAB Channels\n",
    "axes[0, 0].imshow(l_channel, cmap='gray')\n",
    "axes[0, 0].set_title(\"L* Channel (Lightness)\")\n",
    "axes[0, 0].axis('off')\n",
    "\n",
    "axes[0, 1].imshow(a_channel, cmap='gray')\n",
    "axes[0, 1].set_title(\"a* Channel (Green-Red)\")\n",
    "axes[0, 1].axis('off')\n",
    "\n",
    "axes[0, 2].imshow(b_channel, cmap='gray')\n",
    "axes[0, 2].set_title(\"b* Channel (Blue-Yellow)\")\n",
    "axes[0, 2].axis('off')\n",
    "\n",
    "# Row 2: Saturation Channel, Mask, and Masked Result\n",
    "axes[1, 0].imshow(s_channel, cmap='viridis')\n",
    "axes[1, 0].set_title(\"Saturation Channel (HSV)\")\n",
    "axes[1, 0].axis('off')\n",
    "\n",
    "axes[1, 1].imshow(sat_mask, cmap='gray')\n",
    "axes[1, 1].set_title(f\"Saturation Mask (Otsu Thresh: {sat_threshold:.1f})\")\n",
    "axes[1, 1].axis('off')\n",
    "\n",
    "axes[1, 2].imshow(img_masked)\n",
    "axes[1, 2].set_title(\"Saturation-Filtered RGB Image\")\n",
    "axes[1, 2].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]

# Update cell 3 and cell 4
nb["cells"][3]["source"] = new_markdown_source
nb["cells"][4]["source"] = new_code_source

# Save to leaf_segmentation_tutorial.ipynb
with open("leaf_segmentation_tutorial.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully modified and saved leaf_segmentation_tutorial.ipynb")
