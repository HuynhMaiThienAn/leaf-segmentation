import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'leaf-image/veg_1/IMG_7993.JPG'
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

# 1. Split CIELAB channels
l_channel, a_channel, b_channel = cv2.split(img_lab)

# 2. Extract Saturation channel from HSV space
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
_, s_channel, _ = cv2.split(img_hsv)

# 3. Filter by saturation using a threshold
# We will use Otsu's thresholding to dynamically find a good saturation threshold,
# or we can apply a fixed threshold (e.g., Saturation > 40)
sat_threshold, sat_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Mask the original RGB image using the saturation mask to show the filtered result
img_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=sat_mask)

print("Otsu threshold:", sat_threshold)
print("Mask shape:", sat_mask.shape)
print("Masked image shape:", img_masked.shape)
