import cv2
import numpy as np

img_path = 'leaf-image/veg_1/IMG_7993.JPG'
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

# Convert to HSV to get Saturation
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(img_hsv)

print("Saturation channel stats:")
print(f"Min: {s_channel.min()}")
print(f"Max: {s_channel.max()}")
print(f"Mean: {s_channel.mean():.2f}")
print(f"Median: {np.median(s_channel)}")

# Let's see how many pixels are above some thresholds (saturation is 0-255 in OpenCV)
for thresh in [30, 40, 50, 60, 70, 80]:
    cnt = np.count_nonzero(s_channel > thresh)
    pct = 100.0 * cnt / s_channel.size
    print(f"Pixels with Saturation > {thresh}: {cnt} ({pct:.2f}%)")
