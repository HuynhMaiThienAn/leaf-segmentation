import cv2
import numpy as np

img_path = 'leaf-image/veg_1/IMG_7993.JPG'
img_bgr = cv2.imread(img_path)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(img_hsv)

# Apply Otsu's thresholding on Saturation
otsu_thresh, sat_mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's threshold on Saturation channel: {otsu_thresh}")
print(f"Pixels selected: {np.count_nonzero(sat_mask)} ({100.0 * np.count_nonzero(sat_mask) / sat_mask.size:.2f}%)")
