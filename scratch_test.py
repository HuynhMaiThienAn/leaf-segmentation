import cv2
import numpy as np

# Read image
img_path = r"leaf-image\veg_1\IMG_7993.JPG"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")

print("Image shape:", img.shape)

# Convert to Lab
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
l_channel, a_channel, b_channel = cv2.split(img_lab)

# Prepare A and B channels for PCA
a_flat = a_channel.astype(np.float32).reshape(-1, 1)
b_flat = b_channel.astype(np.float32).reshape(-1, 1)
color_features = np.hstack((a_flat, b_flat))

# PCA Compute
mean, eigenvectors, eigenvalues = cv2.PCACompute2(color_features, mean=None)
projected = cv2.PCAProject(color_features, mean, eigenvectors)
pc1 = projected[:, 0].reshape(a_channel.shape)

# Polarity Correction
correlation = np.corrcoef(pc1.ravel(), a_channel.ravel())[0, 1]
if correlation > 0:
    pc1 = -pc1

# Scale to 0-255 range
pc1_min = pc1.min()
pc1_max = pc1.max()
pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))
print("PC1 min/max:", pc1_scaled.min(), pc1_scaled.max())

# Threshold
_, thresh = cv2.threshold(pc1_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Threshold mask non-zero pixels:", np.count_nonzero(thresh))

# Clean
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
print("Cleaned mask non-zero pixels:", np.count_nonzero(cleaned))
