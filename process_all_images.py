import cv2
import numpy as np
import glob
import os

# Find all JPG images in the leaf-image directory recursively
# Matches: leaf-image/veg_1/*.JPG, leaf-image/veg_2/*.JPG, etc.
image_paths = sorted(glob.glob("leaf-image/**/*.JPG", recursive=True))

print(f"Found {len(image_paths)} images to process.\n")
print(f"{'Folder':<10} | {'Image Name':<15} | {'Leaf Area (px)':<15} | {'Mean PC1 (intensity)':<20}")
print("-" * 68)

for img_path in image_paths:
    # 1. Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        continue
        
    # 2. Convert BGR to Lab
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # 3. Prepare A and B channels for PCA color features
    a_flat = a_channel.astype(np.float32).reshape(-1, 1)
    b_flat = b_channel.astype(np.float32).reshape(-1, 1)
    color_features = np.hstack((a_flat, b_flat))
    
    # 4. Perform PCA to isolate the axis of maximum color variation (PC1)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(color_features, mean=None)
    projected = cv2.PCAProject(color_features, mean, eigenvectors)
    pc1 = projected[:, 0].reshape(a_channel.shape)
    
    # 5. Polarity correction: green vegetation has low values in a_channel
    correlation = np.corrcoef(pc1.ravel(), a_channel.ravel())[0, 1]
    if correlation > 0:
        pc1 = -pc1
        
    # 6. Scale PC1 to 0-255 range for consistent thresholding
    pc1_min = pc1.min()
    pc1_max = pc1.max()
    pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))
    
    # 7. Perform Otsu's thresholding to isolate leaf pixels
    ret, thresh = cv2.threshold(pc1_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 8. Clean up noise using a 5x5 elliptical kernel (Opening + Closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # 9. Calculate metrics:
    # - Leaf Area: Count of non-zero pixels in the cleaned mask
    leaf_area = np.count_nonzero(cleaned)
    
    # - Mean PC1 intensity of only the leaf pixels
    if leaf_area > 0:
        mean_pc1 = np.mean(pc1_scaled[cleaned == 255])
    else:
        mean_pc1 = 0.0
        
    # Get folder name (e.g. veg_1) and file name (e.g. IMG_7993.JPG)
    parts = img_path.split(os.sep)
    folder = parts[-2] if len(parts) > 1 else "root"
    filename = parts[-1]
    
    print(f"{folder:<10} | {filename:<15} | {leaf_area:<15,} | {mean_pc1:<20.2f}")
