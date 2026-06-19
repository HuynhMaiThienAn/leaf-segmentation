import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Paths (adjust if necessary)
    img_path = os.path.join("leaf-image", "veg_1", "IMG_7993.JPG")
    
    # 1. Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # ------------------ Plot 1: Color Space PCA ------------------
    print("Computing Color Space PCA...")
    a_flat = a_channel.astype(np.float32).reshape(-1, 1)
    b_flat = b_channel.astype(np.float32).reshape(-1, 1)
    color_features = np.hstack((a_flat, b_flat))
    
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(color_features, mean=None)
    
    # Sample pixels to keep scatter plot fast and clean
    sample_rate = 200
    a_sample = a_channel.ravel()[::sample_rate]
    b_sample = b_channel.ravel()[::sample_rate]
    
    plt.figure(figsize=(8, 7))
    plt.scatter(a_sample, b_sample, alpha=0.15, c='forestgreen', edgecolors='none', label='Pixels (Sampled)')
    
    mean_a, mean_b = mean[0, 0], mean[0, 1]
    plt.scatter(mean_a, mean_b, color='red', s=120, zorder=5, label='Mean Color Centroid')
    
    # Scale PC axes by standard deviations (sqrt of eigenvalues)
    scale_pc1 = 2.5 * np.sqrt(eigenvalues[0, 0])
    scale_pc2 = 2.5 * np.sqrt(eigenvalues[1, 0])
    
    # Draw PC vectors (arrow from mean)
    plt.arrow(mean_a, mean_b, eigenvectors[0, 0] * scale_pc1, eigenvectors[0, 1] * scale_pc1,
              color='red', width=1.5, head_width=5, label='PC1 (Primary Variation)', zorder=6)
    plt.arrow(mean_a, mean_b, eigenvectors[1, 0] * scale_pc2, eigenvectors[1, 1] * scale_pc2,
              color='blue', width=1.5, head_width=5, label='PC2', zorder=6)
              
    plt.title('PCA of CIELAB Color Space (a* vs b*)')
    plt.xlabel('a* Channel (Green to Red)')
    plt.ylabel('b* Channel (Blue to Yellow)')
    plt.axvline(128, color='gray', linestyle='--', alpha=0.4)
    plt.axhline(128, color='gray', linestyle='--', alpha=0.4)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    color_plot_path = "color_pca.png"
    plt.savefig(color_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {color_plot_path}")
    
    # ------------------ Plot 2: Spatial PCA on Plant pixels ------------------
    print("Computing Spatial PCA on largest plant contour...")
    projected = cv2.PCAProject(color_features, mean, eigenvectors)
    pc1 = projected[:, 0].reshape(a_channel.shape)
    
    # Polarity correction
    correlation = np.corrcoef(pc1.ravel(), a_channel.ravel())[0, 1]
    if correlation > 0:
        pc1 = -pc1
        
    # Scale PC1 to 0-255
    pc1_min, pc1_max = pc1.min(), pc1.max()
    pc1_scaled = np.uint8(255 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-8))
    
    # Threshold
    _, thresh = cv2.threshold(pc1_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours geometrically (discard long lines / channel edges)
    plant_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if w > 0.8 * img.shape[1] or aspect_ratio > 4.0 or aspect_ratio < 0.25:
                continue
            plant_contours.append(cnt)
            
    if not plant_contours:
        print("No plant contours found after filtering.")
        return
        
    largest_contour = max(plant_contours, key=cv2.contourArea)
    
    # Extract coordinates
    mask = np.zeros(cleaned.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    pts = np.column_stack(np.where(mask == 255))
    pts = pts[:, [1, 0]].astype(np.float32) # swap to (x, y)
    
    # Run 2D spatial PCA
    s_mean, s_eigenvectors, s_eigenvalues = cv2.PCACompute2(pts, mean=None)
    cx, cy = s_mean[0, 0], s_mean[0, 1]
    
    plt.figure(figsize=(7, 7))
    plt.scatter(pts[:, 0], pts[:, 1], s=1, color='forestgreen', alpha=0.3, label='Segmented Leaf Pixels')
    plt.scatter(cx, cy, color='gold', s=100, edgecolors='black', zorder=5, label='Plant Centroid')
    
    # Scale axes (draw ±2 std deviations)
    len_pc1 = 2.0 * np.sqrt(s_eigenvalues[0, 0])
    len_pc2 = 2.0 * np.sqrt(s_eigenvalues[1, 0])
    
    plt.plot([cx - s_eigenvectors[0, 0]*len_pc1, cx + s_eigenvectors[0, 0]*len_pc1],
             [cy - s_eigenvectors[0, 1]*len_pc1, cy + s_eigenvectors[0, 1]*len_pc1],
             color='red', linewidth=3, label='Major Axis (Length PC1)')
    plt.plot([cx - s_eigenvectors[1, 0]*len_pc2, cx + s_eigenvectors[1, 0]*len_pc2],
             [cy - s_eigenvectors[1, 1]*len_pc2, cy + s_eigenvectors[1, 1]*len_pc2],
             color='blue', linewidth=3, label='Minor Axis (Width PC2)')
             
    plt.gca().invert_yaxis() # Invert y-axis to match image/pixel space coordinates
    plt.title('Spatial PCA of Plant Shape Coordinates')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    spatial_plot_path = "spatial_pca.png"
    plt.savefig(spatial_plot_path, dpi=150)
    plt.close()
    print(f"Saved: {spatial_plot_path}")
    print("Done! Both plots generated successfully.")

if __name__ == "__main__":
    main()
