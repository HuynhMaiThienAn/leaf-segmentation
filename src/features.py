"""
Color Feature Extraction Module
================================
Extracts CIELAB color channels (L*, a*, b*), Chroma (C*), and Hue angle degrees from input BGR imagery.
"""

import cv2
import numpy as np


def extract_cielab_features(img_bgr):
    """
    Converts input BGR image to CIELAB color space, extracts L*, a*, b* channels,
    computes Chroma (C* = sqrt(a*^2 + b*^2)), and calculates Hue in degrees [0, 360).
    
    Returns:
        l_ch (np.ndarray): Lightness channel (0-255 uint8)
        a_star (np.ndarray): Float32 a* channel (-128 to 127)
        b_star (np.ndarray): Float32 b* channel (-128 to 127)
        chroma (np.ndarray): Float32 Chroma magnitude map
        hue_deg (np.ndarray): Float32 Hue angle map in degrees [0, 360)
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    
    a_star = a_ch.astype(np.float32) - 128.0
    b_star = b_ch.astype(np.float32) - 128.0
    chroma = np.sqrt(a_star**2 + b_star**2)
    hue_deg = np.degrees(np.arctan2(b_star, a_star)) % 360.0
    
    return l_ch, a_star, b_star, chroma, hue_deg


def extract_multi_color_pca_features(img_bgr):
    """
    Fuses BGR, CIELAB, and HSV color spaces into a 9D per-pixel feature vector,
    standardizes features, and projects them onto the 1st Principal Component (PC1).
    """
    h, w, _ = img_bgr.shape
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    a_star = a_ch.astype(np.float32) - 128.0
    b_star = b_ch.astype(np.float32) - 128.0
    chroma = np.sqrt(a_star**2 + b_star**2)
    hue_deg = np.degrees(np.arctan2(b_star, a_star)) % 360.0

    features = np.hstack([
        img_bgr.reshape(-1, 3).astype(np.float32),
        img_lab.reshape(-1, 3).astype(np.float32),
        img_hsv.reshape(-1, 3).astype(np.float32)
    ])

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    features_norm = (features - mean) / std

    cov = np.cov(features_norm, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    
    pc1_vec = evecs[:, -1]
    pc1_map = np.dot(features_norm, pc1_vec).reshape(h, w)

    if np.corrcoef(pc1_map.flatten(), chroma.flatten())[0, 1] < 0:
        pc1_map = -pc1_map

    pc1_norm = cv2.normalize(pc1_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return pc1_norm, pc1_map, l_ch, chroma, hue_deg
