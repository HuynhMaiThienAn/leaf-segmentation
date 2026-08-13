"""
Chroma Thresholding Module
===========================
Applies multi-gate fusion, P99 percentile chroma scaling, and Otsu binary thresholding.
"""

import cv2
import numpy as np


def compute_chroma_otsu_threshold(chroma, combined_gate):
    """
    Applies multi-gate fusion to chroma, scales positive values to P99 percentile [0, 255],
    and runs Otsu binary thresholding.
    """
    img_h, img_w = chroma.shape[:2]
    chroma_gated = np.where(combined_gate, chroma, 0.0)
    
    pos = chroma_gated[chroma_gated > 0]
    if len(pos) > 0:
        p99 = np.percentile(pos, 99.0)
        chroma_scaled = ((np.clip(chroma_gated, 0, p99) / (p99 + 1e-8)) * 255.0).astype(np.uint8)
        otsu_v, thresh = cv2.threshold(chroma_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Cap Otsu threshold to prevent high-chroma leaves from inflating threshold and erasing paler leaves
        if otsu_v > 110.0:
            otsu_v = 110.0
            _, thresh = cv2.threshold(chroma_scaled, int(otsu_v), 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, (combined_gate * 255).astype(np.uint8))
    else:
        chroma_scaled = np.zeros((img_h, img_w), dtype=np.uint8)
        thresh = np.zeros((img_h, img_w), dtype=np.uint8)
        otsu_v = 0.0
        
    return chroma_scaled, thresh, otsu_v
