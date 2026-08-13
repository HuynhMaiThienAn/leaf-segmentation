"""
Marker Calibration Module
=========================
Detects standard blue reference markers (400 mm²) in plant imagery to calculate exact mm²/pixel scaling factors.
"""

import cv2


def calibrate_marker(img_bgr, marker_area_mm2=400.0, default_scale_mm2_per_px=0.05):
    """
    Detects blue square calibration marker (400 mm²) or returns default fallback scale factor (mm²/pixel).
    
    Returns:
        scale (float): Area conversion scale factor (mm² / pixel)
        bbox (tuple or None): Bounding box (x, y, w, h) of detected marker
        status (str): 'detected' if marker found, else 'fallback'
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (95, 80, 80), (130, 255, 255))
    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (200 < area < 50000):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / (h + 1e-8)
        if 0.7 <= aspect <= 1.3:  # Square target marker
            return marker_area_mm2 / area, (x, y, w, h), "detected"

    return default_scale_mm2_per_px, None, "fallback"
