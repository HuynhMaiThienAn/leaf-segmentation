"""
Adaptive B0 Foliage Segmentation Pipeline
"""

import cv2
import numpy as np

from src.features import extract_cielab_features
from src.gating import create_lightness_and_border_gates, create_color_gates, estimate_dynamic_parameters
from src.thresholding import compute_chroma_otsu_threshold
from src.postprocessing import apply_morphological_cleanup, filter_contours_by_geometry, clean_submasks


def run_adaptive_b0_pipeline(
    img_bgr,
    l_range=(15, 235),
    chroma_min=None,
    green_hue=(80.0, 170.0),
    red_hue=((0.0, 65.0), (280.0, 360.0)),
    min_area=None,
    kernel_size=None,
    border_margin=0.06,
    target_color="auto",
    expected_plants=5,
    temporal_anchors=None,
    boost_plants=None,
    zone_color_profiles=None,
):
    """
    Runs Adaptive B0 Dual-Color Foliage Segmentation Engine with Marker-Controlled Watershed.
    """
    l_ch, a_star, b_star, chroma, hue_deg = extract_cielab_features(img_bgr)
    chroma_min, min_area, kernel_size = estimate_dynamic_parameters(
        img_bgr.shape, chroma, l_ch, chroma_min=chroma_min, min_area=min_area, kernel_size=kernel_size
    )

    l_gate, border_gate = create_lightness_and_border_gates(l_ch, img_bgr.shape, l_range=l_range, border_margin=border_margin)
    chroma_gate = chroma > chroma_min
    green_hue_gate, red_hue_gate, color_gate = create_color_gates(
        hue_deg, green_hue=green_hue, red_hue=red_hue, a_star=a_star, chroma=chroma, target_color=target_color
    )

    combined_gate = l_gate & chroma_gate & color_gate & border_gate

    # In Pass 2: Inject learned per-zone color profiles into initial threshold gate to rescue early-stage cotyledons
    if zone_color_profiles and temporal_anchors:
        img_h, img_w = img_bgr.shape[:2]
        learned_gate = np.zeros((img_h, img_w), dtype=bool)
        for idx, (ax, ay) in enumerate(temporal_anchors, start=1):
            if idx in zone_color_profiles:
                prof = zone_color_profiles[idx]
                search_r = int(min(img_h, img_w) * 0.28)
                y1, y2 = max(0, ay - search_r), min(img_h, ay + search_r + 1)
                x1, x2 = max(0, ax - search_r), min(img_w, ax + search_r + 1)
                
                roi_L = l_ch[y1:y2, x1:x2]
                roi_a = a_star[y1:y2, x1:x2]
                roi_b = b_star[y1:y2, x1:x2]
                roi_chroma = chroma[y1:y2, x1:x2]
                
                match = (
                    (roi_chroma > 3.2) &
                    (np.abs(roi_L - prof["mean_L"]) < 3.5 * max(prof["std_L"], 10.0)) &
                    (np.abs(roi_a - prof["mean_a"]) < 3.5 * max(prof["std_a"], 6.0)) &
                    (np.abs(roi_b - prof["mean_b"]) < 3.5 * max(prof["std_b"], 6.0))
                )
                learned_gate[y1:y2, x1:x2] |= match
                
        combined_gate |= (learned_gate & l_gate & border_gate)

    chroma_scaled, thresh, otsu_v = compute_chroma_otsu_threshold(chroma, combined_gate)

    if zone_color_profiles and temporal_anchors:
        # Pass 2: Directly include learned foliage pixels near predicted pot locations into binary thresh
        learned_foliage_mask = (learned_gate & l_gate & border_gate).astype(np.uint8) * 255
        thresh = cv2.bitwise_or(thresh, learned_foliage_mask)

    cleaned = apply_morphological_cleanup(thresh, kernel_size=kernel_size)
    filtered_mask, valid_cnts, plant_submasks = filter_contours_by_geometry(
        cleaned, img_bgr.shape, min_area=min_area, a_star=a_star, img_bgr=img_bgr, expected_plants=expected_plants, temporal_anchors=temporal_anchors, boost_plants=boost_plants, zone_color_profiles=zone_color_profiles
    )

    red_hue_mask = red_hue_gate | (a_star > 1.5)
    raw_red = np.where((filtered_mask > 0) & red_hue_mask, 255, 0).astype(np.uint8)
    raw_green = np.where((filtered_mask > 0) & ~red_hue_mask, 255, 0).astype(np.uint8)
    green_mask, red_mask = clean_submasks(raw_green, raw_red, min_submask_area=25)

    return {
        "l_ch": l_ch,
        "chroma": chroma,
        "l_gate": l_gate,
        "color_gate": color_gate,
        "combined_gate": combined_gate,
        "chroma_scaled": chroma_scaled,
        "thresh": thresh,
        "cleaned": cleaned,
        "foliage_mask": filtered_mask,
        "green_mask": green_mask,
        "red_mask": red_mask,
        "valid_contours": valid_cnts,
        "plant_submasks": plant_submasks,
        "otsu_val": otsu_v,
    }
