"""
Gating Rules Module
====================
Defines lightness, border margin, and symmetric green & red foliage hue gates for plant foliage isolation.
"""

import numpy as np


def create_lightness_and_border_gates(l_ch, img_shape, l_range=(15, 235), border_margin=0.04):
    """
    Creates boolean lightness gate (15 <= L* <= 235) and canvas border margin gate.
    Uses conservative border margin (4%) to ensure edge-hole leaves are fully preserved.
    """
    img_h, img_w = img_shape[:2]
    l_gate = (l_ch >= l_range[0]) & (l_ch <= l_range[1])
    
    if isinstance(border_margin, float) and border_margin < 1.0:
        margin_y = max(15, int(border_margin * img_h))
        margin_x = max(55, int(border_margin * img_w))
    else:
        margin_y = int(border_margin)
        margin_x = max(55, int(border_margin))

    border_gate = np.zeros((img_h, img_w), dtype=bool)
    border_gate[margin_y:img_h - margin_y, margin_x:img_w - margin_x] = True
    
    return l_gate, border_gate


def create_color_gates(hue_deg, green_hue=(80.0, 170.0), red_hue=((0.0, 50.0), (290.0, 360.0)), a_star=None, chroma=None, target_color="auto"):
    """
    Computes symmetric hue gates for green, red/anthocyanin, or auto foliage classes.
    """
    green_hue_gate = (hue_deg >= green_hue[0]) & (hue_deg <= green_hue[1])
    if a_star is not None:
        green_hue_gate &= (a_star < -2.0)
    
    red_hue_gate = np.zeros_like(green_hue_gate)
    for lo, hi in red_hue:
        red_hue_gate |= (hue_deg >= lo) & (hue_deg <= hi)
        
    if a_star is not None and chroma is not None:
        # High-chroma anthocyanin red/purple foliage check (prevents warm white tray reflections)
        red_hue_gate |= (a_star > 4.0) & (chroma > 12.0)
        
    if chroma is not None:
        red_hue_gate &= (chroma > 8.0)

    if target_color == "green":
        color_gate = green_hue_gate
    elif target_color == "red":
        color_gate = red_hue_gate
    else:
        color_gate = green_hue_gate | red_hue_gate

    return green_hue_gate, red_hue_gate, color_gate


def estimate_dynamic_parameters(img_shape, chroma, l_ch, chroma_min=None, min_area=None, kernel_size=None):
    """
    Dynamically calculates optimal segmentation parameters per image based on 
    pixel resolution and background chroma noise floor if not explicitly set.
    """
    img_h, img_w = img_shape[:2]
    total_pixels = img_h * img_w
    
    if min_area is None:
        min_area = max(15, int(0.00001 * total_pixels))
        
    if kernel_size is None:
        max_dim = max(img_h, img_w)
        kernel_size = max(3, int(max_dim / 800) | 1)
        
    if chroma_min is None:
        valid_l = (l_ch >= 15) & (l_ch <= 235)
        chroma_valid = chroma[valid_l]
        if len(chroma_valid) > 0:
            noise_floor = float(np.percentile(chroma_valid, 20.0))
            chroma_min = float(np.clip(noise_floor, 4.0, 12.0))
        else:
            chroma_min = 5.0
            
    return chroma_min, min_area, kernel_size
