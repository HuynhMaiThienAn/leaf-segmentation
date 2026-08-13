"""
Post-processing Module
======================
Applies morphological operations and geometric contour area/aspect filtering.
"""

import cv2
import numpy as np


def apply_morphological_cleanup(thresh, kernel_size=3):
    """
    Applies morphological closing first to join seed fragments, followed by a gentle opening
    to preserve tiny early-stage cotyledons while removing single-pixel noise.
    """
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_k)
    return cleaned


def marker_controlled_watershed_segmentation(binary_mask, img_bgr, a_star=None, expected_plants=5, min_area=15, temporal_anchors=None, boost_plants=None, zone_color_profiles=None):
    """
    Applies Marker-Controlled Watershed Segmentation using Euclidean Distance-Transform Local Maxima
    or Dynamic Temporal Pot-Hole Anchors.
    Prioritizes organic plant canopy centers over neutral PVC pipe edge reflections.
    When zone_color_profiles is provided (Pass 2), applies per-zone learned LAB color filtering
    to adaptively reject non-plant artifacts.
    """
    if not np.any(binary_mask):
        return binary_mask.copy(), [], []

    img_h, img_w = binary_mask.shape[:2]
    dist = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Pre-compute LAB channels if color profile filtering is active
    lab_img = None
    if zone_color_profiles:
        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # If temporal anchors are provided, perform anchored seed growing
    if temporal_anchors and len(temporal_anchors) > 0:
        markers = np.zeros((img_h, img_w), dtype=np.int32)
        base_search_r = int(min(img_h, img_w) * 0.05)
        
        # Build 2D Voronoi Spatial Domains for the 5 stationary pot centers
        anchors_arr = np.array(temporal_anchors, dtype=np.float32)
        ys, xs = np.indices((img_h, img_w))
        coords = np.stack([xs, ys], axis=-1).astype(np.float32)
        dists_to_anchors = np.linalg.norm(coords[:, :, None, :] - anchors_arr[None, None, :, :], axis=-1)
        voronoi_map = np.argmin(dists_to_anchors, axis=-1) + 1  # 1-indexed (1..5)

        boost_set = set(boost_plants) if boost_plants else set()

        for idx, (ax, ay) in enumerate(temporal_anchors, start=1):
            # Locate the foliage pixel closest to stationary pot center (ax, ay) within pot ROI
            search_r = int(min(img_h, img_w) * 0.22)
            y1, y2 = max(0, ay - search_r), min(img_h, ay + search_r + 1)
            x1, x2 = max(0, ax - search_r), min(img_w, ax + search_r + 1)
            
            roi_mask = binary_mask[y1:y2, x1:x2]
            foliage_ys, foliage_xs = np.where(roi_mask > 0)
            
            r = 6 if idx in boost_set else 4
            if len(foliage_ys) > 0:
                abs_ys = foliage_ys + y1
                abs_xs = foliage_xs + x1
                dists = np.hypot(abs_xs - ax, abs_ys - ay)
                best_i = np.argmin(dists)
                seed_x, seed_y = int(abs_xs[best_i]), int(abs_ys[best_i])
                cv2.circle(markers, (seed_x, seed_y), r, idx, -1)
            else:
                cv2.circle(markers, (ax, ay), r, idx, -1)

        markers[binary_mask == 0] = 0
        ws_markers = markers.copy()

        # Build 3-channel Distance Transform topography to guide Watershed along canopy seams
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        max_dt = np.max(dist_transform) if np.any(binary_mask) else 1.0
        # Invert depth map so canopy centers are deep basins and thin leaf seams are high ridges
        dt_topo = (255.0 * (1.0 - dist_transform / max(max_dt, 1.0))).astype(np.uint8)
        dt_topo_3ch = cv2.merge([dt_topo, dt_topo, dt_topo])

        cv2.watershed(dt_topo_3ch, ws_markers)

        cleaned_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        valid_cnts = []
        plant_submasks = []
        effective_min_area = 5 if temporal_anchors else min_area

        # Pre-compute Voronoi pot assignments for all binary_mask pixels to ensure disconnected cotyledons are never lost
        anchors_arr = np.array(temporal_anchors, dtype=np.float32)
        ys_b, xs_b = np.where(binary_mask > 0)
        voronoi_submasks = {i: np.zeros((img_h, img_w), dtype=np.uint8) for i in range(1, len(temporal_anchors) + 1)}
        if len(ys_b) > 0:
            coords_b = np.stack([xs_b, ys_b], axis=-1).astype(np.float32)
            dists_b = np.linalg.norm(coords_b[:, None, :] - anchors_arr[None, :, :], axis=-1)
            closest_pots = np.argmin(dists_b, axis=-1) + 1
            min_dists_b = np.min(dists_b, axis=-1)
            # Restrict Voronoi rescue radius to 11% (53px) so disconnected seedling cotyledons are rescued near pot center without distorting mature canopy boundaries
            max_r = min(img_h, img_w) * 0.11
            for px_i in range(len(ys_b)):
                if min_dists_b[px_i] <= max_r:
                    voronoi_submasks[closest_pots[px_i]][ys_b[px_i], xs_b[px_i]] = 255

        for idx in range(1, len(temporal_anchors) + 1):
            ax, ay = temporal_anchors[idx - 1]
            ws_sub = ((ws_markers == idx) & (binary_mask > 0)).astype(np.uint8) * 255
            v_sub = voronoi_submasks.get(idx, np.zeros((img_h, img_w), dtype=np.uint8))
            
            # Combine Watershed with Voronoi proximity submask to catch disconnected cotyledons
            raw_submask = cv2.bitwise_or(ws_sub, v_sub)

            # Pass 2: Apply per-zone learned LAB color profile filtering with robust tolerance
            if zone_color_profiles and idx in zone_color_profiles and lab_img is not None:
                profile = zone_color_profiles[idx]
                k = 3.5  # Relaxed multiplier for broad foliage preservation
                zone_pixels = raw_submask > 0
                if np.any(zone_pixels):
                    L_vals = lab_img[:, :, 0]
                    A_vals = lab_img[:, :, 1]
                    B_vals = lab_img[:, :, 2]
                    chroma_vals = np.hypot(A_vals - 128.0, B_vals - 128.0)
                    color_ok = (
                        (chroma_vals > 3.2) &
                        (np.abs(L_vals - profile["mean_L"]) < k * max(profile["std_L"], 10.0)) &
                        (np.abs(A_vals - profile["mean_a"]) < k * max(profile["std_a"], 6.0)) &
                        (np.abs(B_vals - profile["mean_b"]) < k * max(profile["std_b"], 6.0))
                    )
                    raw_submask = (zone_pixels & color_ok).astype(np.uint8) * 255

            p_cnts, _ = cv2.findContours(raw_submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            p_valid_cnts = []
            plant_submask = np.zeros((img_h, img_w), dtype=np.uint8)

            for c in p_cnts:
                c_area = cv2.contourArea(c)
                if c_area < effective_min_area:
                    continue

                # Compute centroid of candidate foliage contour
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx, cy = float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
                else:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cx, cy = float(bx + bw / 2.0), float(by + bh / 2.0)

                # Distance to this plant's pot anchor vs closest pot anchor
                d_anchor = np.hypot(cx - ax, cy - ay)
                all_anchor_dists = [np.hypot(cx - a_x, cy - a_y) for a_x, a_y in temporal_anchors]
                d_min_anchor = min(all_anchor_dists)

                # Adaptive Proximity Gate: Only tiny noise specks (area < 120px) are restricted to 78px; real leaf lobes expand freely
                max_allowed_r = 78.0 if c_area < 120 else min(img_h, img_w) * 0.32

                if d_anchor <= max(d_min_anchor * 1.35, d_min_anchor + 40.0) and d_anchor <= max_allowed_r:
                    cv2.drawContours(cleaned_mask, [c], -1, 255, -1)
                    cv2.drawContours(plant_submask, [c], -1, 255, -1)
                    valid_cnts.append(c)
                    p_valid_cnts.append(c)

            plant_submasks.append({"plant_id": f"Plant_{idx:02d}", "mask": plant_submask, "contours": p_valid_cnts})

        return cleaned_mask, valid_cnts, plant_submasks

    # 1. Find local distance peaks (minimum 2.0px depth)
    k_size = max(11, int(min(img_h, img_w) * 0.03))
    if k_size % 2 == 0:
        k_size += 1
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dist_dilated = cv2.dilate(dist, peak_kernel)

    min_peak_val = 2.0
    local_max = (dist == dist_dilated) & (dist >= min_peak_val)

    # 2. Extract seed coordinates & compute Organic Priority Score
    cnts, _ = cv2.findContours(local_max.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
        
        # Restrict seed search to interior white growing tray (0.12W <= cx <= 0.88W, 0.08H <= cy <= 0.92H)
        # Prunes green PVC water tubes on far-left (x < 0.10W) and hardware fixtures on far-right (x > 0.90W)
        if 0.12 * img_w <= cx <= 0.88 * img_w and 0.08 * img_h <= cy <= 0.92 * img_h:
            peak_val = float(dist[cy, cx])
            
            # Organic Priority Score & Color Validation
            score = peak_val
            if a_star is not None:
                # Sample local a* value around (cx, cy)
                y1, y2 = max(0, cy - 3), min(img_h, cy + 4)
                x1, x2 = max(0, cx - 3), min(img_w, cx + 4)
                local_a = np.mean(a_star[y1:y2, x1:x2])
                
                # Reject neutral grey/white/blue PVC reflections (-1.0 < local_a < 1.8)
                if -1.0 <= local_a <= 1.8:
                    continue

                # Green foliage (negative a*) and Anthocyanin red (a* > 1.8) get priority boost
                green_boost = max(0.0, -local_a * 1.5)
                red_boost = max(0.0, (local_a - 1.8) * 1.5)
                score = peak_val * (2.0 + green_boost + red_boost)

            seeds.append((score, peak_val, cx, cy))

    # Sort seeds by Organic Priority Score descending
    seeds.sort(key=lambda s: s[0], reverse=True)

    # Non-maximum suppression to filter close seeds
    filtered_seeds = []
    min_gap_px = int(min(img_h, img_w) * 0.08)
    for score, p_val, cx, cy in seeds:
        too_close = False
        for _, _, fcx, fcy in filtered_seeds:
            if np.hypot(cx - fcx, cy - fcy) < min_gap_px:
                too_close = True
                break
        if not too_close:
            filtered_seeds.append((score, p_val, cx, cy))

    if expected_plants is not None and expected_plants > 0 and len(filtered_seeds) > expected_plants:
        filtered_seeds = filtered_seeds[:expected_plants]

    if not filtered_seeds:
        return separate_foliage_from_line_artifacts(binary_mask, min_area=min_area), [], []

    # 3. Create seed markers
    markers = np.zeros((img_h, img_w), dtype=np.int32)
    for idx, (score, p_val, cx, cy) in enumerate(filtered_seeds, start=1):
        r = max(4, int(p_val * 0.45))
        cv2.circle(markers, (cx, cy), r, idx, -1)

    markers[binary_mask == 0] = 0

    # 4. Perform Watershed
    ws_markers = markers.copy()
    cv2.watershed(img_bgr, ws_markers)

    # 5. Extract foliage mask from valid plant seeds
    cleaned_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    valid_cnts = []
    plant_submasks = []
    for idx in range(1, len(filtered_seeds) + 1):
        plant_submask = ((ws_markers == idx) & (binary_mask > 0)).astype(np.uint8) * 255
        p_cnts, _ = cv2.findContours(plant_submask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_valid_cnts = []
        for c in p_cnts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(cleaned_mask, [c], -1, 255, -1)
                valid_cnts.append(c)
                p_valid_cnts.append(c)
        plant_submasks.append({"plant_id": f"Plant_{idx:02d}", "mask": plant_submask, "contours": p_valid_cnts})

    return cleaned_mask, valid_cnts, plant_submasks


def separate_foliage_from_line_artifacts(binary_mask, min_area=15):
    """
    Dynamically decouples organic plant foliage from linear PVC rails and border artifacts
    using Euclidean Distance-Transform Peak Reconstruction and Compactness-Solidity filtering.
    Fully coordinate-free and scale-invariant across all camera angles and tray layouts.
    """
    if not np.any(binary_mask):
        return binary_mask.copy()

    img_h, img_w = binary_mask.shape[:2]
    dist = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)
    
    cnts, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / (hull_area + 1e-8)
        
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / (h + 1e-8)

        # Check border margin proximity (outer 3% margin)
        near_margin = (x < 0.03 * img_w) or ((x + w) > 0.97 * img_w) or (y < 0.03 * img_h) or ((y + h) > 0.97 * img_h)

        # Prune thin border edge hardware reflections and 1D degenerate lines
        if near_margin and (w < 0.035 * img_w or h < 0.035 * img_h or aspect > 2.2 or aspect < 0.45 or solidity > 1.02):
            continue

        # Check if contour is clean organic plant canopy inside growth area
        if solidity >= 0.35 and 0.25 <= aspect <= 3.5 and not (near_margin and solidity > 0.98):
            cv2.drawContours(cleaned_mask, [c], -1, 255, -1)
        else:
            # Low solidity or elongated contour: Apply Distance Transform Peak Reconstruction
            c_roi_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            cv2.drawContours(c_roi_mask, [c], -1, 255, -1)
            
            c_dist = dist * (c_roi_mask > 0)
            max_d = np.max(c_dist)
            
            # Scale-aware minimum peak distance (small cotyledons have peak ~1.5 - 2.5px)
            min_peak_req = 1.5 if area < 350 else 3.0
            if max_d < min_peak_req:
                # Thin 1D PVC line without deep blob center -> Discard
                continue

            # Core seed threshold
            seeds = (c_dist >= max(min_peak_req, 0.25 * max_d)).astype(np.uint8)
            
            # Reconstruct plant canopy by expanding seeds back within c_roi_mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            iterations = max(1, int(max_d * 0.8))
            reconstructed = cv2.dilate(seeds, kernel, iterations=iterations)
            reconstructed = cv2.bitwise_and(reconstructed, c_roi_mask)

            # Keep reconstructed sub-contours that meet plant solidity/area standards
            sub_cnts, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for sub_c in sub_cnts:
                sub_area = cv2.contourArea(sub_c)
                if sub_area >= min_area:
                    sub_hull_area = cv2.contourArea(cv2.convexHull(sub_c))
                    sub_solidity = sub_area / (sub_hull_area + 1e-8)
                    sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(sub_c)
                    sub_near = (sub_x < 0.06 * img_w) or ((sub_x + sub_w) > 0.94 * img_w) or (sub_y < 0.05 * img_h) or ((sub_y + sub_h) > 0.95 * img_h)

                    if sub_solidity >= 0.45 and sub_solidity <= 1.02 and not (sub_near and (sub_w < 0.035 * img_w or sub_h < 0.035 * img_h)):
                        cv2.drawContours(cleaned_mask, [sub_c], -1, 255, -1)

    return cleaned_mask


def filter_contours_by_geometry(cleaned, img_shape, min_area=15, max_area_ratio=0.99, aspect_range=(0.20, 4.0), a_star=None, img_bgr=None, expected_plants=5, temporal_anchors=None, boost_plants=None, zone_color_profiles=None):
    """
    Filters external contours by minimum pixel area, bounded aspect ratio, linear artifact suppression,
    and Marker-Controlled Watershed Segmentation using Distance Transform Local Maxima or Temporal Anchors.
    """
    img_h, img_w = img_shape[:2]

    # 1. Apply Marker-Controlled Watershed if BGR image is available for seed-driven segmentation
    plant_submasks = []
    if img_bgr is not None and np.any(cleaned):
        cleaned, ws_cnts, plant_submasks = marker_controlled_watershed_segmentation(
            cleaned, img_bgr, a_star=a_star, expected_plants=expected_plants, min_area=min_area, temporal_anchors=temporal_anchors, boost_plants=boost_plants, zone_color_profiles=zone_color_profiles
        )
    else:
        cleaned = separate_foliage_from_line_artifacts(cleaned, min_area=min_area)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    roi_mask = np.zeros_like(cleaned)
    valid_cnts = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if max_area_ratio is not None and area > max_area_ratio * (img_h * img_w):
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        aspect = float(w) / (h + 1e-8)
        
        if aspect < aspect_range[0] or aspect > aspect_range[1]:
            continue
            
        # Compute contour solidity (area / convex hull area)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)

        # Enforce minimum solidity for foliage canopy (allow deeply lobed or small young leaves)
        min_solidity_req = 0.25 if area < 400 else 0.30
        if solidity < min_solidity_req:
            continue

        # Suppress red/orange PVC hardware joint pieces near outer border margins
        near_left_right = (x < 0.08 * img_w) or ((x + w) > 0.92 * img_w)
        near_top_bottom = (y < 0.08 * img_h) or ((y + h) > 0.92 * img_h)

        if a_star is not None and (near_left_right or near_top_bottom):
            c_mask = np.zeros_like(cleaned)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            mean_a = np.mean(a_star[c_mask > 0])
            if mean_a > 4.5:  # High-chroma red/orange PVC fitting or clip
                continue

        cv2.drawContours(roi_mask, [c], -1, 255, -1)
        valid_cnts.append(c)
        
    filtered_mask = cv2.bitwise_and(cleaned, roi_mask)
    return filtered_mask, valid_cnts, plant_submasks


def clean_submasks(green_mask, red_mask, min_submask_area=25):
    """
    Applies morphological opening and area thresholding to green and red sub-masks after
    foliage splitting to remove isolated noise specks.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    def _clean_mask(mask):
        if not np.any(mask):
            return mask
        cleaned_m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(cleaned_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) >= min_submask_area:
                cv2.drawContours(filtered, [c], -1, 255, -1)
        return cv2.bitwise_and(filtered, mask)

    return _clean_mask(green_mask), _clean_mask(red_mask)
