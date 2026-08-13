"""
Phenotypic Metrics Module
==========================
Computes the 7 Core Phenotypic Measurements for overall canopy AND per-plant instances:
1. PLA (mm²): Projected plant area
2. Width (mm): Projected canopy width
3. Height (mm): Projected canopy height
4. Convex hull area (mm²): Overall projected envelope
5. Solidity: Shape filling / fragmentation ratio (PLA / Convex Hull Area)
6. PCA orientation (deg): Projected canopy principal axis angle (0° to 180°)
7. PCA aspect ratio: Principal component shape elongation ratio √(λ1 / λ2)
"""

import cv2
import numpy as np


PLANT_COLOR_PALETTE = [
    {"name": "Cyan", "bgr": (255, 255, 0), "hex": "#00FFFF"},         # Plant 01
    {"name": "Magenta", "bgr": (255, 0, 255), "hex": "#FF00FF"},      # Plant 02
    {"name": "Yellow", "bgr": (0, 230, 255), "hex": "#FFFF00"},       # Plant 03
    {"name": "Orange", "bgr": (0, 140, 255), "hex": "#FF8C00"},       # Plant 04
    {"name": "Lime", "bgr": (0, 255, 128), "hex": "#00FF80"},         # Plant 05
    {"name": "Blue", "bgr": (255, 100, 0), "hex": "#0064FF"},         # Plant 06
    {"name": "Purple", "bgr": (255, 0, 128), "hex": "#8000FF"},       # Plant 07
    {"name": "Coral", "bgr": (100, 100, 255), "hex": "#FF6464"},      # Plant 08
]


def compute_per_plant_metrics(seg_result, mm2_per_px, max_gap_px=65, expected_plants=None):
    """
    Extracts individual plant instances from validated contours or Watershed plant_submasks.
    Supports fixed tray-grid K-Means spatial clustering (expected_plants=5) for constant plant tracking.
    """
    mask = seg_result["foliage_mask"]
    valid_cnts = seg_result.get("valid_contours", [])
    green_mask = seg_result.get("green_mask", np.zeros_like(mask))
    red_mask = seg_result.get("red_mask", np.zeros_like(mask))
    plant_submasks = seg_result.get("plant_submasks", [])
    mm_per_px_linear = np.sqrt(mm2_per_px)

    if plant_submasks and len(plant_submasks) > 0:
        per_plant_list = []
        for p_idx, p_info in enumerate(plant_submasks, start=1):
            p_mask = p_info["mask"]
            p_id = p_info.get("plant_id", f"Plant_{p_idx:02d}")
            p_cnts = p_info.get("contours", [])

            leaf_px = np.count_nonzero(p_mask)
            if leaf_px == 0:
                continue

            pla_mm2 = leaf_px * mm2_per_px
            g_px = np.count_nonzero(cv2.bitwise_and(p_mask, green_mask))
            r_px = np.count_nonzero(cv2.bitwise_and(p_mask, red_mask))
            pla_green_mm2 = g_px * mm2_per_px
            pla_red_mm2 = r_px * mm2_per_px

            if not p_cnts:
                p_cnts, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not p_cnts:
                continue

            # Remove distant stray noise contours that artificially stretch the bounding box
            if len(p_cnts) > 1:
                p_cnts = sorted(p_cnts, key=cv2.contourArea, reverse=True)
                main_c = p_cnts[0]
                main_area = cv2.contourArea(main_c)
                
                # We will keep contours that are spatially close to the main contour
                # A fast proxy for contour distance is bounding box proximity
                main_x, main_y, main_w, main_h = cv2.boundingRect(main_c)
                main_cx, main_cy = main_x + main_w / 2.0, main_y + main_h / 2.0
                
                filtered_cnts = [main_c]
                for c in p_cnts[1:]:
                    c_area = cv2.contourArea(c)
                    cx, cy, cw, ch = cv2.boundingRect(c)
                    c_cx, c_cy = cx + cw / 2.0, cy + ch / 2.0
                    
                    # Center-to-center distance
                    dist = np.hypot(main_cx - c_cx, main_cy - c_cy)
                    
                    # Adaptive threshold: larger main bodies allow further disconnected lobes
                    # Base distance of 50px, plus half the main plant's max dimension
                    max_dist = 60.0 + 0.60 * max(main_w, main_h)
                    
                    if dist <= max_dist:
                        filtered_cnts.append(c)
                p_cnts = filtered_cnts

            # Update the mask to only contain the filtered contours
            p_mask_filtered = np.zeros_like(p_mask)
            cv2.drawContours(p_mask_filtered, p_cnts, -1, 255, -1)
            p_mask = cv2.bitwise_and(p_mask, p_mask_filtered)

            leaf_px = np.count_nonzero(p_mask)
            if leaf_px == 0:
                continue

            all_pts = np.vstack(p_cnts)
            x, y, w, h = cv2.boundingRect(all_pts)
            hull = cv2.convexHull(all_pts)
            hull_px = cv2.contourArea(hull)
            hull_mm2 = hull_px * mm2_per_px
            solidity = pla_mm2 / (hull_mm2 + 1e-8)

            w_mm = w * mm_per_px_linear
            h_mm = h * mm_per_px_linear

            pts = all_pts.reshape(-1, 2).astype(np.float32)
            if len(pts) >= 5:
                mean = np.mean(pts, axis=0)
                centered = pts - mean
                cov = np.cov(centered, rowvar=False)
                evals, evecs = np.linalg.eigh(cov)
                sort_indices = np.argsort(evals)[::-1]
                evals = evals[sort_indices]
                evecs = evecs[:, sort_indices]
                vx, vy = evecs[0, 0], evecs[1, 0]
                pca_angle = float(np.degrees(np.arctan2(vy, vx))) % 180.0
                pca_ar = float(np.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6)))
            else:
                pca_angle = 0.0
                pca_ar = 1.0

            color_info = PLANT_COLOR_PALETTE[(p_idx - 1) % len(PLANT_COLOR_PALETTE)]

            per_plant_list.append({
                "plant_id": p_id,
                "contours": p_cnts,
                "bbox": (x, y, w, h),
                "centroid": (x + w // 2, y + h // 2),
                "color_bgr": color_info["bgr"],
                "color_name": color_info["name"],
                "pla_mm2": pla_mm2,
                "pla_green_mm2": pla_green_mm2,
                "pla_red_mm2": pla_red_mm2,
                "width_mm": w_mm,
                "height_mm": h_mm,
                "hull_mm2": hull_mm2,
                "solidity": solidity,
                "pca_angle": pca_angle,
                "pca_aspect_ratio": pca_ar
            })

        return per_plant_list

    if not valid_cnts:
        return []

    img_h, img_w = mask.shape[:2]

    img_h, img_w = mask.shape[:2]

    # 1. Dynamic Spatial Component Clustering (Union-Find based on max_gap_px)
    centroids = []
    for c in valid_cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
        centroids.append((cx, cy))

    n_cnts = len(valid_cnts)
    parent = list(range(n_cnts))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n_cnts):
        for j in range(i + 1, n_cnts):
            dist = np.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            if dist <= max_gap_px:
                union(i, j)

    clusters = {}
    for i in range(n_cnts):
        root = find(i)
        clusters.setdefault(root, []).append(valid_cnts[i])

    plant_groups = list(clusters.values())

    # 2. Sort plant instances deterministically by spatial grid position (top-to-bottom, left-to-right)
    sorted_plants = []
    for group in plant_groups:
        all_pts = np.vstack(group)
        x, y, w, h = cv2.boundingRect(all_pts)
        cx, cy = x + w // 2, y + h // 2
        sorted_plants.append((cy, cx, group, (x, y, w, h)))

    target_slots = [
        (0.23 * img_w, 0.30 * img_h),  # Slot 1: Top-Left
        (0.77 * img_w, 0.30 * img_h),  # Slot 2: Top-Right
        (0.50 * img_w, 0.50 * img_h),  # Slot 3: Center
        (0.23 * img_w, 0.70 * img_h),  # Slot 4: Bottom-Left
        (0.77 * img_w, 0.70 * img_h),  # Slot 5: Bottom-Right
    ]

    def _slot_index(cx, cy):
        best_i = 0
        min_d = float('inf')
        for i, (sx, sy) in enumerate(target_slots):
            d = (cx - sx)**2 + (cy - sy)**2
            if d < min_d:
                min_d = d
                best_i = i
        return best_i

    sorted_plants.sort(key=lambda item: _slot_index(item[1], item[0]))

    # 3. Dynamic Filter & Merge Stray Noise Clusters
    group_areas = []
    for cy, cx, group, bbox in sorted_plants:
        p_mask = np.zeros_like(mask)
        cv2.drawContours(p_mask, group, -1, 255, -1)
        p_mask = cv2.bitwise_and(p_mask, mask)
        group_areas.append(np.count_nonzero(p_mask))

    max_p_area = max(group_areas) if group_areas else 0

    if expected_plants is not None and expected_plants > 0 and len(sorted_plants) > expected_plants:
        # Separate major plant canopies (area >= 60px) from negligible noise specks
        major_plants = []
        minor_plants = []
        for item, area in zip(sorted_plants, group_areas):
            if area >= 60:
                major_plants.append(item)
            else:
                minor_plants.append(item)

        # Merge minor stray specks into nearest plant canopy if within 45px
        if major_plants and minor_plants:
            merged_major = []
            for m_cy, m_cx, m_group, m_bbox in major_plants:
                curr_group = list(m_group)
                for s_cy, s_cx, s_group, s_bbox in minor_plants:
                    dist = np.hypot(m_cx - s_cx, m_cy - s_cy)
                    if dist <= 45:
                        curr_group.extend(s_group)
                all_pts = np.vstack(curr_group)
                x, y, w, h = cv2.boundingRect(all_pts)
                merged_major.append((y + h // 2, x + w // 2, curr_group, (x, y, w, h)))
            valid_sorted_plants = merged_major
        else:
            valid_sorted_plants = major_plants

        # If still more than expected_plants, take the expected_plants largest plant canopies
        if len(valid_sorted_plants) > expected_plants:
            indexed_plants = sorted(zip(valid_sorted_plants, [sum(cv2.contourArea(c) for c in g) for _, _, g, _ in valid_sorted_plants]), key=lambda x: x[1], reverse=True)
            top_plants = [item for item, _ in indexed_plants[:expected_plants]]
            top_plants.sort(key=lambda item: _slot_index(item[1], item[0]))
            valid_sorted_plants = top_plants
    else:
        # Keep all valid plant canopies with area >= 60px
        valid_sorted_plants = [
            item for (item, area) in zip(sorted_plants, group_areas)
            if area >= 60
        ]

    per_plant_list = []
    for plant_idx, (_, _, group, (x, y, w, h)) in enumerate(valid_sorted_plants, 1):
        plant_mask = np.zeros_like(mask)
        cv2.drawContours(plant_mask, group, -1, 255, -1)
        plant_mask = cv2.bitwise_and(plant_mask, mask)

        leaf_px = np.count_nonzero(plant_mask)
        pla_mm2 = leaf_px * mm2_per_px

        g_px = np.count_nonzero(cv2.bitwise_and(plant_mask, green_mask))
        r_px = np.count_nonzero(cv2.bitwise_and(plant_mask, red_mask))
        pla_green_mm2 = g_px * mm2_per_px
        pla_red_mm2 = r_px * mm2_per_px

        all_pts = np.vstack(group)
        hull = cv2.convexHull(all_pts)
        hull_px = cv2.contourArea(hull)
        hull_mm2 = hull_px * mm2_per_px
        solidity = pla_mm2 / (hull_mm2 + 1e-8)

        w_mm = w * mm_per_px_linear
        h_mm = h * mm_per_px_linear

        pts = all_pts.reshape(-1, 2).astype(np.float32)
        if len(pts) >= 5:
            mean = np.mean(pts, axis=0)
            centered = pts - mean
            cov = np.cov(centered, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            sort_indices = np.argsort(evals)[::-1]
            evals = evals[sort_indices]
            evecs = evecs[:, sort_indices]
            vx, vy = evecs[0, 0], evecs[1, 0]
            pca_angle = float(np.degrees(np.arctan2(vy, vx))) % 180.0
            pca_ar = float(np.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6)))
        else:
            pca_angle = 0.0
            pca_ar = 1.0

        color_info = PLANT_COLOR_PALETTE[(plant_idx - 1) % len(PLANT_COLOR_PALETTE)]

        per_plant_list.append({
            "plant_id": f"Plant_{plant_idx:02d}",
            "contours": group,
            "bbox": (x, y, w, h),
            "centroid": (x + w // 2, y + h // 2),
            "color_bgr": color_info["bgr"],
            "color_name": color_info["name"],
            "pla_mm2": pla_mm2,
            "pla_green_mm2": pla_green_mm2,
            "pla_red_mm2": pla_red_mm2,
            "width_mm": w_mm,
            "height_mm": h_mm,
            "hull_mm2": hull_mm2,
            "solidity": solidity,
            "pca_angle": pca_angle,
            "pca_aspect_ratio": pca_ar
        })

    return per_plant_list


def compute_phenotypic_metrics(img_bgr, seg_result, mm2_per_px, calib_status="detected", expected_plants=None):
    """
    Computes summary phenotypic measurements for overall canopy and individual plant instances.
    """
    mask = seg_result["foliage_mask"]
    per_plant_list = compute_per_plant_metrics(seg_result, mm2_per_px, expected_plants=expected_plants)

    if per_plant_list:
        pla_mm2 = float(sum(p["pla_mm2"] for p in per_plant_list))
        pla_green_mm2 = float(sum(p["pla_green_mm2"] for p in per_plant_list))
        pla_red_mm2 = float(sum(p["pla_red_mm2"] for p in per_plant_list))
        valid_contours = [c for p in per_plant_list for c in p["contours"]]
        leaf_px_area = pla_mm2 / (mm2_per_px + 1e-8)
    else:
        valid_contours = seg_result.get("valid_contours", [])
        leaf_px_area = np.count_nonzero(mask)
        pla_mm2 = leaf_px_area * mm2_per_px
        green_mask = seg_result.get("green_mask", np.zeros_like(mask))
        red_mask = seg_result.get("red_mask", np.zeros_like(mask))
        pla_green_mm2 = np.count_nonzero(green_mask) * mm2_per_px
        pla_red_mm2 = np.count_nonzero(red_mask) * mm2_per_px
    
    mm_per_px_linear = np.sqrt(mm2_per_px)

    if len(valid_contours) > 0:
        all_pts = np.vstack(valid_contours)
        hull = cv2.convexHull(all_pts)
        hull_px_area = cv2.contourArea(hull)
        hull_mm2 = hull_px_area * mm2_per_px
        solidity = leaf_px_area / (hull_px_area + 1e-8)
        x, y, w, h = cv2.boundingRect(all_pts)
        canopy_width_mm = w * mm_per_px_linear
        canopy_height_mm = h * mm_per_px_linear
    else:
        hull_mm2, solidity = 0.0, 0.0
        canopy_width_mm, canopy_height_mm = 0.0, 0.0

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) >= 5:
        coords = np.column_stack((x_indices, y_indices)).astype(np.float32)
        mean = np.mean(coords, axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        vx, vy = evecs[0, 0], evecs[1, 0]
        pca_angle = float(np.degrees(np.arctan2(vy, vx))) % 180.0
        pca_aspect_ratio = float(np.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6)))
    else:
        pca_angle = 0.0
        pca_aspect_ratio = 1.0

    return {
        "pla_mm2": pla_mm2,
        "pla_green_mm2": pla_green_mm2,
        "pla_red_mm2": pla_red_mm2,
        "canopy_width_mm": canopy_width_mm,
        "canopy_height_mm": canopy_height_mm,
        "hull_mm2": hull_mm2,
        "solidity": solidity,
        "pca_orientation_angle": pca_angle,
        "pca_aspect_ratio": pca_aspect_ratio,
        "calib_status": calib_status,
        "plant_count": len(per_plant_list),
        "per_plant_metrics": per_plant_list
    }
