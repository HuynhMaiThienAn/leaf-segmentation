"""
Flexible & Industry-Standard Plant Phenotyping CLI
==================================================
Usage examples:
  python run_segmentation.py --input leaf-image/
  python run_segmentation.py --input leaf-image/RedA/IMG_8004.JPG
  python run_segmentation.py --input leaf-image/ --cohort RedA,RedC
"""

import os
import re
import glob
import sys
import argparse
from datetime import datetime
import cv2
import numpy as np

from src.calibration import calibrate_marker
from src.metrics import compute_phenotypic_metrics
from src.visualization import (
    create_all_steps_breakdown_image,
    create_all_samples_master_grid,
    save_summary_table_as_image,
    save_per_plant_summary_table_as_image,
    plot_per_plant_growth_curves,
    create_growth_timelapse_gif,
)
from pipelines.adaptive_b0 import run_adaptive_b0_pipeline


def get_next_run_dir(base_output_dir="result", pipeline_name="adaptive_b0"):
    """
    Determines next numeric run ID and creates output directory: {run_id:03d}_{pipeline_name}_{date}
    Example: result/051_adaptive_b0_13_08_2026
    """
    os.makedirs(base_output_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, d))]

    run_numbers = []
    for d in existing_dirs:
        match_lead = re.match(r'^(\d+)_', d)
        match_run = re.search(r'run_?(\d+)', d, re.IGNORECASE)
        if match_lead:
            run_numbers.append(int(match_lead.group(1)))
        elif match_run:
            run_numbers.append(int(match_run.group(1)))
        else:
            for part in d.split("_"):
                if part.isdigit() and len(part) <= 4:
                    run_numbers.append(int(part))

    next_run_num = max(run_numbers, default=0) + 1
    now = datetime.now()
    folder_name = f"{next_run_num:03d}_{pipeline_name}_{now.strftime('%d_%m_%Y')}"
    run_output_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(run_output_dir, exist_ok=True)
    return run_output_dir, next_run_num


def calibrate_temporal_pot_anchors(image_paths, expected_plants=5):
    """
    Pass 1: Scans time-lapse dataset to dynamically discover ground-truth plant pot centers.
    Returns list of (cx, cy) anchor coordinates sorted in canonical 3-row tray order.
    """
    from src.features import extract_cielab_features
    from src.gating import create_lightness_and_border_gates, estimate_dynamic_parameters
    from src.thresholding import compute_chroma_otsu_threshold
    from src.postprocessing import apply_morphological_cleanup

    if not image_paths or expected_plants is None or expected_plants <= 0:
        return None

    n_samples = len(image_paths)
    if n_samples >= 5:
        sample_indices = np.linspace(int(n_samples * 0.25), n_samples - 1, num=min(12, n_samples), dtype=int)
    else:
        sample_indices = range(n_samples)

    all_detected_centers = []

    for idx in sample_indices:
        img_path = image_paths[idx]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_h, img_w = img_bgr.shape[:2]
        l_ch, a_star, b_star, chroma, hue_deg = extract_cielab_features(img_bgr)
        chroma_min, min_area, kernel_size = estimate_dynamic_parameters(img_bgr.shape, chroma, l_ch)
        l_gate, border_gate = create_lightness_and_border_gates(l_ch, img_bgr.shape, border_margin=0.08)
        combined_gate = l_gate & (chroma > chroma_min) & border_gate
        _, thresh, _ = compute_chroma_otsu_threshold(chroma, combined_gate)
        cleaned = apply_morphological_cleanup(thresh, kernel_size=kernel_size)

        dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        max_dist_val = np.max(dist) if np.any(cleaned) else 0
        if max_dist_val < 3.0:
            continue

        cnts, _ = cv2.findContours((dist > max(3.5, max_dist_val * 0.12)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 10:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                # Exclude side PVC tube artifacts (x < 0.18 * img_w) and border edges
                if 0.18 * img_w <= cx <= 0.86 * img_w and 0.08 * img_h <= cy <= 0.92 * img_h:
                    all_detected_centers.append((cx, cy))

    if len(all_detected_centers) < expected_plants:
        return None

    centers_arr = np.array(all_detected_centers, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(centers_arr, expected_plants, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    anchors = [(int(c[0]), int(c[1])) for c in centers]
    ref_img = cv2.imread(image_paths[0])
    img_h_ref, img_w_ref = ref_img.shape[:2] if ref_img is not None else (1000, 1000)

    target_slots = [
        (0.23 * img_w_ref, 0.31 * img_h_ref),  # Slot 1: Top-Left (147, 148)
        (0.78 * img_w_ref, 0.32 * img_h_ref),  # Slot 2: Top-Right (499, 153)
        (0.50 * img_w_ref, 0.51 * img_h_ref),  # Slot 3: Center (320, 245)
        (0.23 * img_w_ref, 0.70 * img_h_ref),  # Slot 4: Bottom-Left (147, 336)
        (0.80 * img_w_ref, 0.67 * img_h_ref),  # Slot 5: Bottom-Right (512, 322)
    ]

    ordered_anchors = [None] * expected_plants
    used_slots = set()
    for pt in anchors:
        best_slot_idx = None
        min_d = float('inf')
        for s_idx, slot in enumerate(target_slots):
            if s_idx in used_slots:
                continue
            d = np.hypot(pt[0] - slot[0], pt[1] - slot[1])
            if d < min_d:
                min_d = d
                best_slot_idx = s_idx
        if best_slot_idx is not None:
            ordered_anchors[best_slot_idx] = pt
            used_slots.add(best_slot_idx)

    final_anchors = [a if a is not None else (int(target_slots[i][0]), int(target_slots[i][1])) for i, a in enumerate(ordered_anchors)]
    return final_anchors


def main():
    parser = argparse.ArgumentParser(
        description="Industry-Standard Plant Foliage Phenotyping & Segmentation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", type=str, default="leaf-image", help="Path to input image file, directory, or glob pattern.")
    parser.add_argument("-o", "--output-dir", type=str, default="result", help="Base output directory for segmentation results.")
    parser.add_argument("-c", "--cohort", type=str, default=None, help="Filter specific cohort(s) (e.g. 'RedA', 'RedC' or 'RedA,RedC').")
    parser.add_argument("-p", "--pipeline", type=str, choices=["adaptive_b0"], default="adaptive_b0", help="Segmentation pipeline mode.")
    parser.add_argument("--expected-plants", type=int, default=5, help="Enforce fixed tray-grid spatial clustering for expected plant count (e.g. 5). Set to 0 or negative to disable.")
    parser.add_argument("--target-color", type=str, choices=["auto", "green", "red"], default="auto", help="Foliage color target mode: 'auto' (dual-color), 'green', or 'red'.")
    parser.add_argument("--min-area", type=int, default=None, help="Override minimum contour area threshold in pixels.")
    parser.add_argument("--l-min", type=int, default=15, help="Minimum lightness gate threshold (L*).")
    parser.add_argument("--l-max", type=int, default=235, help="Maximum lightness gate threshold (L*).")
    parser.add_argument("--border-margin", type=float, default=0.06, help="ROI border margin crop fraction (e.g. 0.06 = 6% edge margin).")
    parser.add_argument("--self-heal", action="store_true", default=True, help="Enable Pass 2 Temporal Anomaly Self-Healing Engine.")
    parser.add_argument("--gif", action="store_true", default=True, help="Automatically render animated growth timelapse GIF.")
    parser.add_argument("--gif-fps", type=int, default=3, help="Frames per second for the output growth timelapse GIF.")
    parser.add_argument("--no-grid", action="store_true", help="Skip rendering master grid image.")

    args = parser.parse_args()

    image_paths = []
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "**", "*.[pP][nN][gG]"), recursive=True) +
            glob.glob(os.path.join(args.input, "**", "*.[jJ][pP][gG]"), recursive=True)
        )
    else:
        image_paths = sorted(glob.glob(args.input, recursive=True))

    if args.cohort:
        cohort_list = [c.strip().lower() for c in args.cohort.split(",")]
        image_paths = [p for p in image_paths if any(f"/{c}/" in p.replace("\\", "/").lower() or f"/{c}" in p.replace("\\", "/").lower() for c in cohort_list)]

    if not image_paths:
        print(f"Error: No image samples found matching input '{args.input}' with cohort filter '{args.cohort}'.")
        sys.exit(1)

    run_output_dir, run_num = get_next_run_dir(args.output_dir, pipeline_name=args.pipeline)

    cohort_str = f" [Cohort: {args.cohort}]" if args.cohort else ""
    exp_plants_val = args.expected_plants if (args.expected_plants and args.expected_plants > 0) else None
    
    temporal_anchors = None
    if len(image_paths) >= 3 and exp_plants_val and exp_plants_val > 0:
        print("Running Pass 1: Dynamic Temporal Pot Anchor Calibration...")
        temporal_anchors = calibrate_temporal_pot_anchors(image_paths, expected_plants=exp_plants_val)
        if temporal_anchors:
            print(f"-> Discovered {len(temporal_anchors)} Dynamic Temporal Pot Anchors: {temporal_anchors}")
            ref_img = cv2.imread(image_paths[0])
            if ref_img is not None:
                pot_diag = ref_img.copy()
                for p_i, (ax, ay) in enumerate(temporal_anchors, start=1):
                    cv2.circle(pot_diag, (ax, ay), 18, (0, 255, 255), 2)
                    cv2.circle(pot_diag, (ax, ay), 4, (0, 0, 255), -1)
                    cv2.putText(pot_diag, f"Pot #{p_i} ({ax},{ay})", (ax - 45, ay - 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                pot_diag_path = os.path.join(run_output_dir, "POT_LOCATIONS_CALIBRATION.png")
                cv2.imwrite(pot_diag_path, pot_diag)
                print(f"-> Saved Pot Location Calibration Diagnostic Image: '{pot_diag_path}'")

    print("=" * 135)
    print(f"PLANT PHENOTYPING PIPELINE: {args.pipeline.upper()} (Pass 1: Calibration & Profiling | Run #{run_num:03d}) | {len(image_paths)} SAMPLES{cohort_str}")
    print(f"Output Directory: '{run_output_dir}/'")
    print("=" * 135)
    print(f"{'Index':<5} | {'Sample Name':<22} | {'PLA (mm²)':<12} | {'Width (mm)':<11} | {'Height (mm)':<11} | {'Hull (mm²)':<12} | {'Solidity':<9} | {'PCA Angle°':<11} | {'PCA AR':<8}")
    print("-" * 135)

    results, breakdown_list = [], []

    for idx, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        sample_name = os.path.basename(img_path)
        scale, _, calib_status = calibrate_marker(img_bgr)

        seg_res = run_adaptive_b0_pipeline(
            img_bgr, l_range=(args.l_min, args.l_max), min_area=args.min_area, border_margin=args.border_margin, target_color=args.target_color, expected_plants=exp_plants_val, temporal_anchors=temporal_anchors
        )
        metrics = compute_phenotypic_metrics(img_bgr, seg_res, scale, calib_status, expected_plants=exp_plants_val)

        pla_mm2 = metrics["pla_mm2"]
        w_mm = metrics["canopy_width_mm"]
        h_mm = metrics["canopy_height_mm"]
        hull_mm2 = metrics["hull_mm2"]
        solidity = metrics["solidity"]
        pca_angle = metrics["pca_orientation_angle"]
        pca_ar = metrics["pca_aspect_ratio"]
        per_plant_metrics = metrics.get("per_plant_metrics", [])

        print(f"{idx:<5} | {sample_name:<22} | {pla_mm2:<12.1f} | {w_mm:<11.1f} | {h_mm:<11.1f} | {hull_mm2:<12.1f} | {solidity:<9.3f} | {pca_angle:<11.1f} | {pca_ar:<8.2f}")

        base_filename = os.path.splitext(sample_name)[0]
        out_breakdown_path = os.path.join(run_output_dir, f"{base_filename}_result.png")

        results.append({
            "Sample": sample_name,
            "Plant_Count": len(per_plant_metrics),
            "PLA_Total": round(pla_mm2, 2),
            "PLA_Green": round(metrics["pla_green_mm2"], 2),
            "PLA_Red": round(metrics["pla_red_mm2"], 2),
            "Width_mm": round(w_mm, 2),
            "Height_mm": round(h_mm, 2),
            "Hull_mm2": round(hull_mm2, 2),
            "Solidity": round(solidity, 4),
            "PCA_Angle": round(pca_angle, 2),
            "PCA_Aspect": round(pca_ar, 3),
            "Breakdown_Path": out_breakdown_path,
            "Per_Plant": per_plant_metrics
        })

    # Pass 2: Per-Zone Color Profile Learning & Adaptive Re-Segmentation Engine
    if args.self_heal and len(results) > 2 and temporal_anchors and exp_plants_val:
        print("\n" + "=" * 135)
        print("PASS 2: PER-ZONE COLOR PROFILE LEARNING & ADAPTIVE RE-SEGMENTATION ENGINE")
        print("=" * 135)

        # Step 2a: Collect LAB pixel statistics from each zone's foliage across all Pass 1 frames
        print("-> Learning per-zone foliage LAB color profiles from Pass 1 results...")
        zone_lab_pixels = {p_i: [] for p_i in range(1, exp_plants_val + 1)}

        for f_idx, img_path in enumerate(image_paths):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            chroma_map = np.hypot(lab_img[:, :, 1] - 128.0, lab_img[:, :, 2] - 128.0)

            if f_idx < len(results):
                for p_info in results[f_idx].get("Per_Plant", []):
                    try:
                        p_num = int(p_info["plant_id"].split("_")[1])
                        if "mask" in p_info and p_info["mask"] is not None:
                            mask = p_info["mask"]
                        else:
                            # Reconstruct mask from contours if not stored
                            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                            if "contours" in p_info and p_info["contours"]:
                                cv2.drawContours(mask, p_info["contours"], -1, 255, -1)
                        if np.any(mask > 0):
                            valid_foliage = (mask > 0) & (chroma_map > 4.0)
                            if np.any(valid_foliage):
                                pixels = lab_img[valid_foliage]
                                zone_lab_pixels[p_num].append(pixels)
                    except (ValueError, KeyError, IndexError):
                        pass

        # Step 2b: Compute per-zone color fingerprints (mean ± std for L, a*, b*)
        zone_color_profiles = {}
        for p_num, pixel_batches in zone_lab_pixels.items():
            if pixel_batches:
                all_pixels = np.concatenate(pixel_batches, axis=0)
                if len(all_pixels) > 50:
                    zone_color_profiles[p_num] = {
                        "mean_L": float(np.mean(all_pixels[:, 0])),
                        "std_L": float(np.std(all_pixels[:, 0])),
                        "mean_a": float(np.mean(all_pixels[:, 1])),
                        "std_a": float(np.std(all_pixels[:, 1])),
                        "mean_b": float(np.mean(all_pixels[:, 2])),
                        "std_b": float(np.std(all_pixels[:, 2])),
                    }
                    print(f"   Zone {p_num} (Plant_{p_num:02d}): L={zone_color_profiles[p_num]['mean_L']:.1f}±{zone_color_profiles[p_num]['std_L']:.1f}, "
                          f"a*={zone_color_profiles[p_num]['mean_a']:.1f}±{zone_color_profiles[p_num]['std_a']:.1f}, "
                          f"b*={zone_color_profiles[p_num]['mean_b']:.1f}±{zone_color_profiles[p_num]['std_b']:.1f} "
                          f"({len(all_pixels)} pixels)")

        if zone_color_profiles:
            # Step 2c: Re-process ALL frames with learned color profiles
            print(f"\n-> Re-segmenting all {len(image_paths)} frames with learned color profiles...")
            print(f"{'Index':<5} | {'Sample Name':<22} | {'PLA (mm²)':<12} | {'Width (mm)':<11} | {'Height (mm)':<11} | {'Hull (mm²)':<12} | {'Solidity':<9} | {'PCA Angle°':<11} | {'PCA AR':<8}")
            print("-" * 135)

            results.clear()
            breakdown_list.clear()

            for idx, img_path in enumerate(image_paths, 1):
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                sample_name = os.path.basename(img_path)
                scale, _, calib_status = calibrate_marker(img_bgr)

                seg_res = run_adaptive_b0_pipeline(
                    img_bgr, l_range=(args.l_min, args.l_max), min_area=args.min_area, border_margin=args.border_margin, target_color=args.target_color, expected_plants=exp_plants_val, temporal_anchors=temporal_anchors, zone_color_profiles=zone_color_profiles
                )
                metrics = compute_phenotypic_metrics(img_bgr, seg_res, scale, calib_status, expected_plants=exp_plants_val)

                pla_mm2 = metrics["pla_mm2"]
                w_mm = metrics["canopy_width_mm"]
                h_mm = metrics["canopy_height_mm"]
                hull_mm2 = metrics["hull_mm2"]
                solidity = metrics["solidity"]
                pca_angle = metrics["pca_orientation_angle"]
                pca_ar = metrics["pca_aspect_ratio"]
                per_plant_metrics = metrics.get("per_plant_metrics", [])

                print(f"{idx:<5} | {sample_name:<22} | {pla_mm2:<12.1f} | {w_mm:<11.1f} | {h_mm:<11.1f} | {hull_mm2:<12.1f} | {solidity:<9.3f} | {pca_angle:<11.1f} | {pca_ar:<8.2f}")

                base_filename = os.path.splitext(sample_name)[0]
                step_breakdown = create_all_steps_breakdown_image(
                    img_bgr, seg_res, sample_name=sample_name, pla_mm2=pla_mm2, pca_angle=pca_angle, solidity=solidity, per_plant_metrics=per_plant_metrics
                )
                out_breakdown_path = os.path.join(run_output_dir, f"{base_filename}_result.png")
                cv2.imwrite(out_breakdown_path, step_breakdown)
                breakdown_list.append(step_breakdown)

                results.append({
                    "Sample": sample_name,
                    "Plant_Count": len(per_plant_metrics),
                    "PLA_Total": round(pla_mm2, 2),
                    "PLA_Green": round(metrics["pla_green_mm2"], 2),
                    "PLA_Red": round(metrics["pla_red_mm2"], 2),
                    "Width_mm": round(w_mm, 2),
                    "Height_mm": round(h_mm, 2),
                    "Hull_mm2": round(hull_mm2, 2),
                    "Solidity": round(solidity, 4),
                    "PCA_Angle": round(pca_angle, 2),
                    "PCA_Aspect": round(pca_ar, 3),
                    "Breakdown_Path": out_breakdown_path,
                    "Per_Plant": per_plant_metrics
                })
        else:
            print("-> Insufficient foliage data for color profile learning. Using Pass 1 results.")

    if results:
        table_path = os.path.join(run_output_dir, "SUMMARY_METRICS_TABLE.png")
        save_summary_table_as_image(results, table_path)
        print("=" * 135)
        print(f"SUCCESS! Rendered overall summary metrics table image: '{table_path}'")

        per_plant_table_path = os.path.join(run_output_dir, "PER_PLANT_SUMMARY_TABLE.png")
        save_per_plant_summary_table_as_image(results, per_plant_table_path)
        print(f"SUCCESS! Rendered individual plant summary statistics table image: '{per_plant_table_path}'")

        growth_curves_path = os.path.join(run_output_dir, "PER_PLANT_GROWTH_CURVES.png")
        plot_per_plant_growth_curves(results, growth_curves_path)
        print(f"SUCCESS! Rendered individual plant growth trajectory curves plot: '{growth_curves_path}'")

        # Export CSV file of per-plant metrics
        csv_path = os.path.join(run_output_dir, "PER_PLANT_PLA_METRICS.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample", "Plant_ID", "PLA_mm2", "PLA_Green_mm2", "PLA_Red_mm2", "Width_mm", "Height_mm", "Hull_mm2", "Solidity", "PCA_Angle_deg", "PCA_Aspect_Ratio"])
            for res in results:
                sample_n = res["Sample"]
                for pm in res["Per_Plant"]:
                    writer.writerow([
                        sample_n, pm["plant_id"], round(pm["pla_mm2"], 2), round(pm["pla_green_mm2"], 2),
                        round(pm["pla_red_mm2"], 2), round(pm["width_mm"], 2), round(pm["height_mm"], 2),
                        round(pm["hull_mm2"], 2), round(pm["solidity"], 4), round(pm["pca_angle"], 2), round(pm["pca_aspect_ratio"], 3)
                    ])
        print(f"SUCCESS! Exported per-plant PLA CSV metrics: '{csv_path}'")

    if results and args.gif:
        breakdown_paths = [r["Breakdown_Path"] for r in results]
        gif_path = os.path.join(run_output_dir, "GROWTH_TIMELAPSE.gif")
        create_growth_timelapse_gif(breakdown_paths, gif_path, fps=args.gif_fps)
        print(f"SUCCESS! Rendered animated growth time-lapse GIF: '{gif_path}' (FPS: {args.gif_fps})")

    if breakdown_list and not args.no_grid:
        master_grid = create_all_samples_master_grid(breakdown_list, cols=1, target_cell_w=1600)
        master_path = os.path.join(run_output_dir, "ALL_SAMPLES_MASTER_GRID.png")
        cv2.imwrite(master_path, master_grid)
        print(f"SUCCESS! Stitched all {len(breakdown_list)} sample step breakdowns into master grid: '{master_path}'")

    print("=" * 135)
    print(f"DONE! Processed {len(results)} samples. All outputs saved in '{run_output_dir}/'.")
    print("=" * 135)


if __name__ == "__main__":
    main()
