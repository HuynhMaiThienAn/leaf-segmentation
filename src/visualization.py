"""
Visualization & Composite Image Merging Module
===============================================
Generates clean 4-panel merged composites, per-plant numbered overlays, master grids,
and exports formatted summary metric table images (no text overlap).
"""

import cv2
import numpy as np


def _draw_subpanel_label(img, label, color=(0, 255, 255)):
    """
    Draws a concise sub-panel label enclosed in a top dark banner box to prevent text overlap.
    """
    img_h, img_w = img.shape[:2]
    font_scale = max(0.5, min(1.2, img_w / 450.0))
    thickness = max(1, int(img_w / 350.0))
    banner_h = int(font_scale * 35) + 10
    
    cv2.rectangle(img, (0, 0), (img_w, banner_h), (20, 20, 20), -1)
    cv2.putText(img, label, (10, banner_h - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, label, (10, banner_h - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def create_all_steps_breakdown_image(img_bgr, seg_result, sample_name="", pla_mm2=0.0, pca_angle=0.0, solidity=0.0, per_plant_metrics=None, alpha=0.45):
    """
    Creates a clean 2-panel result image combining:
      1. Original RGB Image
      2. Final Segmented Result Overlay (Multi-color instance masks, bounding boxes, per-plant PLA annotations)
    """
    img_h, img_w = img_bgr.shape[:2]
    contour_thick = max(2, int(img_w / 350.0))

    # 1. Original RGB Image
    p1 = img_bgr.copy()
    _draw_subpanel_label(p1, "1. Original RGB Image", color=(255, 255, 255))

    # 2. Final Segmented Result Overlay
    p2 = img_bgr.copy()
    overlay = img_bgr.copy()

    if per_plant_metrics:
        for idx, pm in enumerate(per_plant_metrics, 1):
            color = pm.get("color_bgr", (0, 255, 255))
            group = pm["contours"]
            cv2.drawContours(overlay, group, -1, color, -1)

        p2 = cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)

        for idx, pm in enumerate(per_plant_metrics, 1):
            color = pm.get("color_bgr", (0, 255, 255))
            group = pm["contours"]
            x, y, w, h = pm["bbox"]
            pla_val = pm["pla_mm2"]

            # Draw outer contour line
            cv2.drawContours(p2, group, -1, color, contour_thick)

            # Draw bounding box
            cv2.rectangle(p2, (x, y), (x + w, y + h), color, 2)

            # Draw per-plant annotation text tag with dark background box
            label = f"#{idx}: {pla_val:.1f}mm2"
            font_s = max(0.4, min(0.75, img_w / 900.0))
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, 2)

            text_y = max(text_h + 5, y - 5)
            cv2.rectangle(p2, (x, text_y - text_h - 4), (x + text_w + 6, text_y + baseline), (20, 20, 20), -1)
            cv2.putText(p2, label, (x + 3, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_s, color, 2, cv2.LINE_AA)

    _draw_subpanel_label(p2, "2. Final Segmented Result", color=(0, 255, 255))

    panels = np.hstack([p1, p2])

    banner_w = panels.shape[1]
    font_scale = max(0.6, min(1.5, banner_w / 1200.0))
    banner_h = int(font_scale * 45) + 15
    master_banner = np.zeros((banner_h, banner_w, 3), dtype=np.uint8)

    info_text = f"Sample: {sample_name}  |  Total Plants: {len(per_plant_metrics or [])}"
    cv2.putText(master_banner, info_text, (20, banner_h - 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    all_steps_breakdown = np.vstack([master_banner, panels])
    return all_steps_breakdown


def create_all_samples_master_grid(composite_images, cols=2, target_cell_w=1200):
    """
    Stitches a list of composite images into one master grid image.
    """
    if not composite_images:
        return None

    resized_cells = []
    for img in composite_images:
        h, w = img.shape[:2]
        target_h = int(h * (target_cell_w / float(w)))
        cell_resized = cv2.resize(img, (target_cell_w, target_h), interpolation=cv2.INTER_AREA)
        resized_cells.append(cell_resized)

    cell_h, cell_w = resized_cells[0].shape[:2]
    num_imgs = len(resized_cells)
    rows = (num_imgs + cols - 1) // cols

    master_grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=resized_cells[0].dtype)

    for idx, cell in enumerate(resized_cells):
        r = idx // cols
        c = idx % cols
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        ch, cw = cell.shape[:2]
        master_grid[y1:y1 + ch, x1:x1 + cw] = cell

    return master_grid


def save_summary_table_as_image(results, output_path):
    """
    Renders the phenotypic measurements as a high-resolution PNG image table.
    """
    if not results:
        return

    headers = [
        "Sample Name", "Plants", "PLA Total (mm2)", "PLA Green (mm2)", "PLA Red (mm2)", "Width (mm)", "Height (mm)", 
        "Convex Hull (mm2)", "Solidity", "PCA Angle (deg)", "PCA Aspect Ratio"
    ]
    col_widths = [200, 75, 150, 150, 150, 110, 110, 160, 95, 140, 140]

    row_height = 40
    header_height = 50
    margin = 30

    total_w = sum(col_widths) + 2 * margin
    total_h = header_height + len(results) * row_height + 2 * margin + 60

    canvas = np.full((total_h, total_w, 3), (245, 245, 245), dtype=np.uint8)

    cv2.putText(canvas, "Core Phenotypic Growth & Leaf Orientation Summary Table", (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2, cv2.LINE_AA)

    y_start = margin + 40
    cv2.rectangle(canvas, (margin, y_start), (total_w - margin, y_start + header_height), (45, 45, 45), -1)

    curr_x = margin
    for header, width in zip(headers, col_widths):
        cv2.putText(canvas, header, (curr_x + 8, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        curr_x += width

    for row_idx, res in enumerate(results):
        y_row = y_start + header_height + row_idx * row_height
        bg_color = (255, 255, 255) if row_idx % 2 == 0 else (235, 240, 245)
        cv2.rectangle(canvas, (margin, y_row), (total_w - margin, y_row + row_height), bg_color, -1)

        pla_tot = res.get("PLA_Total", res.get("PLA_mm2", 0.0))
        pla_g = res.get("PLA_Green", 0.0)
        pla_r = res.get("PLA_Red", 0.0)
        plant_cnt = res.get("Plant_Count", 1)

        row_data = [
            str(res.get("Sample", "")),
            str(plant_cnt),
            f"{pla_tot:,.1f}",
            f"{pla_g:,.1f}",
            f"{pla_r:,.1f}",
            f"{res.get('Width_mm', 0.0):.1f}",
            f"{res.get('Height_mm', 0.0):.1f}",
            f"{res.get('Hull_mm2', 0.0):,.1f}",
            f"{res.get('Solidity', 0.0):.3f}",
            f"{res.get('PCA_Angle', 0.0):.1f} deg",
            f"{res.get('PCA_Aspect', 0.0):.2f}"
        ]

        curr_x = margin
        for val, width in zip(row_data, col_widths):
            cv2.putText(canvas, val, (curr_x + 8, y_row + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
            curr_x += width

        cv2.line(canvas, (margin, y_row + row_height), (total_w - margin, y_row + row_height), (210, 210, 210), 1)

    cv2.rectangle(canvas, (margin, y_start), (total_w - margin, y_start + header_height + len(results) * row_height), (45, 45, 45), 2)
    cv2.imwrite(output_path, canvas)


def plot_per_plant_growth_curves(results, output_path):
    """
    Generates a publication-quality line plot of individual plant PLA growth over time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return

    samples = [r["Sample"] for r in results]
    sample_indices = list(range(len(samples)))

    plant_series = {}
    for r in results:
        for pm in r.get("Per_Plant", []):
            pid = pm["plant_id"]
            pla = pm["pla_mm2"]
            plant_series.setdefault(pid, []).append(pla)

    plt.figure(figsize=(12, 6.5), dpi=300)

    colors_hex = {
        "Plant_01": "#00C8C8",
        "Plant_02": "#D800D8",
        "Plant_03": "#D8D800",
        "Plant_04": "#FF8C00",
        "Plant_05": "#00C800",
    }

    for pid in sorted(plant_series.keys()):
        pla_vals = plant_series[pid]
        c_hex = colors_hex.get(pid, None)
        plt.plot(
            sample_indices[:len(pla_vals)], pla_vals, 
            marker="o", markersize=5, linewidth=2.2, label=pid, color=c_hex
        )

    plt.title("Individual Plant PLA Growth Trajectory Over Time", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sample Timepoint / Frame Index", fontsize=11, labelpad=10)
    plt.ylabel("Projected Leaf Area (PLA in mm²)", fontsize=11, labelpad=10)
    plt.xticks(sample_indices[::max(1, len(sample_indices)//10)], samples[::max(1, len(sample_indices)//10)], rotation=45, ha="right")
    plt.legend(title="Plant Instance", fontsize=10, title_fontsize=11, frameon=True, facecolor="white", edgecolor="gray")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_per_plant_summary_table_as_image(results, output_path):
    """
    Renders individual plant summary statistics across all timepoints as a clean image table.
    """
    if not results:
        return

    plant_series = {}
    for r in results:
        for pm in r.get("Per_Plant", []):
            pid = pm["plant_id"]
            pla = pm["pla_mm2"]
            plant_series.setdefault(pid, []).append(pla)

    stats = []
    for pid in sorted(plant_series.keys()):
        vals = plant_series[pid]
        if not vals:
            continue
        init_pla = vals[0]
        final_pla = vals[-1]
        max_pla = max(vals)
        mean_pla = float(np.mean(vals))
        growth_delta = final_pla - init_pla
        growth_rate_pct = (growth_delta / (init_pla + 1e-8)) * 100.0

        stats.append({
            "Plant_ID": pid,
            "Initial_PLA": init_pla,
            "Final_PLA": final_pla,
            "Max_PLA": max_pla,
            "Mean_PLA": mean_pla,
            "Growth_Delta": growth_delta,
            "Growth_Rate_Pct": growth_rate_pct,
        })

    headers = ["Plant ID", "Initial PLA (mm2)", "Final PLA (mm2)", "Max PLA (mm2)", "Mean PLA (mm2)", "Growth Δ (mm2)", "Growth Rate (%)"]
    col_widths = [130, 160, 160, 150, 150, 160, 160]

    row_height = 42
    header_height = 50
    margin = 30

    total_w = sum(col_widths) + 2 * margin
    total_h = header_height + len(stats) * row_height + 2 * margin + 60

    canvas = np.full((total_h, total_w, 3), (245, 245, 245), dtype=np.uint8)

    cv2.putText(canvas, "Individual Plant Phenotypic Growth Statistics Summary Table", (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (40, 40, 40), 2, cv2.LINE_AA)

    y_start = margin + 40
    cv2.rectangle(canvas, (margin, y_start), (total_w - margin, y_start + header_height), (45, 45, 45), -1)

    curr_x = margin
    for header, width in zip(headers, col_widths):
        cv2.putText(canvas, header, (curr_x + 8, y_start + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        curr_x += width

    for row_idx, st in enumerate(stats):
        y_row = y_start + header_height + row_idx * row_height
        bg_color = (255, 255, 255) if row_idx % 2 == 0 else (235, 240, 245)
        cv2.rectangle(canvas, (margin, y_row), (total_w - margin, y_row + row_height), bg_color, -1)

        row_data = [
            st["Plant_ID"],
            f"{st['Initial_PLA']:,.1f}",
            f"{st['Final_PLA']:,.1f}",
            f"{st['Max_PLA']:,.1f}",
            f"{st['Mean_PLA']:,.1f}",
            f"+{st['Growth_Delta']:,.1f}" if st['Growth_Delta'] >= 0 else f"{st['Growth_Delta']:,.1f}",
            f"+{st['Growth_Rate_Pct']:.1f}%" if st['Growth_Rate_Pct'] >= 0 else f"{st['Growth_Rate_Pct']:.1f}%",
        ]

        curr_x = margin
        for val, width in zip(row_data, col_widths):
            cv2.putText(canvas, val, (curr_x + 8, y_row + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
            curr_x += width

        cv2.line(canvas, (margin, y_row + row_height), (total_w - margin, y_row + row_height), (210, 210, 210), 1)

    cv2.rectangle(canvas, (margin, y_start), (total_w - margin, y_start + header_height + len(stats) * row_height), (45, 45, 45), 2)
    cv2.imwrite(output_path, canvas)


def create_growth_timelapse_gif(image_paths, output_gif_path, fps=3, max_width=1200):
    """
    Stitches a sequence of segmentation breakdown frame PNGs into an animated growth time-lapse GIF.
    """
    from PIL import Image
    import os

    if not image_paths:
        return

    frames = []
    duration_ms = int(1000.0 / max(1, fps))

    for path in image_paths:
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w > max_width:
                    new_h = int(h * (max_width / float(w)))
                    img_resized = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
                else:
                    img_resized = img.copy()
                frames.append(img_resized.convert("P", palette=Image.Palette.ADAPTIVE))
        except Exception as e:
            print(f"Warning: Failed to load frame {path} for GIF: {e}")

    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0
        )
