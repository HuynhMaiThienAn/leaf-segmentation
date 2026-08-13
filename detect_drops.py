"""Detect PLA drops and anomalies in per-plant metrics."""
import csv

import os, glob, csv

# Find latest result run directory
dirs = sorted(glob.glob("result/[0-9][0-9][0-9]_*"))
latest_dir = dirs[-1] if dirs else "result/039_adaptive_b0_13_08_2026"
csv_path = os.path.join(latest_dir, "PER_PLANT_PLA_METRICS.csv")
print(f"Reading PLA metrics from: {csv_path}")
rows = list(csv.DictReader(open(csv_path)))

# Group by plant
plants = {}
for r in rows:
    pid = r["Plant_ID"]
    plants.setdefault(pid, []).append((r["Sample"], float(r["PLA_mm2"])))

for pid in sorted(plants):
    print(f"\n=== {pid} ===")
    prev_pla = None
    for i, (sample, pla) in enumerate(plants[pid]):
        flags = []
        if pla < 100:
            flags.append("SMALL")
        if prev_pla is not None and pla < prev_pla * 0.75:
            flags.append(f"DROP ({pla/prev_pla*100:.0f}%)")
        if prev_pla is not None and pla < prev_pla * 0.5:
            flags.append("SEVERE_DROP")
        flag_str = "  *** " + ", ".join(flags) + " ***" if flags else ""
        print(f"  {sample:22s} PLA={pla:8.1f}{flag_str}")
        prev_pla = pla
