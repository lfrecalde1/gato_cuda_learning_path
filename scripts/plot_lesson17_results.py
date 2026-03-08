#!/usr/bin/env python3

import csv
import math
import os
import sys

import matplotlib.pyplot as plt


def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "lesson17_pose_comparison.csv"

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return 1

    idx = []
    ceres_t = []
    ceres_r = []
    gpu_t = []
    gpu_r = []
    gt_abs_x = []
    gt_abs_y = []
    gt_abs_z = []
    cpu_abs_x = []
    cpu_abs_y = []
    cpu_abs_z = []
    cpu_abs_t_err = []
    ceres_abs_x = []
    ceres_abs_y = []
    ceres_abs_z = []
    ceres_abs_t_err = []
    gpu_abs_x = []
    gpu_abs_y = []
    gpu_abs_z = []
    gpu_abs_t_err = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row["pair_index"])
            idx.append(i)
            ceres_t.append(to_float(row["ceres_t_err_m"]))
            ceres_r.append(to_float(row["ceres_r_err_rad"]))
            gpu_t.append(to_float(row["gpu_t_err_m"]))
            gpu_r.append(to_float(row["gpu_r_err_rad"]))
            gt_abs_x.append(to_float(row.get("gt_abs_tx", "nan")))
            gt_abs_y.append(to_float(row.get("gt_abs_ty", "nan")))
            gt_abs_z.append(to_float(row.get("gt_abs_tz", "nan")))
            cpu_abs_x.append(to_float(row.get("cpu_abs_tx", "nan")))
            cpu_abs_y.append(to_float(row.get("cpu_abs_ty", "nan")))
            cpu_abs_z.append(to_float(row.get("cpu_abs_tz", "nan")))
            cpu_abs_t_err.append(to_float(row.get("cpu_abs_t_err_m", "nan")))
            ceres_abs_x.append(to_float(row.get("ceres_abs_tx", "nan")))
            ceres_abs_y.append(to_float(row.get("ceres_abs_ty", "nan")))
            ceres_abs_z.append(to_float(row.get("ceres_abs_tz", "nan")))
            ceres_abs_t_err.append(to_float(row.get("ceres_abs_t_err_m", "nan")))
            gpu_abs_x.append(to_float(row.get("gpu_abs_tx", "nan")))
            gpu_abs_y.append(to_float(row.get("gpu_abs_ty", "nan")))
            gpu_abs_z.append(to_float(row.get("gpu_abs_tz", "nan")))
            gpu_abs_t_err.append(to_float(row.get("gpu_abs_t_err_m", "nan")))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(idx, ceres_t, label="Ceres CPU", linewidth=1.4)
    if any(math.isfinite(v) for v in gpu_t):
        axes[0].plot(idx, gpu_t, label="Custom GPU", linewidth=1.2)
    axes[0].set_ylabel("Translation Error [m]")
    axes[0].set_title("ICP Pose Error vs Ground Truth (/eagle11/odom)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(idx, ceres_r, label="Ceres CPU", linewidth=1.4)
    if any(math.isfinite(v) for v in gpu_r):
        axes[1].plot(idx, gpu_r, label="Custom GPU", linewidth=1.2)
    axes[1].set_ylabel("Rotation Error [rad]")
    axes[1].set_xlabel("Pair Index")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    out_png = "lesson17_pose_error_plot.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"Saved plot: {out_png}")

    gpu_valid = any(math.isfinite(v) for v in gpu_abs_x) and any(math.isfinite(v) for v in gpu_abs_y)
    ceres_step_err_cum = [0.0]
    cpu_step_err_cum = [0.0]
    gpu_step_err_cum = [0.0]
    for i in range(len(idx)):
        ceres_step_err_cum.append(ceres_step_err_cum[-1] + (ceres_t[i] if math.isfinite(ceres_t[i]) else 0.0))
        cpu_step_err_cum.append(cpu_step_err_cum[-1] + (cpu_abs_t_err[i] if math.isfinite(cpu_abs_t_err[i]) else 0.0))
        gpu_step_err_cum.append(gpu_step_err_cum[-1] + (gpu_t[i] if math.isfinite(gpu_t[i]) else 0.0))

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))

    axes2[0, 0].plot(gt_abs_x, gt_abs_y, label="Ground Truth", linewidth=1.8)
    axes2[0, 0].plot(cpu_abs_x, cpu_abs_y, label="Custom CPU", linewidth=1.2)
    axes2[0, 0].plot(ceres_abs_x, ceres_abs_y, label="Ceres CPU", linewidth=1.3)
    if gpu_valid:
        axes2[0, 0].plot(gpu_abs_x, gpu_abs_y, label="Custom GPU", linewidth=1.1)
    axes2[0, 0].set_title("XY Trajectory")
    axes2[0, 0].set_xlabel("X [m]")
    axes2[0, 0].set_ylabel("Y [m]")
    axes2[0, 0].axis("equal")
    axes2[0, 0].grid(True, alpha=0.25)
    axes2[0, 0].legend()

    idx_plus = [0] + idx
    axes2[0, 1].plot(idx_plus, cpu_step_err_cum, label="Custom CPU", linewidth=1.2)
    axes2[0, 1].plot(idx_plus, ceres_step_err_cum, label="Ceres CPU", linewidth=1.3)
    if gpu_valid and len(gpu_step_err_cum) == len(idx_plus):
        axes2[0, 1].plot(idx_plus, gpu_step_err_cum, label="Custom GPU", linewidth=1.1)
    axes2[0, 1].set_title("Cumulative Step Translation Error")
    axes2[0, 1].set_xlabel("Pair Index")
    axes2[0, 1].set_ylabel("Accumulated Error [m]")
    axes2[0, 1].grid(True, alpha=0.25)
    axes2[0, 1].legend()

    axes2[1, 0].plot(idx, cpu_abs_t_err, label="Custom CPU", linewidth=1.2)
    axes2[1, 0].plot(idx, ceres_abs_t_err, label="Ceres CPU", linewidth=1.3)
    if gpu_valid:
        axes2[1, 0].plot(idx, gpu_abs_t_err, label="Custom GPU", linewidth=1.1)
    axes2[1, 0].set_title("Trajectory Drift vs Ground Truth")
    axes2[1, 0].set_xlabel("Pair Index")
    axes2[1, 0].set_ylabel("Drift [m]")
    axes2[1, 0].grid(True, alpha=0.25)
    axes2[1, 0].legend()

    axes2[1, 1].plot(idx, ceres_t, label="Ceres t err [m]", linewidth=1.1)
    axes2[1, 1].plot(idx, ceres_r, label="Ceres r err [rad]", linewidth=1.1)
    if any(math.isfinite(v) for v in gpu_t):
        axes2[1, 1].plot(idx, gpu_t, label="GPU t err [m]", linewidth=1.0)
    if any(math.isfinite(v) for v in gpu_r):
        axes2[1, 1].plot(idx, gpu_r, label="GPU r err [rad]", linewidth=1.0)
    axes2[1, 1].set_title("Per-Pair Error Signals")
    axes2[1, 1].set_xlabel("Pair Index")
    axes2[1, 1].grid(True, alpha=0.25)
    axes2[1, 1].legend()

    out_png2 = "lesson17_trajectory_drift_plot.png"
    fig2.tight_layout()
    fig2.savefig(out_png2, dpi=160)
    print(f"Saved plot: {out_png2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
