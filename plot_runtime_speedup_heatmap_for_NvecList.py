import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from datetime import datetime


# ============================================================
#                OUTPUT DIRECTORY
# ============================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_OUTDIR = "combined_plots"
OUTDIR = os.path.join(BASE_OUTDIR, timestamp)

os.makedirs(OUTDIR, exist_ok=True)

INDIVDIR = os.path.join(OUTDIR, "individual_plots")
os.makedirs(INDIVDIR, exist_ok=True)



# ============================================================
#                LOAD CSV + PREPARE FIELDS
# ============================================================
df = pd.read_csv("src_py/results.csv")

# df = pd.read_csv("src_py/cu_results_concurrent_only.csv")

if "serialized_ms" not in df.columns:
    df["serialized_ms"] = df["cuda_ms"] + df["tensor_ms"]

df["speedup"] = df["serialized_ms"] / df["concurrent_ms"]

nvecs_sorted = sorted(df["Nvec"].unique())
M_vals = sorted(df["M"].unique())


# ============================================================
#           GENERATE INDIVIDUAL RUNTIME & SPEEDUP PLOTS
# ============================================================
runtime_pngs = []
speedup_pngs = []

for Nvec in nvecs_sorted:
    sub = df[df["Nvec"] == Nvec].sort_values("M")

    iters = sub["iters"].iloc[0]
    t_iters = sub["tensor_iters"].iloc[0]
    repeats = sub["Repeats"].iloc[0]

    # ---------- RUNTIME PLOT ----------
    plt.figure()
    plt.title(f"Serialized vs Concurrent (Nvec={Nvec}, iters={iters}, t_iters={t_iters}, repeats={repeats})")
    plt.xlabel("Matrix size M (M=N=K)")
    plt.ylabel("Time [ms]")

    plt.plot(sub["M"], sub["serialized_ms"], marker="o", label="Serialized")
    plt.plot(sub["M"], sub["concurrent_ms"], marker="s", label="Concurrent")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    fname = f"runtime_Nvec_{int(Nvec)}.png"
    path_runtime = os.path.join(INDIVDIR, fname)
    runtime_pngs.append(path_runtime)

    plt.savefig(path_runtime, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path_runtime}")

    # ---------- SPEEDUP PLOT ----------
    plt.figure()
    plt.title(f"Speedup (Nvec={Nvec}, iters={iters}, t_iters={t_iters}, repeats={repeats})")
    plt.xlabel("Matrix size M (M=N=K)")
    plt.ylabel("Speedup = serialized / concurrent")

    plt.plot(sub["M"], sub["speedup"], marker="o")
    plt.axhline(1.0, linestyle="--", color="gray")

    plt.grid(True, linestyle="--", alpha=0.4)

    fname_s = f"speedup_Nvec_{int(Nvec)}.png"
    path_speedup = os.path.join(INDIVDIR, fname_s)
    speedup_pngs.append(path_speedup)

    plt.savefig(path_speedup, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path_speedup}")


# ============================================================
#           COMBINED GRID: RUNTIME (top) + SPEEDUP (bottom)
# ============================================================
cols = len(nvecs_sorted)
fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8))

for idx, (run_img, spd_img) in enumerate(zip(runtime_pngs, speedup_pngs)):
    runtime_pic = mpimg.imread(run_img)
    speedup_pic = mpimg.imread(spd_img)

    # Row 0 = runtime
    ax = axes[0, idx] if cols > 1 else axes[0]
    ax.imshow(runtime_pic)
    ax.axis("off")
    ax.set_title(os.path.basename(run_img), fontsize=9)

    # Row 1 = speedup
    ax = axes[1, idx] if cols > 1 else axes[1]
    ax.imshow(speedup_pic)
    ax.axis("off")
    ax.set_title(os.path.basename(speedup_pngs[idx]), fontsize=9)

combined_grid_path = os.path.join(OUTDIR, "runtime_speedup_grid.png")
plt.savefig(combined_grid_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved combined grid: {combined_grid_path}")


# ============================================================
#                     HEATMAP GENERATION
# ============================================================
Z = np.zeros((len(nvecs_sorted), len(M_vals)))

for i, Nvec in enumerate(nvecs_sorted):
    for j, M in enumerate(M_vals):
        row = df[(df["Nvec"] == Nvec) & (df["M"] == M)].iloc[0]
        Z[i, j] = row["speedup"]

plt.figure()
im = plt.imshow(
    Z,
    aspect="auto",
    origin="lower",
    extent=[min(M_vals), max(M_vals), min(nvecs_sorted), max(nvecs_sorted)],
)
plt.colorbar(im, label="Speedup (serialized / concurrent)")
plt.xlabel("Matrix size M (M=N=K)")
plt.ylabel("Nvec")
plt.title("Speedup Heatmap")

heatmap_path = os.path.join(OUTDIR, "heatmap_speedup.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved heatmap: {heatmap_path}")
