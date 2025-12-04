# plot_time_speedup_heatmap_IterList.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ Output folder ------------
OUTDIR = "iter_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ------------ Load and prepare data ------------
df = pd.read_csv("results.csv")

# Handle Nvec / NVECS naming just in case
if "Nvec" not in df.columns and "NVECS" in df.columns:
    df["Nvec"] = df["NVECS"]

# Ensure we have serialized_ms
if "serialized_ms" not in df.columns:
    df["serialized_ms"] = df["cuda_ms"] + df["tensor_ms"]

# Speedup column
df["speedup"] = df["serialized_ms"] / df["concurrent_ms"]

iters_sorted = sorted(df["iters"].unique())
M_vals = sorted(df["M"].unique())


# Helper to get mean metrics vs M for a given iters value
def mean_vs_M_for_iters(it_val):
    sub = df[df["iters"] == it_val]
    grouped = (
        sub.groupby("M", as_index=False)[["serialized_ms", "concurrent_ms", "speedup"]]
        .mean()
    )
    return grouped.sort_values("M")


# =========================================================
# 1) Individual runtime plots: one per iters
# =========================================================
for it in iters_sorted:
    g = mean_vs_M_for_iters(it)

    fig, ax = plt.subplots()
    ax.set_title(f"Serialized vs Concurrent runtime (iters={it})")
    ax.set_xlabel("Matrix size M (M=N=K)")
    ax.set_ylabel("Time [ms]")

    ax.plot(g["M"], g["serialized_ms"], marker="o", label="Serialized")
    ax.plot(g["M"], g["concurrent_ms"], marker="s", linestyle="--",
            label="Concurrent")

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fname = os.path.join(OUTDIR, f"runtime_iters_{it}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved runtime plot: {fname}")


# =========================================================
# 2) Combined horizontal runtime strip
# =========================================================
n_iters = len(iters_sorted)
fig, axes = plt.subplots(
    1, n_iters, figsize=(4 * n_iters, 4), squeeze=False, sharey=True
)

for col, it in enumerate(iters_sorted):
    ax = axes[0, col]
    g = mean_vs_M_for_iters(it)

    ax.set_title(f"iters={it}")
    ax.set_xlabel("M")
    if col == 0:
        ax.set_ylabel("Time [ms]")

    ax.plot(g["M"], g["serialized_ms"], marker="o", label="Serialized")
    ax.plot(g["M"], g["concurrent_ms"], marker="s", linestyle="--",
            label="Concurrent")

    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if col == 0:
        ax.legend(fontsize=8)

fig.tight_layout()
combined_runtime = os.path.join(OUTDIR, "combined_runtime_iters.png")
fig.savefig(combined_runtime, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved combined runtime strip: {combined_runtime}")


# =========================================================
# 3) Individual speedup plots: one per iters
# =========================================================
for it in iters_sorted:
    g = mean_vs_M_for_iters(it)

    fig, ax = plt.subplots()
    ax.set_title(f"Speedup (iters={it})")
    ax.set_xlabel("Matrix size M (M=N=K)")
    ax.set_ylabel("Speedup = serialized / concurrent")

    ax.plot(g["M"], g["speedup"], marker="o")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fname = os.path.join(OUTDIR, f"speedup_iters_{it}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved speedup plot: {fname}")


# =========================================================
# 4) Combined horizontal speedup strip
# =========================================================
fig, axes = plt.subplots(
    1, n_iters, figsize=(4 * n_iters, 4), squeeze=False, sharey=True
)

for col, it in enumerate(iters_sorted):
    ax = axes[0, col]
    g = mean_vs_M_for_iters(it)

    ax.set_title(f"iters={it}")
    ax.set_xlabel("M")
    if col == 0:
        ax.set_ylabel("Speedup")

    ax.plot(g["M"], g["speedup"], marker="o")
    ax.axhline(1.0, linestyle="--", color="black", linewidth=0.8)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()
combined_speedup = os.path.join(OUTDIR, "combined_speedup_iters.png")
fig.savefig(combined_speedup, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved combined speedup strip: {combined_speedup}")


# =========================================================
# 5) Combined runtime + speedup grid (2 rows × n_iters cols)
#    Top row: runtime, bottom row: speedup, matching iters per column
# =========================================================
fig, axes = plt.subplots(
    2, n_iters, figsize=(4 * n_iters, 6), squeeze=False, sharex="col"
)

for col, it in enumerate(iters_sorted):
    g = mean_vs_M_for_iters(it)

    # Top row: runtime
    ax_rt = axes[0, col]
    ax_rt.set_title(f"iters={it}")
    if col == 0:
        ax_rt.set_ylabel("Time [ms]")

    ax_rt.plot(g["M"], g["serialized_ms"], marker="o", label="Serialized")
    ax_rt.plot(g["M"], g["concurrent_ms"], marker="s", linestyle="--",
               label="Concurrent")
    ax_rt.grid(True, which="both", linestyle="--", alpha=0.4)
    if col == 0:
        ax_rt.legend(fontsize=8)

    # Bottom row: speedup
    ax_sp = axes[1, col]
    ax_sp.set_xlabel("M")
    if col == 0:
        ax_sp.set_ylabel("Speedup")

    ax_sp.plot(g["M"], g["speedup"], marker="o")
    ax_sp.axhline(1.0, linestyle="--", color="black", linewidth=0.8)
    ax_sp.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Serialized VS Concurrent Runtime (top row). Speedup (bottom row) vs Matrix size M for different Nvec iters",
             y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

combined_grid = os.path.join(OUTDIR, "combined_runtime_speedup_grid.png")
fig.savefig(combined_grid, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved combined runtime+speedup grid: {combined_grid}")


# =========================================================
# 6) Heatmap: y = iters, x = M, color = speedup
# =========================================================
Z = np.full((len(iters_sorted), len(M_vals)), np.nan)

for i, it in enumerate(iters_sorted):
    for j, M in enumerate(M_vals):
        rows = df[(df["iters"] == it) & (df["M"] == M)]
        if not rows.empty:
            Z[i, j] = rows["speedup"].mean()

fig, ax = plt.subplots()

im = ax.imshow(Z, aspect="auto", origin="lower")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Speedup (serialized / concurrent)")

ax.set_xticks(range(len(M_vals)))
ax.set_xticklabels(M_vals)
ax.set_yticks(range(len(iters_sorted)))
ax.set_yticklabels(iters_sorted)

ax.set_xlabel("Matrix size M (M=N=K)")
ax.set_ylabel("iters")
ax.set_title("Speedup Heatmap")

heatmap_path = os.path.join(OUTDIR, "heatmap_speedup.png")
fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved heatmap: {heatmap_path}")
