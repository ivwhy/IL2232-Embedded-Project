import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- Paths ----------------
CUDA_DIR = "../ncu_logs_cuda_only"
TENSOR_DIR = "../ncu_logs_tensor_only"
CONCURRENT_DIR = "../ncu_logs_concurrent_only"
OUT_DIR = "."

os.makedirs(OUT_DIR, exist_ok=True)



# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# SRC = os.path.join(REPO_ROOT, "src_cuda", "serial_concurrent.cu")
# BIN_DIR = os.path.join(REPO_ROOT, "src_cuda", "bin_cuda")
# BINARY = os.path.join(BIN_DIR, "serial_concurrent")



# ---------------- Regex patterns ----------------
CUDA_FILE_RE = re.compile(r"ncu_N(\d+)_iters")
TENSOR_FILE_RE = re.compile(r"tensor_M(\d+)_it")
CONCURRENT_FILE_RE = re.compile(r"ncu_N(\d+)_M(\d+)")

# COMPUTE_RE = re.compile(r"Compute \(SM\) Throughput\s+%\s+([\d,\.]+)")
# OCCUPANCY_RE = re.compile(r"Achieved Occupancy\s+%\s+([\d,\.]+)")

COMPUTE_RE = re.compile(
    r"Compute\s*\(SM\)\s*Throughput.*?([\d,\.]+)",
    re.IGNORECASE
)

OCCUPANCY_RE = re.compile(
    r"Achieved\s*Occupancy.*?([\d,\.]+)",
    re.IGNORECASE
)

# ---------------- Helpers ----------------
def parse_number(s):
    return float(s.replace(",", "."))

def extract_metrics(text):
    compute = COMPUTE_RE.search(text)
    occupancy = OCCUPANCY_RE.search(text)
    return {
        "compute": parse_number(compute.group(1)) if compute else None,
        "occupancy": parse_number(occupancy.group(1)) if occupancy else None,
    }

def extract_last_kernel_block(text, kernel_name):
    """
    Returns the text starting from the LAST occurrence of kernel_name
    """
    idx = text.rfind(kernel_name)
    if idx == -1:
        return ""
    return text[idx:]

def metrics_complete(m):
    return m["compute"] is not None and m["occupancy"] is not None


# ---------------- Load data ----------------
def load_experiments():
    data = defaultdict(dict)

    # ---- CUDA-only ----
    for fname in os.listdir(CUDA_DIR):
        m = CUDA_FILE_RE.search(fname)
        if not m:
            continue
        N = int(m.group(1))
        path = os.path.join(CUDA_DIR, fname)
        with open(path) as f:
            metrics = extract_metrics(f.read())
        data[(N, None)]["CUDA_solo"] = metrics

    # ---- Tensor-only ----
    for fname in os.listdir(TENSOR_DIR):
        m = TENSOR_FILE_RE.search(fname)
        if not m:
            continue
        M = int(m.group(1))
        path = os.path.join(TENSOR_DIR, fname)
        with open(path) as f:
            metrics = extract_metrics(f.read())
        data[(None, M)]["Tensor_solo"] = metrics

    # ---- Concurrent ----
    for fname in os.listdir(CONCURRENT_DIR):
        m = CONCURRENT_FILE_RE.search(fname)
        if not m:
            continue
        N, M = map(int, m.groups())
        path = os.path.join(CONCURRENT_DIR, fname)
        text = open(path).read()

        # # CUDA kernel metrics
        # cuda_section = text.split("wmma_gemm_kernel")[0]
        # data[(N, M)]["CUDA_concurrent"] = extract_metrics(cuda_section)

        # # Tensor kernel metrics
        # tensor_section = text.split("wmma_gemm_kernel")[-1]
        # data[(N, M)]["Tensor_concurrent"] = extract_metrics(tensor_section)

        cuda_block = extract_last_kernel_block(text, "cuda_core_fma_kernel")
        tensor_block = extract_last_kernel_block(text, "wmma_gemm_kernel")

        data[(N, M)]["CUDA_concurrent"] = extract_metrics(cuda_block)
        data[(N, M)]["Tensor_concurrent"] = extract_metrics(tensor_block)

    return data

# ---------------- Plotting ----------------
def plot_bars(plot_data, N, M):
    kernels = ["CUDA", "Tensor"]
    modes = ["solo", "concurrent"]

    compute_vals = np.array([
        [plot_data[k]["solo"]["compute"], plot_data[k]["concurrent"]["compute"]]
        for k in kernels
    ])

    occupancy_vals = np.array([
        [plot_data[k]["solo"]["occupancy"], plot_data[k]["concurrent"]["occupancy"]]
        for k in kernels
    ])

    x = np.arange(len(kernels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ---- Compute ----
    axes[0].bar(x - width/2, compute_vals[:, 0], width, label="Solo")
    axes[0].bar(x + width/2, compute_vals[:, 1], width, label="Concurrent")
    axes[0].set_title("Compute (SM) Throughput")
    axes[0].set_ylabel("Throughput (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(kernels)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # ---- Occupancy ----
    axes[1].bar(x - width/2, occupancy_vals[:, 0], width, label="Solo")
    axes[1].bar(x + width/2, occupancy_vals[:, 1], width, label="Concurrent")
    axes[1].set_title("Achieved Occupancy")
    axes[1].set_ylabel("Occupancy (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(kernels)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(f"CUDA vs Tensor Core Utilization\nN={N}, M={M}")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"N{N}_M{M}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------- Main ----------------
def main():
    raw = load_experiments()

    # merge (N,None) and (None,M) into (N,M)
    experiments = defaultdict(dict)

    for (N, M), v in raw.items():
        if N is not None and M is not None:
            experiments[(N, M)].update(v)
        elif N is not None:
            for (n2, m2), vv in raw.items():
                if n2 == N and m2 is not None:
                    experiments[(N, m2)].update(v)
        elif M is not None:
            for (n2, m2), vv in raw.items():
                if m2 == M and n2 is not None:
                    experiments[(n2, M)].update(v)

    for (N, M), e in experiments.items():
        print(f"\nN={N}, M={M}")
        for k, v in e.items():
            print(f"  {k}: {v}")

        required = {
            "CUDA_solo", "Tensor_solo",
            "CUDA_concurrent", "Tensor_concurrent"
        }
        if not required.issubset(e):
            continue

        plot_data = {
            "CUDA": {
                "solo": e["CUDA_solo"],
                "concurrent": e["CUDA_concurrent"],
            },
            "Tensor": {
                "solo": e["Tensor_solo"],
                "concurrent": e["Tensor_concurrent"],
            },
        }

        if not all(metrics_complete(plot_data[k][m]) 
                for k in ["CUDA", "Tensor"] 
                for m in ["solo", "concurrent"]):
            print(f"Skipping N={N}, M={M} due to missing metrics")
            continue

        plot_bars(plot_data, N, M)

if __name__ == "__main__":
    main()
