import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------- Paths ----------------
CUDA_DIR = "../ncu_logs_cuda_only"
TENSOR_DIR = "../ncu_logs_tensor_only"
CONCURRENT_DIR = "../ncu_logs_concurrent_only"
OUT_DIR = "heatmaps"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Regex patterns ----------------
CUDA_FILE_RE = re.compile(r"ncu_N(\d+)_iters")
TENSOR_FILE_RE = re.compile(r"tensor_M(\d+)_it")
CONCURRENT_FILE_RE = re.compile(r"ncu_N(\d+)_M(\d+)")

COMPUTE_RE = re.compile(
    r"Compute\s*\(SM\)\s*Throughput.*?([\d,\.]+)",
    re.IGNORECASE
)

# ---------------- Helpers ----------------
def parse_number(s):
    return float(s.replace(",", "."))

def extract_compute(text):
    m = COMPUTE_RE.search(text)
    return parse_number(m.group(1)) if m else None

def extract_last_kernel_block(text, kernel_name):
    idx = text.rfind(kernel_name)
    return text[idx:] if idx != -1 else ""

# ---------------- Load experiments ----------------
def load_data():
    raw = defaultdict(dict)

    # ---- CUDA solo ----
    for fname in os.listdir(CUDA_DIR):
        m = CUDA_FILE_RE.search(fname)
        if not m:
            continue
        N = int(m.group(1))
        with open(os.path.join(CUDA_DIR, fname)) as f:
            raw[(N, None)]["CUDA_solo"] = extract_compute(f.read())

    # ---- Tensor solo ----
    for fname in os.listdir(TENSOR_DIR):
        m = TENSOR_FILE_RE.search(fname)
        if not m:
            continue
        M = int(m.group(1))
        with open(os.path.join(TENSOR_DIR, fname)) as f:
            raw[(None, M)]["Tensor_solo"] = extract_compute(f.read())

    # ---- Concurrent ----
    for fname in os.listdir(CONCURRENT_DIR):
        m = CONCURRENT_FILE_RE.search(fname)
        if not m:
            continue
        N, M = map(int, m.groups())
        with open(os.path.join(CONCURRENT_DIR, fname)) as f:
            text = f.read()

        cuda_block = extract_last_kernel_block(text, "cuda_core_fma_kernel")
        tensor_block = extract_last_kernel_block(text, "wmma_gemm_kernel")

        raw[(N, M)]["CUDA_concurrent"] = extract_compute(cuda_block)
        raw[(N, M)]["Tensor_concurrent"] = extract_compute(tensor_block)

    return raw

# ---------------- Merge solo + concurrent ----------------
def merge_experiments(raw):
    merged = defaultdict(dict)

    for (N, M), v in raw.items():
        if N is not None and M is not None:
            merged[(N, M)].update(v)
        elif N is not None:
            for (_, m2), vv in raw.items():
                if m2 is not None:
                    merged[(N, m2)].update(v)
        elif M is not None:
            for (n2, _), vv in raw.items():
                if n2 is not None:
                    merged[(n2, M)].update(v)

    return merged

# ---------------- Build heatmap arrays ----------------
def build_heatmap(merged, kernel):
    Ns = sorted({N for (N, _) in merged})
    Ms = sorted({M for (_, M) in merged})

    heatmap = np.full((len(Ms), len(Ns)), np.nan)

    for i, M in enumerate(Ms):
        for j, N in enumerate(Ns):
            e = merged.get((N, M), {})
            solo = e.get(f"{kernel}_solo")
            conc = e.get(f"{kernel}_concurrent")
            if solo is not None and conc is not None:
                heatmap[i, j] = conc - solo

    return heatmap, Ns, Ms

# ---------------- Plot heatmap ----------------
def plot_heatmap(data, Ns, Ms, kernel):
    vmax = np.nanmax(np.abs(data))
    vmin = -vmax

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        data,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(im, label="Δ Compute (SM) Throughput (%)")
    plt.xticks(range(len(Ns)), Ns, rotation=45)
    plt.yticks(range(len(Ms)), Ms)

    plt.xlabel("CUDA problem size (N)")
    plt.ylabel("Tensor problem size (M)")
    plt.title(f"{kernel} SM Throughput Change\nConcurrent − Solo")

    out = os.path.join(OUT_DIR, f"heatmap_{kernel}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

# ---------------- Main ----------------
def main():
    raw = load_data()
    merged = merge_experiments(raw)

    for kernel in ["CUDA", "Tensor"]:
        heatmap, Ns, Ms = build_heatmap(merged, kernel)
        plot_heatmap(heatmap, Ns, Ms, kernel)

if __name__ == "__main__":
    main()
