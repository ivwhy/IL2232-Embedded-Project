#!/usr/bin/env python3
import subprocess
import re
import csv
import os

# =========================== Parameters ==================================

# Matrix sizes to test: M = N = K
M_SIZES = [
    # 16,
    # 32,
    # 64,
    128,
    # 144,
    # 176,
    # 192,
    # 208,
    # 224,
    240,
    256,
    512,
    # 1024,
    2048,
    # 4096,
    # 8192,
    # 16384,
    # 32768,
    # 65536,
    # 131072
]

REPEATS = 1

# BIN_DIR = "../src_cuda/bin_cuda"
# SRC = "../src_cuda/tensor_core_only.cu"
# BINARY = os.path.join(BIN_DIR, "tensor_core_only")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

SRC = os.path.join(REPO_ROOT, "src_cuda", "tensor_core_only.cu")
BIN_DIR = os.path.join(REPO_ROOT, "src_cuda", "bin_cuda")
BINARY = os.path.join(BIN_DIR, "tensor_core_only.cu")


# NEW: per-kernel repeat count, matches tensor_iters in your CUDA code
TENSOR_ITERS = 1000  # <-- tweak this as you like

# Run under Nsight Compute by default
USE_NCU_DEFAULT = True

# Folder to store Nsight Compute logs
NCU_LOG_DIR = "../ncu_logs_tensor_only"


# =================== Helper: trim ncu logs per kernel ======================

def trim_ncu_log(log_path):
    """
    Keep only the first report per kernel in Nsight Compute text output.
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Nsight log not found, cannot trim: {log_path}")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()

    kernel_header_pattern = re.compile(r"^\s+([A-Za-z0-9_]+)\(.*\), Context ")
    trimmed = []
    seen_kernels = set()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = kernel_header_pattern.match(line)

        if not m:
            trimmed.append(line)
            i += 1
            continue

        kernel_name = m.group(1)

        if kernel_name not in seen_kernels:
            seen_kernels.add(kernel_name)
            trimmed.append(line)
            i += 1
            while i < n and not kernel_header_pattern.match(lines[i]):
                trimmed.append(lines[i])
                i += 1
        else:
            i += 1
            while i < n and not kernel_header_pattern.match(lines[i]):
                i += 1

    with open(log_path, "w") as f:
        f.writelines(trimmed)

    print(f"[INFO] Trimmed Nsight log: {log_path}")


# =================== One run (optionally under ncu) ========================

def run_single(M, N, K, tensor_iters, use_ncu=True):
    """
    Run tensor_core_only once (optionally under ncu).

    Expected output (new style):
      Tensor Core WMMA GEMM: 4096x4096x4096 tensor_iters=100000 time=1.234 ms

    Also supports old style without tensor_iters for robustness.
    """
    # Ensure log folder exists
    if use_ncu and not os.path.exists(NCU_LOG_DIR):
        os.makedirs(NCU_LOG_DIR)

    app_cmd = [BINARY, str(M), str(N), str(K), str(tensor_iters)]
    log_file = None

    if use_ncu:
        cmd = ["ncu", *app_cmd]
        log_file = os.path.join(
            NCU_LOG_DIR,
            f"tensor_M{M}_it{tensor_iters}.txt"
        )
    else:
        cmd = app_cmd

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # If ncu fails, fall back to a plain run
    if result.returncode != 0:
        if use_ncu:
            print(f"[WARN] ncu failed for M={M}, N={N}, K={K}, iters={tensor_iters}, retrying without ncu...")
            print("[WARN] ncu stderr:\n", result.stderr)
            result = subprocess.run(
                app_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{BINARY} failed (code {result.returncode})\n"
                f"stderr:\n{result.stderr}"
            )
        # don't try to trim a non-ncu log
        log_file = None

    combined_out = (result.stdout or "") + (result.stderr or "")

    # Save and trim Nsight log
    if use_ncu and log_file is not None:
        with open(log_file, "w") as f:
            f.write(combined_out)
        trim_ncu_log(log_file)

    # ---- Parse timing ----
    # New format with tensor_iters:
    #   Tensor Core WMMA GEMM: MxNxK tensor_iters=T time=XXX ms
    m = re.search(
        r"Tensor Core WMMA GEMM:\s*(\d+)x(\d+)x(\d+)\s+tensor_iters=(\d+)\s+time=([0-9.]+)\s*ms",
        combined_out
    )

    # Fallback: old format without tensor_iters
    if not m:
        m_old = re.search(
            r"Tensor Core WMMA GEMM:\s*(\d+)x(\d+)x(\d+)\s+time=([0-9.]+)\s*ms",
            combined_out
        )
        if not m_old:
            raise RuntimeError(
                f"Could not parse Tensor Core timing from output:\n{combined_out}"
            )
        time_ms = float(m_old.group(4))
    else:
        time_ms = float(m.group(5))

    return time_ms, combined_out

def ensure_binary():
    """
    Ensure bin_cuda/cuda_core_only exists.
    If not, build it with:
      nvcc -O3 -arch=sm_80 cuda_core_only.cu -o bin_cuda/cuda_core_only
    """
    os.makedirs(BIN_DIR, exist_ok=True)

    if os.path.exists(BINARY):
        return  # already built

    print(f"[INFO] Building {BINARY} from {SRC} ...")
    cmd = [
        "nvcc",
        "-O3",
        "-arch=sm_80",
        SRC,
        "-o",
        BINARY,
    ]
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc build failed with code {result.returncode}")
    print(f"[INFO] Build complete: {BINARY}")


# ============================= Run case ====================================

def run_case(M, N, K, tensor_iters, repeats, use_ncu=True):
    times = []
    last_raw = ""

    for r in range(repeats):
        time_ms, raw = run_single(M, N, K, tensor_iters, use_ncu)
        times.append(time_ms)
        last_raw = raw

    avg_time = sum(times) / len(times)
    return times, avg_time, last_raw


# ================================ Main =====================================

def main():
    ensure_binary()
    print("\n================ Tensor Core WMMA Sweep ================\n")
    print(
        f"{'M':>8} | {'N':>8} | {'K':>8} | {'t_it':>8} | {'rep':>4} | "
        f"{'time (ms)':>12} | {'avg time (ms)':>15}"
    )
    print("-" * 90)

    out_csv = "results_tensor_only.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "tensor_iters",
                         "repeat_index", "time_ms", "avg_over_repeats"])

    for M in M_SIZES:
        N = M
        K = M

        times, avg, raw = run_case(M, N, K, TENSOR_ITERS, REPEATS, use_ncu=USE_NCU_DEFAULT)

        for r_i, t in enumerate(times):
            print(
                f"{M:>8} | {N:>8} | {K:>8} | {TENSOR_ITERS:>8} | {r_i:>4} | "
                f"{t:12.3f} | {avg:15.3f}"
            )
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([M, N, K, TENSOR_ITERS, r_i, t, avg])

    print(f"\nResults saved to {out_csv}")
    print(f"Nsight logs saved to: {NCU_LOG_DIR}/")


if __name__ == "__main__":
    main()
