#!/usr/bin/env python3
import subprocess
import re
import csv
import os

BIN_DIR = "../src_cuda/bin_cuda"
SRC = "../src_cuda/cuda_core_only.cu"
BINARY = os.path.join(BIN_DIR, "cuda_core_only.cu")

# =========================== Parameters ==================================
ITERS = 150000
REPEATS = 1

NVECS = [
    # 1 << 8, #256
    # 1 << 9, # 512
    # 1 << 10, #1024
    # 1 << 12, #4096
    # 1 << 13, #8192      INTRESSANT!!!!
    # 1 << 14, #16384
    1 << 15, #32768
    40000,
    50000,
    1 << 16, # 65536
    1 << 17, # 131072
    1 << 18, #262144
    1 << 19, # 524288
    1 << 20 # 1048576
]

USE_NCU_DEFAULT = True

# NEW: Folder to store Nsight Compute logs
NCU_LOG_DIR = "ncu_logs_cuda_only"


# =================== Helper: trim ncu logs per kernel ======================

def trim_ncu_log(log_path):
    """Keep only the first report per kernel in Nsight Compute text output."""
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


# =================== One run (optionally under ncu) ========================

def run_single(N, iters, use_ncu=True):
    """Run cuda_core_only once (optionally under ncu)."""

    # Ensure log folder exists
    if use_ncu and not os.path.exists(NCU_LOG_DIR):
        os.makedirs(NCU_LOG_DIR)

    app_cmd = [BINARY, str(N), str(iters)]
    log_file = None

    if use_ncu:
        cmd = ["ncu", *app_cmd]
        log_file = os.path.join(NCU_LOG_DIR, f"ncu_N{N}_iters{iters}.txt")
    else:
        cmd = app_cmd

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        if use_ncu:
            print(f"[WARN] ncu failed for N={N}, iters={iters}, retrying without ncu...")
            result = subprocess.run(
                app_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

    combined_out = (result.stdout or "") + (result.stderr or "")

    # Save and trim logs
    if use_ncu and log_file:
        with open(log_file, "w") as f:
            f.write(combined_out)
        trim_ncu_log(log_file)

    # Parse the CUDA timing line
    m = re.search(
        r"CUDA-core kernel:\s*N=\d+\s+iters=\d+\s+time=([0-9.]+)\s*ms", combined_out
    )
    if not m:
        raise RuntimeError(f"Could not parse timing from output:\n{combined_out}")

    return float(m.group(1)), combined_out


# ============================= Run case ====================================

def run_case(N, iters, repeats, use_ncu=True):
    times = []
    last_raw = ""

    for r in range(repeats):
        time_ms, raw = run_single(N, iters, use_ncu)
        times.append(time_ms)
        last_raw = raw

    avg_time = sum(times) / len(times)
    return times, avg_time, last_raw


# ================================ Main =====================================

def main():
    ensure_binary()
    print("\n================ CUDA-Core Sweep ================\n")
    print(
        f"{'N':>10} | {'iters':>12} | {'rep':>4} | {'time (ms)':>12} | {'avg time (ms)':>15}"
    )
    print("-" * 80)

    #out_csv = "../results_csv/results_cuda_core.csv"

    out_csv = "results_cuda_core.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "iters", "repeat_index", "time_ms", "avg_over_repeats"])

    for N in NVECS:
        times, avg, raw = run_case(N, ITERS, REPEATS, use_ncu=USE_NCU_DEFAULT)

        for r_i, t in enumerate(times):
            print(f"{N:>10} | {ITERS:>12} | {r_i:>4} | {t:12.3f} | {avg:15.3f}")

            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([N, ITERS, r_i, t, avg])

    print(f"\nResults saved to {out_csv}")
    print(f"NcU logs saved to: {NCU_LOG_DIR}/")


if __name__ == "__main__":
    main()
