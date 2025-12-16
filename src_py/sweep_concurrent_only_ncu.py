#!/usr/bin/env python3
import subprocess
import re
import csv
import os

# =========================== Parameters ==================================
# Tune these to match your experiments

ITERS = 150000       # inner loop in CUDA-core kernel
TENSOR_ITERS = 1000 # inner loop in WMMA kernel (inside wmma_gemm_kernel)
REPEATS = 3             # outer repeat in timed region (passed to concurrent_only)

# Vector sizes (CUDA-core workload)
NVECS = [
    1 << 8,   # 256
    1 << 9,   # 512
    1 << 10,  # 1024
    1 << 12,  # 4096
    1 << 13,  # 8192
    1 << 14,  # 16384
    1 << 15,  # 32768
    1 << 16,  # 65536
    1 << 17,  # 131072
    1 << 18,  # 262144
]

# Matrix sizes (Tensor Core workload): M = N = K
M_SIZES = [
    256,
    512,
    1024,
    2048,
]

# from src_py/, go up one level into src_cuda
BIN_DIR = "../src_cuda/bin_cuda"
SRC = "../src_cuda/concurrent_only.cu"
BINARY = os.path.join(BIN_DIR, "concurrent_only")

# Run under Nsight Compute by default
USE_NCU_DEFAULT = True

# Folder to store Nsight Compute logs
NCU_LOG_DIR = "../ncu_logs_concurrent_only"


# =================== Helper: trim ncu logs per kernel ======================

def trim_ncu_log(log_path):
    """
    Keep only the first report for each kernel name in an Nsight Compute text log.

    - Preserves all non-kernel lines (==PROF==, summaries, etc.).
    - For each kernel header line like:
        "  some_kernel_name(...), Context 1, Stream 14, Device 0, CC 8.6"
      it keeps only the *first* section for that kernel name and drops later repeats.
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Nsight log file not found, cannot trim: {log_path}")
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
            # Not a kernel header, keep as-is
            trimmed.append(line)
            i += 1
            continue

        kernel_name = m.group(1)

        if kernel_name not in seen_kernels:
            # First time we see this kernel: keep this section
            seen_kernels.add(kernel_name)
            trimmed.append(line)
            i += 1
            # Copy lines until the next kernel header or EOF
            while i < n and not kernel_header_pattern.match(lines[i]):
                trimmed.append(lines[i])
                i += 1
        else:
            # Already saw this kernel: skip this repeated section
            i += 1
            while i < n and not kernel_header_pattern.match(lines[i]):
                i += 1

    with open(log_path, "w") as f:
        f.writelines(trimmed)

    print(f"[INFO] Trimmed Nsight log: {log_path}")


# =================== One run (optionally under ncu) ========================

def run_case(Nvec, iters, M, N, K, tensor_iters, repeats, use_ncu=True):
    """
    Run ./concurrent_only once (optionally under ncu) for given parameters.

    concurrent_only.cu is expected to print a line like:
      Concurrent total time (avg)=123.456 ms (repeats=3)
    """
    app_cmd = [
        BINARY,
        str(Nvec),
        str(iters),
        str(M),
        str(N),
        str(K),
        str(tensor_iters),
        str(repeats),
    ]

    log_file = None

    if use_ncu:
        # Ensure log directory exists
        os.makedirs(NCU_LOG_DIR, exist_ok=True)

        # Simple form: ncu ./concurrent_only ...
        cmd = ["ncu", *app_cmd]
        log_file = os.path.join(NCU_LOG_DIR, f"ncu_N{Nvec}_M{M}.txt")
    else:
        cmd = app_cmd

    # Run (possibly under ncu), capturing both stdout and stderr
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        # If ncu fails, fall back to plain execution
        if use_ncu:
            print(f"[WARN] ncu returned code {result.returncode} for Nvec={Nvec}, M={M}")
            print("[WARN] ncu stderr:\n", result.stderr)
            print("[WARN] Falling back to running without ncu for this case.\n")

            result = subprocess.run(
                app_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        # If the plain run also fails, bail
        if result.returncode != 0:
            raise RuntimeError(
                f"{BINARY} failed (code {result.returncode})\n"
                f"stderr:\n{result.stderr}"
            )

        # If we got here, we ran without ncu and shouldn't try to trim logs
        log_file = None

    # Combine stdout + stderr: ncu messages + program prints all together
    combined_out = (result.stdout or "") + (result.stderr or "")

    # If we were using ncu and have somewhere to log, write and trim the log
    if use_ncu and log_file is not None:
        with open(log_file, "w") as f:
            f.write(combined_out)
        trim_ncu_log(log_file)

    # ----------------- Parse concurrent timing from combined_out -----------------
    # Expected line:
    #   Concurrent total time (avg)=123.456 ms (repeats=3)
    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+)\s*ms",
        combined_out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{combined_out}")

    conc_ms = float(c.group(1))
    return conc_ms, combined_out

def ensure_binary():
    """
    Ensure bin_cuda/cuda_core_only exists.
    If not, build it with:

      nvcc -O3 -arch=sm_80 concurrent_only.cu -o bin_cuda/concurrent_only
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


# ================================ Main =====================================

def main():
    ensure_binary()
    # -------- Pretty Header for Terminal --------
    print("\n================ CUDA + Tensor Concurrent-Only Sweep (ncu) ================\n")
    print(
        f"{'Nvec':>10} | {'iters':>9} | {'t_iters':>10} | "
        f"{'M':>5} | {'N':>5} | {'K':>5} | {'Repeats':>7} | "
        f"{'Concurrent (ms)':>16}"
    )
    print("-" * 100)

    # --------------- CSV Output (Python-side) -----------------
    out_csv = "../results_csv/results_concurrent_only_ncu.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Nvec", "iters", "tensor_iters", "M", "N", "K", "Repeats",
            "concurrent_ms",
        ])

    for Nvec in NVECS:
        for M in M_SIZES:
            N = M
            K = M

            conc_ms, raw = run_case(
                Nvec, ITERS, M, N, K, TENSOR_ITERS, REPEATS, use_ncu=USE_NCU_DEFAULT
            )

            # Print aligned table row
            print(
                f"{Nvec:>10} | {ITERS:>9} | {TENSOR_ITERS:>10} | "
                f"{M:>5} | {N:>5} | {K:>5} | {REPEATS:>7} | "
                f"{conc_ms:16.3f}"
            )

            # Append to CSV
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    Nvec, ITERS, TENSOR_ITERS, M, N, K, REPEATS,
                    conc_ms,
                ])

    print(f"\nResults saved to: {out_csv}")
    print(f"Nsight logs saved to: {NCU_LOG_DIR}/")


if __name__ == "__main__":
    main()
