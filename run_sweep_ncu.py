import subprocess
import re
import csv
import os

# =========================== Parameters 1 ==================================
# ITERS = 2000000         # inner loop in CUDA-core kernel
# TENSOR_ITERS = 100000   # inner loop in WMMA kernel (inside wmma_gemm_kernel)
# REPEATS = 3           # outer repeat in timed region

# NVECS = [
#     1 << 8,
#     1 << 10,
#     1 << 12,
#     1 << 14,
#     1 << 16,
#     #1 << 18,
#     #1 << 20,
#     #1 << 22
# ]

# M_SIZES = [
#     16,
#     32,
#     64,
#     128,
#     256,
#     #512,
#     #1024,
#     #2048,
# ]


# =========================== Parameters 2 ==================================
ITERS = 150000000         # inner loop in CUDA-core kernel
TENSOR_ITERS = 10000000   # inner loop in WMMA kernel (inside wmma_gemm_kernel)
REPEATS = 3               # outer repeat in timed region

NVECS = [
    1 << 8, #256
    1 << 9, # 512
    1 << 10, #1024
    1 << 12, #4096
    1 << 13, #8192      INTRESSANT!!!!
    1 << 14, #16384
    1 << 15, #32768
    1 << 16,
    1 << 17,
    1 << 18, #262144
]

M_SIZES = [
    16,
    32,
    64,
    128,
    #256,
    #512,
    #1024,
    #2048,
]

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

    print(f"[INFO] Trimmed Nsight log to first report per kernel: {log_path}")


# ./concurrent_cuda_tensor Nvec iters M N K tensor_iters repeats
# def run_case(Nvec, iters, M, N, K, tensor_iters, repeats, use_ncu=True):
    app_cmd = [
        "./concurrent_cuda_tensor",
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
        log_file = f"ncu_N{Nvec}_M{M}.txt"
        cmd = [
            "ncu",
            "--set", "full",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", log_file,
            "--",
            *app_cmd,
        ]
    else:
        cmd = app_cmd

    # Run (possibly under ncu)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # If ncu failed, warn and fall back to plain execution
    if use_ncu and result.returncode != 0:
        print(f"[WARN] ncu returned code {result.returncode} for Nvec={Nvec}, M={M}")
        print("[WARN] ncu stderr:\n", result.stderr)
        print("[WARN] ncu stdout (truncated):\n", result.stdout[:400], "...\n")
        log_file = None  # don't try to trim a broken log

        result = subprocess.run(
            app_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"concurrent_cuda_tensor failed (code {result.returncode})\n"
            f"stderr:\n{result.stderr}"
        )

    # If ncu worked and produced a log, trim repeated kernel reports
    if use_ncu and log_file is not None and os.path.exists(log_file):
        trim_ncu_log(log_file)

    out = result.stdout

    # Parse serialized CUDA & Tensor timings (averages)
    m = re.search(
        r"CUDA-only \(avg\)=([0-9.]+) ms, Tensor-only \(avg\)=([0-9.]+) ms",
        out
    )
    if not m:
        raise RuntimeError(f"Could not parse serialized timings from:\n{out}")
    cuda_ms = float(m.group(1))
    tensor_ms = float(m.group(2))

    # Parse concurrent timing (average)
    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+) ms",
        out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{out}")
    conc_ms = float(c.group(1))

    return cuda_ms, tensor_ms, conc_ms, out

import subprocess

def run_case(Nvec, iters, M, N, K, tensor_iters, repeats, use_ncu=True):
    app_cmd = [
        "./concurrent_cuda_tensor",
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
        # No extra options: use the simplest form that we know works:
        #   ncu ./concurrent_cuda_tensor ...
        cmd = ["ncu", *app_cmd]
        log_file = f"ncu_N{Nvec}_M{M}.txt"
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
                f"concurrent_cuda_tensor failed (code {result.returncode})\n"
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

    # ----------------- Parse CUDA + Tensor timings from combined_out -----------------
    m = re.search(
        r"CUDA-only \(avg\)=([0-9.]+) ms, Tensor-only \(avg\)=([0-9.]+) ms",
        combined_out
    )
    if not m:
        raise RuntimeError(f"Could not parse serialized timings from:\n{combined_out}")
    cuda_ms = float(m.group(1))
    tensor_ms = float(m.group(2))

    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+) ms",
        combined_out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{combined_out}")
    conc_ms = float(c.group(1))

    return cuda_ms, tensor_ms, conc_ms, combined_out


# iterates through NVEC list
def main():
    # -------- Pretty Header for Terminal --------
    print("\n================ CUDA + Tensor Sweep ================\n")
    print(
        f"{'Nvec':>10} | {'iters':>9} | {'t_iters':>8} | {'M':>5} | {'N':>5} | {'K':>5} | {'Repeats':>7} | "
        f"{'CUDA (ms)':>12} | {'Tensor (ms)':>12} | {'Serialized (ms)':>15} | {'Concurrent (ms)':>15}"
    )
    print("-" * 140)

    # --------------- CSV Output (Python-side) -----------------
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Nvec", "iters", "tensor_iters", "M", "N", "K", "Repeats",
            "cuda_ms", "tensor_ms", "serialized_ms", "concurrent_ms",
        ])

    for Nvec in NVECS:
        for M in M_SIZES:
            N = M
            K = M

            cuda_ms, tensor_ms, conc_ms, raw = run_case(
                Nvec, ITERS, M, N, K, TENSOR_ITERS, REPEATS, use_ncu=True
            )
            serialized_ms = cuda_ms + tensor_ms

            # Print aligned table row
            print(
                f"{Nvec:>10} | {ITERS:>9} | {TENSOR_ITERS:>8} | {M:>5} | {N:>5} | {K:>5} | {REPEATS:>7} | "
                f"{cuda_ms:12.3f} | {tensor_ms:12.3f} | {serialized_ms:15.3f} | {conc_ms:15.3f}"
            )

            # Write to Python's own CSV (optional, CUDA also logs its own CSV)
            with open("results.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    Nvec, ITERS, TENSOR_ITERS, M, N, K, REPEATS,
                    cuda_ms, tensor_ms, serialized_ms, conc_ms,
                ])

    print("\nResults saved to: results.csv\n")


if __name__ == "__main__":
    main()
