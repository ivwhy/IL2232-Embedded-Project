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


# # =========================== Parameters 2 ==================================
ITERS = 150000000         # inner loop in CUDA-core kernel   ex: (iters=2000000000, nvec=4) => 9878ms
TENSOR_ITERS = 10000000   # inner loop in WMMA kernel (inside wmma_gemm_kernel)
REPEATS = 3           # outer repeat in timed region

NVECS = [
    #1 << 2,
    #1 << 3,
    #1 << 4,
    1 << 8,    # 256
    #1 << 10,
    1 << 12,   # 4096
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 17,
    1 << 18,
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

# # =========================== Parameters 3 ================================== USE OTHER MAIN
# NVECS = 256         # inner loop in CUDA-core kernel   ex: (iters=2000000000, nvec=4) => 9878ms
# TENSOR_ITERS = 10000000   # inner loop in WMMA kernel (inside wmma_gemm_kernel)
# REPEATS = 3           # outer repeat in timed region

# ITERS = [
#     200000000,
#     400000000,
#     800000000,
#     1000000000,
# ]

# M_SIZES = [
#     16,
#     32,
#     64,
#     128,
#     #256,
#     #512,
#     #1024,
#     #2048,
# ]


def trim_ncu_log(log_path):
    """
    Trim an Nsight Compute text log so that it only keeps:
      - Everything up to (and including) the *first* kernel report
      - And everything inside that first kernel report
    Any subsequent repeated kernel reports are dropped.

    The original file is overwritten in-place.
    """
    if not os.path.exists(log_path):
        print(f"[WARN] Nsight log file not found, cannot trim: {log_path}")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()

    # We treat any line like:
    #   "  wmma_gemm_kernel(...), Context 1, Stream ..., Device ..., CC ..."
    # or in general:
    #   "  <kernel_name>(...), Context ..."
    # as the start of a kernel report.
    kernel_header_pattern = re.compile(r"^\s+\w+\(.*\), Context ")

    trimmed = []
    in_first_kernel = False
    seen_first_kernel = False

    for line in lines:
        if not seen_first_kernel:
            trimmed.append(line)
            # Look for the first kernel header line
            if kernel_header_pattern.match(line):
                in_first_kernel = True
                seen_first_kernel = True
        else:
            # We already started the first kernel section
            if kernel_header_pattern.match(line):
                # This is the *second* kernel header: stop before duplicating
                break
            trimmed.append(line)

    # Overwrite original log with trimmed version
    with open(log_path, "w") as f:
        f.writelines(trimmed)

    print(f"[INFO] Trimmed Nsight log to first kernel report: {log_path}")


# ./concurrent_cuda_tensor Nvec iters M N K tensor_iters repeats
#def run_case(Nvec, iters, M, N, K, tensor_iters, repeats):
    cmd = [
        "./concurrent_cuda_tensor",
        str(Nvec),
        str(iters),
        str(M),
        str(N),
        str(K),
        str(tensor_iters),
        str(repeats),
    ]

    out = subprocess.check_output(cmd, text=True)

    # Parse serialized CUDA & Tensor timings (averages)
    # Matches:
    # Serialized: CUDA-only (avg)=123.456 ms, Tensor-only (avg)=789.012 ms, ...
    m = re.search(
        r"CUDA-only \(avg\)=([0-9.]+) ms, Tensor-only \(avg\)=([0-9.]+) ms",
        out
    )
    if not m:
        raise RuntimeError(f"Could not parse serialized timings from:\n{out}")
    cuda_ms = float(m.group(1))
    tensor_ms = float(m.group(2))

    # Parse concurrent timing (average)
    # Matches:
    # Concurrent total time (avg)=345.678 ms (repeats=10)
    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+) ms",
        out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{out}")
    conc_ms = float(c.group(1))

    return cuda_ms, tensor_ms, conc_ms, out


# run case with NCU
#def run_case(Nvec, iters, M, N, K, tensor_iters, repeats, use_ncu=True):
    # Base application command
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

    if use_ncu:
        # Nsight Compute wrapper
        # --log-file keeps profiler output out of stdout so your regex still works
        cmd = [
            "ncu",
            "--set", "full",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", f"ncu_N{Nvec}_M{M}.txt",
            "--",
            *app_cmd,
        ]
    else:
        cmd = app_cmd

    out = subprocess.check_output(cmd, text=True)

    # --- keep the rest of your parsing exactly as-is ---
    m = re.search(
        r"CUDA-only \(avg\)=([0-9.]+) ms, Tensor-only \(avg\)=([0-9.]+) ms",
        out
    )
    if not m:
        raise RuntimeError(f"Could not parse serialized timings from:\n{out}")
    cuda_ms = float(m.group(1))
    tensor_ms = float(m.group(2))

    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+) ms",
        out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{out}")
    conc_ms = float(c.group(1))

    return cuda_ms, tensor_ms, conc_ms, out

# run case with NCU error handling
def run_case(Nvec, iters, M, N, K, tensor_iters, repeats, use_ncu=True):
    # Base application command
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

    if use_ncu:
        cmd = [
            "ncu",
            "--set", "full",
            "--target-processes", "all",
            "--force-overwrite",
            "--log-file", f"ncu_N{Nvec}_M{M}.txt",
            "--",
            *app_cmd,
        ]
    else:
        cmd = app_cmd

    # Run (possibly under ncu), but don't throw on non-zero exit
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # If ncu failed, fall back to plain execution so the sweep can continue
    if use_ncu and result.returncode != 0:
        print(f"[WARN] ncu failed for Nvec={Nvec}, M={M} (code {result.returncode})")
        print("[WARN] ncu stderr:\n", result.stderr)
        print("[WARN] Falling back to running without ncu for this case.\n")

        result = subprocess.run(
            app_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        # At this point it's the app itself failing – bail with a clear message
        raise RuntimeError(
            f"concurrent_cuda_tensor failed (code {result.returncode})\n"
            f"stderr:\n{result.stderr}"
        )

    # If we have a valid Nsight log, trim it down to just one kernel section !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11 ivys edit
    if use_ncu and log_file is not None:
        trim_ncu_log(log_file)

    out = result.stdout

    # ----------------- keep parsing logic as-is -----------------
    m = re.search(
        r"CUDA-only \(avg\)=([0-9.]+) ms, Tensor-only \(avg\)=([0-9.]+) ms",
        out
    )
    if not m:
        raise RuntimeError(f"Could not parse serialized timings from:\n{out}")
    cuda_ms = float(m.group(1))
    tensor_ms = float(m.group(2))

    c = re.search(
        r"Concurrent total time \(avg\)=([0-9.]+) ms",
        out
    )
    if not c:
        raise RuntimeError(f"Could not parse concurrent timing from:\n{out}")
    conc_ms = float(c.group(1))

    return cuda_ms, tensor_ms, conc_ms, out



# iterates through NVEC list
def main():
    # -------- Pretty Header for Terminal --------
    print("\n================ CUDA + Tensor Sweep ================\n")
    print(
        f"{'Nvec':>10} | {'iters':>7} | {'t_iters':>8} | {'M':>5} | {'N':>5} | {'K':>5} | {'Repeats':>7} | "
        f"{'CUDA (ms)':>12} | {'Tensor (ms)':>12} | {'Serialized (ms)':>15} | {'Concurrent (ms)':>15}"
    )
    print("-" * 130)

    # --------------- CSV Output -----------------
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
                f"{Nvec:>10} | {ITERS:>7} | {TENSOR_ITERS:>8} | {M:>5} | {N:>5} | {K:>5} | {REPEATS:>7} | "
                f"{cuda_ms:12.3f} | {tensor_ms:12.3f} | {serialized_ms:15.3f} | {conc_ms:15.3f}"
            )

            # Write to CSV
            with open("results.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    Nvec, ITERS, TENSOR_ITERS, M, N, K, REPEATS,
                    cuda_ms, tensor_ms, serialized_ms, conc_ms,
                ])
            
            # #Write to CSV
            # writer.writerow([
            #     Nvec, ITERS, TENSOR_ITERS, M, N, K, REPEATS,
            #     cuda_ms, tensor_ms, serialized_ms, conc_ms,
            # ])

    print("\nResults saved to: results.csv\n")

# iterates through Iter list
# def main():
#     # -------- Pretty Header for Terminal --------
#     print("\n================ CUDA + Tensor Sweep ================\n")
#     print(
#         f"{'NVECS':>5} | {'iters':>8} | {'t_iters':>8} | {'M':>5} | {'N':>5} | {'K':>5} | {'Repeats':>7} | "
#         f"{'CUDA (ms)':>12} | {'Tensor (ms)':>12} | {'Serialized (ms)':>15} | {'Concurrent (ms)':>15}"
#     )
#     print("-" * 130)

#     # --------------- CSV Output -----------------
#     with open("results.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "Nvec", "iters", "tensor_iters", "M", "N", "K", "Repeats",
#             "cuda_ms", "tensor_ms", "serialized_ms", "concurrent_ms",
#         ])

#     for Iter in ITERS:
#         for M in M_SIZES:
#             N = M
#             K = M

#             cuda_ms, tensor_ms, conc_ms, raw = run_case(
#                 NVECS, Iter, M, N, K, TENSOR_ITERS, REPEATS
#             )
#             serialized_ms = cuda_ms + tensor_ms

#             # Print aligned table row
#             print(
#                 f"{NVECS:>5} | {Iter:>8} | {TENSOR_ITERS:>8} | {M:>5} | {N:>5} | {K:>5} | {REPEATS:>7} | "
#                 f"{cuda_ms:12.3f} | {tensor_ms:12.3f} | {serialized_ms:15.3f} | {conc_ms:15.3f}"
#             )

#             # # Write to CSV
#             # writer.writerow([
#             #     NVECS, Iter, TENSOR_ITERS, M, N, K, REPEATS,
#             #     cuda_ms, tensor_ms, serialized_ms, conc_ms,
#             # ])

#             # Write to CSV
#             with open("results.csv", "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                 NVECS, Iter, TENSOR_ITERS, M, N, K, REPEATS,
#                 cuda_ms, tensor_ms, serialized_ms, conc_ms,
#             ])

#     print("\nResults saved to: results.csv\n")

if __name__ == "__main__":
    main()
