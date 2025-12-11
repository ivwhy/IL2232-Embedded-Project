import subprocess
import re
import csv

# =========================== Parameters 2 ==================================
ITERS = 1500000         # inner loop in CUDA-core kernel
TENSOR_ITERS = 1000   # inner loop in WMMA kernel (inside wmma_gemm_kernel)
REPEATS = 3               # outer repeat in timed region

# NVECS = [
#     #1 << 8, #256
#     #1 << 9, # 512
#     #1 << 10, #1024
#     #1 << 12, #4096
#     #1 << 13, #8192      INTRESSANT!!!!
#     1 << 14, #16384
#     1 << 15, #32768
#     40000,
#     50000,
#     1 << 16, # 65536
#     1 << 17, # 131072
#     1 << 18, #262144
#     1 << 19, # 524288
#     1 << 20 # 1048576
# ]


# M_SIZES = [
#     #16,
#     #32,
#     #64,
#     #128,
#     256,
#     512,
#     1024,
#     2048,
# ]

# OLDER, SMALLER OCCUPANCY CONFIG!!!!! =============================================================

NVECS = [
    1 << 8, #256
    1 << 9, # 512
    1 << 10, #1024
    1 << 12, #4096
    1 << 13, #8192      INTRESSANT!!!!
    1 << 14, #16384
    1 << 15, #32768
    40000,
    50000,
    1 << 16, # 65536
    1 << 17, # 131072
    1 << 18, #262144
    1 << 19, # 524288
    1 << 20 # 1048576
]


M_SIZES = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
]

# ./concurrent_cuda_tensor Nvec iters M N K tensor_iters repeats
def run_case(Nvec, iters, M, N, K, tensor_iters, repeats):
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


def main():
    # -------- Pretty Header for Terminal --------
    print("\n================ CUDA + Tensor Sweep ================\n")
    print(
        f"{'Nvec':>10} | {'iters':>9} | {'t_iters':>8} | {'M':>5} | {'N':>5} | "
        f"{'K':>5} | {'Repeats':>7} | {'CUDA (ms)':>12} | {'Tensor (ms)':>12} | "
        f"{'Serialized (ms)':>15} | {'Concurrent (ms)':>15}"
    )
    print("-" * 140)

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
                Nvec, ITERS, M, N, K, TENSOR_ITERS, REPEATS
            )
            serialized_ms = cuda_ms + tensor_ms

            # Print aligned table row
            print(
                f"{Nvec:>10} | {ITERS:>9} | {TENSOR_ITERS:>8} | {M:>5} | {N:>5} | "
                f"{K:>5} | {REPEATS:>7} | {cuda_ms:12.3f} | {tensor_ms:12.3f} | "
                f"{serialized_ms:15.3f} | {conc_ms:15.3f}"
            )

            # Write to CSV
            with open("results.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    Nvec, ITERS, TENSOR_ITERS, M, N, K, REPEATS,
                    cuda_ms, tensor_ms, serialized_ms, conc_ms,
                ])

    print("\nResults saved to: results.csv\n")

    subprocess.run(
        ["python3", "plot_runtime_speedup_heatmap_for_NvecList.py"],
        check=True
    )


if __name__ == "__main__":
    main()
