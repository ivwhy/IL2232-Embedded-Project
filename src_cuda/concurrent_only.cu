// concurrent_only.cu
// Build: nvcc -O3 -arch=sm_80 concurrent_only.cu -o concurrent_only
// Run:   ./concurrent_only [Nvec=1<<24] [iters=1024] [M=1024] [N=1024] [K=1024] [tensor_iters=1024] [repeats=1]
// Example:
//   ./concurrent_only 256 100 4096 4096 4096 100 100
//   ncu ./concurrent_only 256 150000000 16 16 16 10000000 3

#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <fstream>

using namespace nvcuda;

// ---- CUDA-core kernel (same as in your concurrent code) ----
__global__ void cuda_core_fma_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     int N, int iters) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {

        float x = a[idx];
        float y = b[idx];
        float acc = 0.f;

        #pragma unroll 4
        for (int k = 0; k < iters; ++k) {
            acc = fmaf(x, y, acc);
            // decorrelate operands
            x = fmaf(x, 1.000001f, 0.000001f);
            y = fmaf(y, 0.999999f, 0.000002f);
        }
        c[idx] = acc;
    }
}

// ---- Tensor Core WMMA kernel (same as in your updated code) ----
__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 int tensor_iters) {
    int tileRow = blockIdx.y; // along M
    int tileCol = blockIdx.x; // along N

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    int row = tileRow * 16;
    int col = tileCol * 16;

    // Initialize accumulator once
    wmma::fill_fragment(cFrag, 0.0f);

    // Repeat GEMM body tensor_iters times
    for (int rep = 0; rep < tensor_iters; ++rep) {
        for (int k = 0; k < K; k += 16) {
            const half* tileA = A + row * K + k;
            const half* tileB = B + k * N + col;
            wmma::load_matrix_sync(aFrag, tileA, K);
            wmma::load_matrix_sync(bFrag, tileB, N);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    float* tileC = C + row * N + col;
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);
}

// ---- small helpers -------------------------------------------------------

static void init_half(half* h, int n, float seed) {
    for (int i = 0; i < n; ++i) {
        h[i] = __float2half(seed + (i % 13) * 0.01f);
    }
}

bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void append_csv_row(int Nvec, int iters, int tensor_iters,
                    int M, int N, int K, int repeats,
                    float conc_ms) {
    const char* filename = "cu_results_concurrent_only.csv";

    bool exists = file_exists(filename);

    std::ofstream file;
    file.open(filename, std::ios::app);

    if (!file.is_open()) {
        printf("[WARN] Could not open %s for writing.\n", filename);
        return;
    }

    if (!exists) {
        // Write header once
        file << "Nvec,iters,tensor_iters,M,N,K,repeats,concurrent_ms\n";
    }

    file << Nvec << ","
         << iters << ","
         << tensor_iters << ","
         << M << ","
         << N << ","
         << K << ","
         << repeats << ","
         << conc_ms << "\n";

    file.close();
}

// ---- main: concurrent timing only ----------------------------------------

int main(int argc, char** argv) {
    // Vector work (CUDA cores)
    int Nvec         = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    int iters        = (argc > 2) ? atoi(argv[2]) : 1024;

    // Matrix sizes (Tensor Cores)
    int M            = (argc > 3) ? atoi(argv[3]) : 1024;
    int N            = (argc > 4) ? atoi(argv[4]) : 1024;
    int K            = (argc > 5) ? atoi(argv[5]) : 1024;
    int tensor_iters = (argc > 6) ? atoi(argv[6]) : 1024;
    int repeats      = (argc > 7) ? atoi(argv[7]) : 1;

    if (M % 16 || N % 16 || K % 16) {
        printf("M,N,K must be multiples of 16 (got %d %d %d).\n", M, N, K);
        return 1;
    }

    printf("Concurrent-only run:\n");
    printf("  Nvec=%d, iters=%d\n", Nvec, iters);
    printf("  M=%d, N=%d, K=%d, tensor_iters=%d\n", M, N, K, tensor_iters);
    printf("  repeats=%d\n", repeats);

    // --- Allocate & init CUDA vector data ---
    size_t vbytes = (size_t)Nvec * sizeof(float);
    float *a, *b, *c;
    cudaMalloc(&a, vbytes);
    cudaMalloc(&b, vbytes);
    cudaMalloc(&c, vbytes);

    float *ha = (float*)malloc(vbytes);
    float *hb = (float*)malloc(vbytes);
    for (int i = 0; i < Nvec; ++i) {
        ha[i] = 1.0f + (i % 13) * 0.001f;
        hb[i] = 2.0f + (i % 7) * 0.002f;
    }
    cudaMemcpy(a, ha, vbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, vbytes, cudaMemcpyHostToDevice);

    // --- Allocate & init Tensor matrix data ---
    size_t aBytes = (size_t)M * K * sizeof(half);
    size_t bBytes = (size_t)K * N * sizeof(half);
    size_t cBytes = (size_t)M * N * sizeof(float);

    half *A, *B;
    float *C;
    cudaMalloc(&A, aBytes);
    cudaMalloc(&B, bBytes);
    cudaMalloc(&C, cBytes);

    half *hA = (half*)malloc(aBytes);
    half *hB = (half*)malloc(bBytes);
    init_half(hA, M * K, 1.0f);
    init_half(hB, K * N, 2.0f);
    cudaMemcpy(A, hA, aBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bBytes, cudaMemcpyHostToDevice);

    // --- Launch configs ---
    int block = 256;
    int grid  = (Nvec + block - 1) / block;
    grid      = (grid > 65535) ? 65535 : grid;  // cap for safety

    dim3 blockW(32, 1, 1);
    dim3 gridW(N / 16, M / 16, 1);

    // --- Streams ---
    cudaStream_t sCuda, sTensor;
    cudaStreamCreate(&sCuda);
    cudaStreamCreate(&sTensor);

    // --- Warmup: launch both concurrently once ---
    cuda_core_fma_kernel<<<grid, block, 0, sCuda>>>(a, b, c, Nvec, iters);
    wmma_gemm_kernel<<<gridW, blockW, 0, sTensor>>>(A, B, C, M, N, K, tensor_iters);
    cudaDeviceSynchronize();

    // --- Timed: concurrent only ---
    cudaEvent_t cStart, cStop;
    cudaEventCreate(&cStart);
    cudaEventCreate(&cStop);

    float ms_conc     = 0.0f;
    float ms_conc_sum = 0.0f;

    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(cStart);  // default stream

        cuda_core_fma_kernel<<<grid, block, 0, sCuda>>>(a, b, c, Nvec, iters);
        wmma_gemm_kernel<<<gridW, blockW, 0, sTensor>>>(A, B, C, M, N, K, tensor_iters);

        // wait for all streams to finish
        cudaDeviceSynchronize();

        cudaEventRecord(cStop);
        cudaEventSynchronize(cStop);

        cudaEventElapsedTime(&ms_conc, cStart, cStop);
        ms_conc_sum += ms_conc;
    }

    float ms_conc_avg = ms_conc_sum / repeats;
    printf("Concurrent total time (avg)=%.3f ms (repeats=%d)\n",
           ms_conc_avg, repeats);

    // Write CSV with concurrent-only timing
    append_csv_row(
        Nvec, iters, tensor_iters,
        M, N, K, repeats,
        ms_conc_avg
    );

    // Cleanup
    cudaEventDestroy(cStart);
    cudaEventDestroy(cStop);
    cudaStreamDestroy(sCuda);
    cudaStreamDestroy(sTensor);

    cudaFree(a); cudaFree(b); cudaFree(c);
    free(ha); free(hb);

    cudaFree(A); cudaFree(B); cudaFree(C);
    free(hA); free(hB);

    return 0;
}
