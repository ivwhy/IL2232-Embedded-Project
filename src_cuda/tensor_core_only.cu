// tensor_core_only.cu
// Build: nvcc -arch=sm_86 tensor_core_only.cu -o tensor_core_only
// Run:   ./tensor_core_only [M=1024] [N=1024] [K=1024] [tensor_iters=1]
// Example: ./tensor_core_only 4096 4096 4096 10000

#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// Each block = one warp computes one 16x16 C tile
__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 int tensor_iters) {
    int tileRow = blockIdx.y;     // along M
    int tileCol = blockIdx.x;     // along N

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

    int row = tileRow * 16;
    int col = tileCol * 16;

    // Initialize accumulator once
    wmma::fill_fragment(cFrag, 0.0f);

    // Repeat the GEMM body tensor_iters times
    for (int rep = 0; rep < tensor_iters; ++rep) {
        for (int k = 0; k < K; k += 16) {
            const half* tileA = A + row * K + k;
            const half* tileB = B + k * N + col;  // B is treated as col-major
            wmma::load_matrix_sync(aFrag, tileA, K);
            wmma::load_matrix_sync(bFrag, tileB, N);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }

    // Store C tile (row-major)
    float* tileC = C + row * N + col;
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);
}

// initialize an array of half with a small repeating pattern starting at seed
static void init_half(half* h, int n, float seed) {
    for (int i = 0; i < n; ++i) {
        float v = seed + (i % 13) * 0.01f;
        h[i] = __float2half(v);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);  // disable buffering

    //printf("in main.\n");

    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int tensor_iters = (argc > 4) ? atoi(argv[4]) : 1;  // NEW: per-kernel repeat count

    if (M % 16 || N % 16 || K % 16) {
        printf("M, N, K must be multiples of 16 (got %d %d %d)\n", M, N, K);
        return 1;
    }

    //printf("M=%d, N=%d, K=%d, tensor_iters=%d\n", M, N, K, tensor_iters);

    size_t aBytes = (size_t)M * K * sizeof(half);
    size_t bBytes = (size_t)K * N * sizeof(half);
    size_t cBytes = (size_t)M * N * sizeof(float);

    half *A, *B;
    float *C;
    cudaMalloc(&A, aBytes);
    cudaMalloc(&B, bBytes);
    cudaMalloc(&C, cBytes);

    // Host init
    half *hA = (half*)malloc(aBytes);
    half *hB = (half*)malloc(bBytes);
    init_half(hA, M * K, 1.0f);
    init_half(hB, K * N, 2.0f);
    cudaMemcpy(A, hA, aBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bBytes, cudaMemcpyHostToDevice);

    dim3 block(32, 1, 1);          // one warp per block
    dim3 grid(N / 16, M / 16, 1);  // one block per 16x16 tile of C

    // Warmup
    wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, tensor_iters);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K, tensor_iters);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("Tensor Core WMMA GEMM: M=%d tensor_iters=%d time=%.3f ms\n", M, tensor_iters, ms);

    printf("Tensor Core WMMA GEMM: %dx%dx%d tensor_iters=%d time=%.3f ms\n",
       M, N, K, tensor_iters, ms);

    // Optional: read back a few values to ensure activity
    float sample = 0.f;
    cudaMemcpy(&sample, C + (M / 2) * N + (N / 2), sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sample C[mid,mid]=%f\n", sample);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(hA);
    free(hB);

    return 0;
}

