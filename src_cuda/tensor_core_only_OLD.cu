// tensor_core_only.cu
// Build: nvcc -arch=sm_86 tensor_core_only.cu -o tensor_core_only
// Run: ./tensor_core_only 4096 4096 4096

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

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    int row = tileRow * 16;                           
    int col = tileCol * 16;   
    
    // Initialize accumulator once
    wmma::fill_fragment(cFrag, 0.0f);

    // NEW: repeat the GEMM body tensor_iters times
    for (int rep = 0; rep < tensor_iters; ++rep) {

        for (int k = 0; k < K; k += 16) {                  // Tensor Cores multiply 16×16×16 chunks; loop over K in steps of 16 to accumulate the full dot product.
            const half* tileA = A + row * K + k;
            const half* tileB = B + k * N + col;
            wmma::load_matrix_sync(aFrag, tileA, K);
            wmma::load_matrix_sync(bFrag, tileB, N);
            wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        }
    }
    
    float* tileC = C + row * N + col;
    wmma::store_matrix_sync(tileC, cFrag, N, wmma::mem_row_major);     // Store C tile (row-major)
}

// initialize an array of half with a small repeating pattern starting at seed. Converts from float to half using __float2half.
static void init_half(half* h, int n, float seed) {
    for (int i = 0; i < n; ++i) {
        float v = seed + (i % 13) * 0.01f;
        h[i] = __float2half(v);
    }
}


int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);                 // disable buffering

    printf("in main.\n");

    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;

    if (M % 16 || N % 16 || K % 16) {                    // WMMA tiles are 16×16×16; dimensions must be multiples of 16 to tile exactly.
        printf("M, N, K must be multiples of 16 (got %d %d %d)\n", M, N, K);
        return 1;
    }

    size_t aBytes = (size_t)M * K * sizeof(half);        // Compute byte size for each array a, b, c.
    size_t bBytes = (size_t)K * N * sizeof(half);
    size_t cBytes = (size_t)M * N * sizeof(float);

    half *A, *B; float *C;
    cudaMalloc(&A, aBytes); cudaMalloc(&B, bBytes); cudaMalloc(&C, cBytes);  // Allocate device memory for A, B (FP16) and C (FP32).

    // Host init
    half *hA = (half*)malloc(aBytes), *hB = (half*)malloc(bBytes);    // Allocate host buffers
    init_half(hA, M*K, 1.0f); init_half(hB, K*N, 2.0f);               // initialize hos buffers with simple patterns (different seeds so A and B differ) using the helper function from earlier. 
    cudaMemcpy(A, hA, aBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bBytes, cudaMemcpyHostToDevice);

    dim3 block(32, 1, 1);                        // Launch one warp (32 threads) per block; WMMA fragments are warp-level primitives.
    dim3 grid(N/16, M/16, 1);                    // 2D grid: one block per 16×16 tile of C.

    // Warmup
    wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);     // Launch once as a warmup and synchronize to ensure completion (helpful to stabilize clocks/JIT and catch errors early).
    cudaDeviceSynchronize();

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f; cudaEventElapsedTime(&ms, start, stop);
    printf("Tensor Core WMMA GEMM: %dx%dx%d time=%.3f ms\n", M, N, K, ms);


    // Optional: read back a few values to ensure activity
    float sample = 0.f;                                                                 // host-side variable to hold one value from C.
    cudaMemcpy(&sample, C + (M/2)*N + (N/2), sizeof(float), cudaMemcpyDeviceToHost);    // Copies that single float from device memory (C) into host memory (sample).
    /*
    C is a pointer to the start of the C matrix in device memory. 
    C is stored row-major with leading dimension N. 
    Indexing the middle element: Middle row index = M/2, Middle column index = N/2.
    Row-major offset = row * N + col = (M/2) * N + (N/2). 
    So C + (M/2)*N + (N/2) is a pointer to C[M/2][N/2] on the GPU.

    This is a sanity check to confirm: a) The kernel actually wrote something to C. b) Data transfer works.
    */ 
    printf("Sample C[mid,mid]=%f\n", sample);                // Prints the value of the center element of C so you can see the result is nonzero and plausible.


    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    free(hA); free(hB);

    return 0;
}