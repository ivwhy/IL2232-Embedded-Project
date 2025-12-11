// concurrent_cuda_tensor.cu
// Build: nvcc -O3 -arch=sm_80 concurrent_cuda_tensor.cu -o concurrent_cuda_tensor
// Run: ./concurrent_cuda_tensor [N=1<<24] [iters=1024] [M=1024] [N=1024] [K=1024]
// Run: ./concurrent_cuda_tensor Nvec iters M N K tensor_iters repeats 
// Run: ./concurrent_cuda_tensor 256 100 4096 4096 4096 100 100
// Run: ncu concurrent_cuda_tensor 256 150000000 16 16 16 10000000 3

#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <fstream>
using namespace nvcuda;


// ---- CUDA-core kernel (same as in program #1) ----
__global__ void cuda_core_fma_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int N, int iters) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        float x = a[idx];
        float y = b[idx];
        float acc = 0.f;
        #pragma unroll 4
        for (int k = 0; k < iters; ++k) {
            acc = fmaf(x, y, acc);
            x = fmaf(x, 1.000001f, 0.000001f);
            y = fmaf(y, 0.999999f, 0.000002f);
        }
        c[idx] = acc;
    }
}


// ---- Tensor Core WMMA kernel (same as in program #2) ----
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

// initialize an array of half
static void init_half(half* h, int n, float seed) { 
    for (int i = 0; i < n; ++i){
        h[i] = __float2half(seed + (i % 13) * 0.01f); 
    }
}


bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void append_csv_row(int Nvec, int iters, int tensor_iters,
                    int M, int N, int K, int repeats,
                    float cuda_ms, float tensor_ms,
                    float serialized_ms, float conc_ms)
{
    const char* filename = "results_cuda.csv";

    bool exists = file_exists(filename);

    std::ofstream file;
    file.open(filename, std::ios::app);

    if (!exists) {
        // Write header once
        file << "Nvec,iters,tensor_iters,M,N,K,repeats,"
                "cuda_ms,tensor_ms,serialized_ms,concurrent_ms\n";
    }

    // Write the actual row
    file << Nvec << ","
         << iters << ","
         << tensor_iters << ","
         << M << ","
         << N << ","
         << K << ","
         << repeats << ","
         << cuda_ms << ","
         << tensor_ms << ","
         << serialized_ms << ","
         << conc_ms << "\n";

    file.close();
}


int main(int argc, char** argv) {
    // Vector work (CUDA cores)
    int Nvec = (argc > 1) ? atoi(argv[1]) : (1<<24);
    int iters = (argc > 2) ? atoi(argv[2]) : 1024;
    
    // Matrix sizes (Tensor Cores)
    int M = (argc > 3) ? atoi(argv[3]) : 1024;
    int N = (argc > 4) ? atoi(argv[4]) : 1024;
    int K = (argc > 5) ? atoi(argv[5]) : 1024;
    int tensor_iters = (argc > 6) ? atoi(argv[6]) : 1024; // how many times to repeat the WMMA GEMM inside the kernel

    if (M % 16 || N % 16 || K % 16) { printf("M,N,K must be multiples of 16.\n"); return 1; }

    // how many times to repeat the timed kernels
    int repeats = (argc > 7) ? atoi(argv[7]) : 1;

    // --- Allocate & init CUDA vector data ---
    size_t vbytes = (size_t)Nvec * sizeof(float);
    float *a, *b, *c; 
    cudaMalloc(&a, vbytes); cudaMalloc(&b, vbytes); cudaMalloc(&c, vbytes);
    float *ha = (float*)malloc(vbytes), *hb = (float*)malloc(vbytes);
    for (int i = 0; i < Nvec; ++i) { ha[i] = 1.0f + (i % 13) * 0.001f; hb[i] = 2.0f + (i % 7) * 0.002f; }
    cudaMemcpy(a, ha, vbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, vbytes, cudaMemcpyHostToDevice);


    // --- Allocate & init Tensor matrix data ---
    size_t aBytes = (size_t)M * K * sizeof(half);
    size_t bBytes = (size_t)K * N * sizeof(half);
    size_t cBytes = (size_t)M * N * sizeof(float);
    half *A, *B; float *C; cudaMalloc(&A, aBytes); 
    cudaMalloc(&B, bBytes); cudaMalloc(&C, cBytes);
    half *hA = (half*)malloc(aBytes), *hB = (half*)malloc(bBytes);
    init_half(hA, M*K, 1.0f); init_half(hB, K*N, 2.0f);
    cudaMemcpy(A, hA, aBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, bBytes, cudaMemcpyHostToDevice);


    // Launch configs
    int block = 256; int grid = (Nvec + block - 1) / block; grid = min(grid, 65535);   // For the CUDA-core vector kernel
    dim3 blockW(32,1,1); dim3 gridW(N/16, M/16, 1);                                    // For the WMMA kernel


    // Declare two streams
    cudaStream_t sCuda, sTensor; cudaStreamCreate(&sCuda); cudaStreamCreate(&sTensor);


    // --- Warmup in separate streams (help stabilize JIT compilation, GPU clock, avoid one-time overhead in perf. measurement)
    cuda_core_fma_kernel<<<grid, block, 0, sCuda>>>(a, b, c, Nvec, iters);  // Launch CUDA-core FMA kernel in stream sCuda
    wmma_gemm_kernel<<<gridW, blockW, 0, sTensor>>>(A, B, C, M, N, K, tensor_iters);      // Launch WMMA GEMM kernel in stream sTensor.
    cudaDeviceSynchronize();                                                // Note, may run concurrently already.


    // --- Timed: run separately (serialize) ---
    cudaEvent_t s1,s2,s3,s4; cudaEventCreate(&s1); cudaEventCreate(&s2); cudaEventCreate(&s3); cudaEventCreate(&s4);
    float ms_cuda_only=0; 
    float ms_cuda_sum=0;

    // CUDA-core only timing (stream sCuda)
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(s1, sCuda);
        cuda_core_fma_kernel<<<grid, block, 0, sCuda>>>(a, b, c, Nvec, iters); // Launch CUDA-core kernel in sCuda
        cudaEventRecord(s2, sCuda); // end of CUDA-core workload
        cudaEventSynchronize(s2);  // Wait for stream sCuda to finish

        cudaEventElapsedTime(&ms_cuda_only, s1, s2);
        ms_cuda_sum = ms_cuda_sum + ms_cuda_only;
    } 
    float ms_cuda_avg=0;
    ms_cuda_avg = ms_cuda_sum / repeats;
    //printf("debug print: ms_cuda_avg=%.3f ms, ms_cuda_sum=%.3f ms, repeats=%d \n", ms_cuda_avg, ms_cuda_sum, repeats);

    float ms_tensor_only=0; 
    float ms_tensor_sum=0;

    // Tensor core only timing (stream sTensor)
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(s3, sTensor); // start on sTensor
        wmma_gemm_kernel<<<gridW, blockW, 0, sTensor>>>(A, B, C, M, N, K, tensor_iters); //Launch WMMA GEMM kernel in stream sTensor
        cudaEventRecord(s4, sTensor); // end of Tensor-core workload
        cudaEventSynchronize(s4);

        cudaEventElapsedTime(&ms_tensor_only, s3, s4);
        ms_tensor_sum = ms_tensor_sum + ms_tensor_only;
    }
    float ms_tensor_avg=0;
    ms_tensor_avg = ms_tensor_sum / repeats;
    //printf("debug print: ms_tensor_avg=%.3f ms, ms_tensor_sum=%.3f ms, repeats=%d \n", ms_tensor_avg, ms_tensor_sum, repeats);

    printf("Serialized: CUDA-only (avg)=%.3f ms, Tensor-only (avg)=%.3f ms, sum (avg)=%.3f ms (repeats=%d)\n", ms_cuda_avg, ms_tensor_avg, ms_cuda_avg+ms_tensor_avg, repeats);


    // --- Timed: run concurrently ---
    cudaEvent_t cStart, cStop; cudaEventCreate(&cStart); cudaEventCreate(&cStop);
    float ms_conc=0;
    float ms_conc_sum=0;

    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(cStart);  // default stream, start timestamp
        cuda_core_fma_kernel<<<grid, block, 0, sCuda>>>(a, b, c, Nvec, iters);
        wmma_gemm_kernel<<<gridW, blockW, 0, sTensor>>>(A, B, C, M, N, K, tensor_iters);

        // Wait for ALL work in ALL streams to finish
        cudaDeviceSynchronize();

        cudaEventRecord(cStop);    // Mark end time
        cudaEventSynchronize(cStop); 

        cudaEventElapsedTime(&ms_conc, cStart, cStop);
        ms_conc_sum = ms_conc_sum + ms_conc;
    }
    float ms_conc_avg=0;
    ms_conc_avg = ms_conc_sum / repeats;

    printf("Concurrent total time (avg)=%.3f ms (repeats=%d)\n", ms_conc_avg, repeats);

    // Append results to CSV (with auto header)
    append_csv_row(
        Nvec, iters, tensor_iters,
        M, N, K, repeats,
        ms_cuda_avg, ms_tensor_avg,
        ms_cuda_avg + ms_tensor_avg,
        ms_conc_avg
    );

    // // --- Append results to CSV file ---
    // // Format matches run_sweep.py:
    // // Nvec, iters, tensor_iters, M, N, K, Repeats,
    // // cuda_ms, tensor_ms, serialized_ms, concurrent_ms
    // FILE* fp = fopen("results_cuda.csv", "a");
    // if (fp) {
    //     fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
    //             Nvec, iters, tensor_iters, M, N, K, repeats,
    //             ms_cuda_avg,
    //             ms_tensor_avg,
    //             ms_cuda_avg + ms_tensor_avg,
    //             ms_conc_avg);
    //     fclose(fp);
    // } else {
    //     printf("[WARN] Could not open results_cuda.csv for writing.\n");
    // }

    // Cleanup
    cudaEventDestroy(s1); cudaEventDestroy(s2); cudaEventDestroy(s3); cudaEventDestroy(s4); cudaEventDestroy(cStart); cudaEventDestroy(cStop);
    cudaStreamDestroy(sCuda); cudaStreamDestroy(sTensor);
    cudaFree(a); cudaFree(b); cudaFree(c); free(ha); free(hb);
    cudaFree(A); cudaFree(B); cudaFree(C); free(hA); free(hB);
    return 0;
}