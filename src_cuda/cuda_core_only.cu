// cuda_core_only.cu
// Build: nvcc -O3 -arch=sm_80 cuda_core_only.cu -o cuda_core_only
// Run: ./cuda_core_only [N=1<<24] [iters=1024]
// Run: ./cuda_core_only 16777216 1024


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
  cudaError_t err__ = (x); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while(0)

__global__ void cuda_core_fma_kernel(const float* __restrict__ a, 
                                    const float* __restrict__ b, 
                                    float* __restrict__ c, 
                                    int N, int iters) {
    
    //printf("in cuda_core_fma_kernel. \n");
    
    // Grid-stride loop over elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        float x = a[idx];                       //Loads the per-element inputs into registers x and y
        float y = b[idx];
        float acc = 0.f;                        // initializes an accumulator acc to 0
        // Compute-heavy loop: lots of FMAs on CUDA cores
        #pragma unroll 4                  
                                                // hints the compiler to unroll the loop by a factor of 4 to increase instruction-level parallelism
        for (int k = 0; k < iters; ++k) {       // loop runs iters times per element.
            acc = fmaf(x, y, acc);              // acc = x*y + acc. Per iteration, perform a fused multiply-add
            // Decorrelate operands a bit
            x = fmaf(x, 1.000001f, 0.000001f);  // change slightly so compiler doesnt optimize away the FMA
            y = fmaf(y, 0.999999f, 0.000002f);
        }
    c[idx] = acc;                               //Stores result back to global memory so the work has a visible side-effect. further prevents dead-code elimination.
    }
}


int main(int argc, char** argv) {

    setvbuf(stdout, nullptr, _IONBF, 0);  // disable buffering

    //printf("in main.\n");

    // arguments
    int N = (argc > 1) ? atoi(argv[1]) : (1<<24);   // number of elements. Default = 2^24=16,777,216
    int iters = (argc > 2) ? atoi(argv[2]) : 1024;  // per-element FMAs

    //printf("after arguments.\n");

    // show device for sanity
    int dev = 0;
    //fprintf(stderr, "before cudaSetDevice\n"); //
    CHECK_CUDA(cudaSetDevice(dev));
    //fprintf(stderr, "after cudaSetDevice\n"); //
    cudaDeviceProp prop{};
    //fprintf(stderr, "before get props\n"); //
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    //fprintf(stderr, "after get props\n"); //
    // printf("GPU: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N=%d, iters=%d\n", N, iters);
    //CHECK_CUDA(cudaFree(0));  // creates context without allocating anything



    size_t bytes = N * sizeof(float);               // Computes the total byte size of each array.
    float *a, *b, *c;                               // Declares device pointers a, b, c.
    CHECK_CUDA(cudaMalloc(&a, bytes));                          // Allocates device memory for device pointers. 
    CHECK_CUDA(cudaMalloc(&b, bytes));
    CHECK_CUDA(cudaMalloc(&c, bytes));


    // Allocates host memory to create initial data for a and b.
    float *ha = (float*)malloc(bytes);
    float *hb = (float*)malloc(bytes);

    // Fills host arrays with small repeating patterns (modulo arithmetic) to avoid all-constant input, which can lead to unrealistic compiler optimizations.
    for (int i = 0; i < N; ++i) { 
        ha[i] = 1.0f + (i % 13) * 0.001f; 
        hb[i] = 2.0f + (i % 7) * 0.002f; 
    }
    CHECK_CUDA(cudaMemcpy(a, ha, bytes, cudaMemcpyHostToDevice));   // Copies initialized data from host to device.
    CHECK_CUDA(cudaMemcpy(b, hb, bytes, cudaMemcpyHostToDevice));


    // Launch config (tune as needed)
    int block = 256;
    int grid = (N + block - 1) / block;
    grid = min(grid, 65535); // cap grid size for safety. If grid size exceeds the device’s limit, the kernel launch will fail.


    // Warmup
    //printf("start warmup.\n");
    cuda_core_fma_kernel<<<grid, block>>>(a, b, c, N, iters);
    CHECK_CUDA(cudaPeekAtLastError());           // catch launch errors early
    CHECK_CUDA(cudaDeviceSynchronize());
    //printf("end warmup.\n");


    // Timed run
    //printf("start timed run.\n");
    cudaEvent_t start, stop; 
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    cuda_core_fma_kernel<<<grid, block>>>(a, b, c, N, iters);

    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    //printf("end timed run.\n");
    float ms = 0.f; 
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));     // Now that both events have actual timestamps, compute the elapsed time in milliseconds.

    printf("CUDA-core kernel: N=%d iters=%d time=%.3f ms\n", N, iters, ms);


    // Cleanup
    //printf("start cleanup.\n");
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    free(ha); free(hb);
    //printf("end cleanup.\n");
    return 0;
}