// tc_concurrency_bench.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <random>

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " \
            << cudaGetErrorString(err) << std::endl; std::exit(1);} } while(0)

#define CHECK_CUBLAS(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { \
  std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " -> " << st << std::endl; std::exit(1);} } while(0)

#define CHECK_NVML(x) do { nvmlReturn_t r = (x); if (r != NVML_SUCCESS) { \
  std::cerr << "NVML error at " << __FILE__ << ":" << __LINE__ << " -> " \
            << nvmlErrorString(r) << std::endl; std::exit(1);} } while(0)

struct Args {
  bool run_cuda = false;
  bool run_tensor = false;
  int ms_target = 1000; // target ~duration per workload to get meaningful power
  int M = 4096, N = 4096, K = 4096; // GEMM dims for tensor op
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s=="--cuda") a.run_cuda = true;
    else if (s=="--tensor") a.run_tensor = true;
    else if (s=="--both") { a.run_cuda = true; a.run_tensor = true; }
    else if (s=="--ms" && i+1<argc) a.ms_target = std::stoi(argv[++i]);
    else if (s=="--m" && i+1<argc) a.M = std::stoi(argv[++i]);
    else if (s=="--n" && i+1<argc) a.N = std::stoi(argv[++i]);
    else if (s=="--k" && i+1<argc) a.K = std::stoi(argv[++i]);
    else if (s=="--help") {
      std::cout <<
      "Usage: tcbench [--cuda|--tensor|--both] [--ms <approx per-workload ms>] [--m M --n N --k K]\n";
      std::exit(0);
    }
  }
  if (!a.run_cuda && !a.run_tensor) a.run_cuda = true; // default
  return a;
}

// ---------------- CUDA-core workload -----------------
// Compute-bound FP32 FMAs in registers to saturate CUDA cores.
__global__ void fma_burn(float* __restrict__ out, const float* __restrict__ in, int elems, int iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elems) return;
  float x = in[idx];
  float a = 1.000001f, b = 0.999973f, c = 1.414213f;
  // unrolled FMAs
  #pragma unroll 4
  for (int i=0;i<iters;i++) {
    x = fmaf(x, a, b);
    x = fmaf(x, c, a);
    x = fmaf(x, b, c);
    x = fmaf(x, a, b);
  }
  out[idx] = x;
}

float run_cuda_core_work(int target_ms, cudaStream_t stream, bool verbose=true) {
  const int elems = 1<<24; // ~16.7M floats (~64MB in+out total)
  float *d_in = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, elems*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, elems*sizeof(float)));

  // init
  std::vector<float> h(elems, 1.0f);
  CHECK_CUDA(cudaMemcpyAsync(d_in, h.data(), elems*sizeof(float), cudaMemcpyHostToDevice, stream));

  // choose iters to hit ~target_ms (rough heuristic; will vary by GPU)
  int iters = 256;
  // quick calibrate
  {
    dim3 blk(256), grid((elems+blk.x-1)/blk.x);
    cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s, stream));
    fma_burn<<<grid, blk, 0, stream>>>(d_out, d_in, elems, iters);
    CHECK_CUDA(cudaEventRecord(e, stream));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    if (ms>0) {
      float scale = target_ms / ms;
      iters = std::max(1, int(iters * scale));
    }
  }

  // timed run
  cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  dim3 blk(256), grid((elems+blk.x-1)/blk.x);
  CHECK_CUDA(cudaEventRecord(s, stream));
  fma_burn<<<grid, blk, 0, stream>>>(d_out, d_in, elems, iters);
  CHECK_CUDA(cudaEventRecord(e, stream));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));

  if (verbose) {
    std::cout << "[CUDA-CORE] elems="<<elems<<", iters="<<iters<<", time_ms="<<ms<<"\n";
  }

  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_in));
  return ms;
}

// ---------------- Tensor Core workload (cuBLAS HGEMM) -----------------
#include <cuda_fp16.h>

float run_tensor_core_gemm(int M, int N, int K, cudaStream_t stream, bool verbose=true) {
  // Sanity: multiples for TC alignment improve performance (e.g., 128)
  auto align = [](int x){ return (x + 127) / 128 * 128; };
  M = align(M); N = align(N); K = align(K);

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  // Enable Tensor Core math
#if CUDART_VERSION >= 11000
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#else
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
#endif

  __half *A=nullptr, *B=nullptr;
  float *C=nullptr;
  size_t szA = size_t(M)*K, szB = size_t(K)*N, szC = size_t(M)*N;
  CHECK_CUDA(cudaMalloc(&A, szA*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&B, szB*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&C, szC*sizeof(float)));

  // init host buffers with small randoms
  std::vector<__half> hA(szA), hB(szB);
  std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f,1.f);
  for (size_t i=0;i<szA;i++) hA[i] = __float2half(dist(rng));
  for (size_t i=0;i<szB;i++) hB[i] = __float2half(dist(rng));
  CHECK_CUDA(cudaMemcpyAsync(A, hA.data(), szA*sizeof(__half), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(B, hB.data(), szB*sizeof(__half), cudaMemcpyHostToDevice, stream));

  const float alpha = 1.0f, beta = 0.0f;

  // Time GEMM
  cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s, stream));
  // C = alpha*A*B + beta*C ; A: MxK, B: KxN, C: MxN
  CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, CUDA_R_16F, N,
      A, CUDA_R_16F, K,
      &beta,
      C, CUDA_R_32F, N,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaEventRecord(e, stream));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));

  if (verbose) {
    double flops = 2.0 * (double)M * N * K; // MAC = 2 flops
    double tsec = ms * 1e-3;
    double tflops = flops / tsec / 1e12;
    std::cout << "[TENSOR] GEMM " << M<<"x"<<N<<"x"<<K
              << " time_ms="<<ms<<" ~"<<tflops<<" TFLOP/s\n";
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(C));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(A));
  return ms;
}

// --------------- NVML power sampler -------------------
struct PowerSample { double t_ms; double watts; };

struct PowerLog {
  std::vector<PowerSample> s;
  double avg_w=0, peak_w=0, energy_j=0;
};

PowerLog integrate_power(const std::vector<PowerSample>& v) {
  PowerLog out; out.s = v;
  if (v.empty()) return out;
  double sum=0, peak=0, energy=0;
  for (size_t i=0;i<v.size();++i) {
    sum += v[i].watts;
    peak = std::max(peak, v[i].watts);
    if (i>0) {
      double dt = (v[i].t_ms - v[i-1].t_ms) / 1000.0;
      energy += 0.5 * (v[i].watts + v[i-1].watts) * dt;
    }
  }
  out.avg_w = sum / v.size();
  out.peak_w = peak;
  out.energy_j = energy;
  return out;
}

struct Sampler {
  std::atomic<bool> run{false};
  std::vector<PowerSample> buf;
  std::thread th;
  nvmlDevice_t dev;
  std::chrono::steady_clock::time_point t0;

  void start() {
    CHECK_NVML(nvmlInit_v2());
    CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(0, &dev));
    run.store(true);
    buf.clear();
    t0 = std::chrono::steady_clock::now();
    th = std::thread([&]{
      while (run.load(std::memory_order_relaxed)) {
        unsigned int mW=0;
        if (nvmlDeviceGetPowerUsage(dev, &mW) == NVML_SUCCESS) {
          auto now = std::chrono::steady_clock::now();
          double tms = std::chrono::duration<double,std::milli>(now - t0).count();
          buf.push_back({tms, mW/1000.0});
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }
  PowerLog stop() {
    run.store(false);
    if (th.joinable()) th.join();
    CHECK_NVML(nvmlShutdown());
    return integrate_power(buf);
  }
};

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  // Device check
  cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";
  if (args.run_tensor && prop.major < 7) {
    std::cerr << "This GPU does not support Tensor Cores (needs cc >= 7.0). Exiting.\n";
    return 1;
  }

  // Create streams
  cudaStream_t s_cuda = nullptr, s_tc = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&s_cuda, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&s_tc, cudaStreamNonBlocking));

  // Power sampler
  Sampler sampler;
  sampler.start();

  // Record overall time
  cudaEvent_t g_s, g_e; CHECK_CUDA(cudaEventCreate(&g_s)); CHECK_CUDA(cudaEventCreate(&g_e));
  CHECK_CUDA(cudaEventRecord(g_s));

  float t_cuda_ms = 0.0f, t_tc_ms = 0.0f;

  if (args.run_cuda && args.run_tensor) {
    // CONCURRENT
    std::cout << "Mode: CONCURRENT (CUDA-core + Tensor Core)\n";
    // Launch both without sync; sync both at end
    // We call CUDA-core first to get calibration duration, but run the actual kernel concurrently.
    // For concurrency, run CUDA-core as-is and tensor GEMM as-is; no dependencies between streams.

    // Launch both
    std::thread th_cuda([&](){
      t_cuda_ms = run_cuda_core_work(args.ms_target, s_cuda, /*verbose*/false);
    });
    std::thread th_tc([&](){
      t_tc_ms = run_tensor_core_gemm(args.M, args.N, args.K, s_tc, /*verbose*/false);
    });
    th_cuda.join(); th_tc.join();

    CHECK_CUDA(cudaStreamSynchronize(s_cuda));
    CHECK_CUDA(cudaStreamSynchronize(s_tc));
    std::cout << "[CUDA-CORE] time_ms="<<t_cuda_ms<<"\n";
    std::cout << "[TENSOR]    time_ms="<<t_tc_ms<<"\n";
  } else if (args.run_cuda) {
    std::cout << "Mode: CUDA-core only\n";
    t_cuda_ms = run_cuda_core_work(args.ms_target, s_cuda, /*verbose*/true);
    CHECK_CUDA(cudaStreamSynchronize(s_cuda));
  } else if (args.run_tensor) {
    std::cout << "Mode: Tensor Core only\n";
    t_tc_ms = run_tensor_core_gemm(args.M, args.N, args.K, s_tc, /*verbose*/true);
    CHECK_CUDA(cudaStreamSynchronize(s_tc));
  }

  CHECK_CUDA(cudaEventRecord(g_e));
  CHECK_CUDA(cudaEventSynchronize(g_e));
  float t_total_ms=0; CHECK_CUDA(cudaEventElapsedTime(&t_total_ms, g_s, g_e));
  CHECK_CUDA(cudaEventDestroy(g_s)); CHECK_CUDA(cudaEventDestroy(g_e));

  auto plog = sampler.stop();

  std::cout << "---------- Results ----------\n";
  std::cout << "Total wall time: " << t_total_ms << " ms\n";
  if (args.run_cuda)  std::cout << "CUDA-core time:  " << t_cuda_ms  << " ms\n";
  if (args.run_tensor)std::cout << "Tensor time:     " << t_tc_ms    << " ms\n";
  std::cout << "Power (avg):     " << plog.avg_w  << " W\n";
  std::cout << "Power (peak):    " << plog.peak_w << " W\n";
  std::cout << "Energy (est.):   " << plog.energy_j << " J\n";
  std::cout << "Samples:         " << plog.s.size() << "\n";

  CHECK_CUDA(cudaStreamDestroy(s_tc));
  CHECK_CUDA(cudaStreamDestroy(s_cuda));
  return 0;
}
