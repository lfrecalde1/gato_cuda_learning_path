#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

namespace
{

void throw_on_cublas(cublasStatus_t status, const char * context)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS call failed");
  }
}

void matmul_cpu_naive(
  const std::vector<float> & A,
  const std::vector<float> & B,
  std::vector<float> & C,
  int m,
  int n,
  int k)
{
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < k; ++c) {
      float sum = 0.0F;
      for (int i = 0; i < n; ++i) {
        sum += A[static_cast<std::size_t>(r * n + i)] * B[static_cast<std::size_t>(i * k + c)];
      }
      C[static_cast<std::size_t>(r * k + c)] = sum;
    }
  }
}

__global__ void matmul_gpu_naive(const float * A, const float * B, float * C, int m, int n, int k)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < k) {
    float sum = 0.0F;
    for (int i = 0; i < n; ++i) {
      sum += A[row * n + i] * B[i * k + col];
    }
    C[row * k + col] = sum;
  }
}

template<int TILE>
__global__ void matmul_gpu_tiled(const float * A, const float * B, float * C, int m, int n, int k)
{
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0F;
  const int tiles = (n + TILE - 1) / TILE;

  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
      (row < m && a_col < n) ? A[row * n + a_col] : 0.0F;
    Bs[threadIdx.y][threadIdx.x] =
      (b_row < n && col < k) ? B[b_row * k + col] : 0.0F;

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < TILE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < k) {
    C[row * k + col] = sum;
  }
}

float measure_kernel_ms(cudaStream_t stream, int repeats, const std::function<void()> & launch)
{
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  gclp::throw_on_cuda_error(cudaEventCreate(&start), "cudaEventCreate start");
  gclp::throw_on_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Warmup
  launch();
  gclp::throw_on_cuda_error(cudaGetLastError(), "kernel launch warmup");

  gclp::throw_on_cuda_error(cudaEventRecord(start, stream), "cudaEventRecord start");
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  gclp::throw_on_cuda_error(cudaEventRecord(stop, stream), "cudaEventRecord stop");
  gclp::throw_on_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float total_ms = 0.0F;
  gclp::throw_on_cuda_error(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total_ms / static_cast<float>(repeats);
}

}  // namespace

int main()
{
  try {
    constexpr int m = 512;
    constexpr int n = 512;
    constexpr int k = 512;
    constexpr int repeats = 20;

    std::vector<float> h_A(static_cast<std::size_t>(m * n));
    std::vector<float> h_B(static_cast<std::size_t>(n * k));
    std::vector<float> h_C_cpu(static_cast<std::size_t>(m * k));
    std::vector<float> h_C_naive(static_cast<std::size_t>(m * k));
    std::vector<float> h_C_tiled(static_cast<std::size_t>(m * k));
    std::vector<float> h_C_cublas(static_cast<std::size_t>(m * k));

    for (int i = 0; i < m * n; ++i) {
      h_A[static_cast<std::size_t>(i)] = 0.005F * static_cast<float>(i % 31);
    }
    for (int i = 0; i < n * k; ++i) {
      h_B[static_cast<std::size_t>(i)] = 0.004F * static_cast<float>((i + 11) % 37);
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_C = nullptr;
    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, h_B.size() * sizeof(float)), "cudaMalloc d_B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_C, h_C_cublas.size() * sizeof(float)), "cudaMalloc d_C");
    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");

    cudaStream_t stream = nullptr;
    gclp::throw_on_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    throw_on_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    const dim3 block_naive(16, 16);
    const dim3 grid_naive(
      static_cast<unsigned>((k + block_naive.x - 1) / block_naive.x),
      static_cast<unsigned>((m + block_naive.y - 1) / block_naive.y));

    const dim3 block_tiled(16, 16);
    const dim3 grid_tiled(
      static_cast<unsigned>((k + 15) / 16),
      static_cast<unsigned>((m + 15) / 16));

    const float gpu_naive_ms = measure_kernel_ms(stream, repeats, [&]() {
      matmul_gpu_naive<<<grid_naive, block_naive, 0, stream>>>(d_A, d_B, d_C, m, n, k);
    });
    gclp::throw_on_cuda_error(cudaMemcpy(h_C_naive.data(), d_C, h_C_naive.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H naive");

    const float gpu_tiled_ms = measure_kernel_ms(stream, repeats, [&]() {
      matmul_gpu_tiled<16><<<grid_tiled, block_tiled, 0, stream>>>(d_A, d_B, d_C, m, n, k);
    });
    gclp::throw_on_cuda_error(cudaMemcpy(h_C_tiled.data(), d_C, h_C_tiled.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H tiled");

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const float gpu_cublas_ms = measure_kernel_ms(stream, repeats, [&]() {
      // row-major C = A * B by swapping operands in column-major GEMM
      throw_on_cublas(
        cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          k,
          m,
          n,
          &alpha,
          d_B,
          k,
          d_A,
          n,
          &beta,
          d_C,
          k),
        "cublasSgemm");
    });
    gclp::throw_on_cuda_error(cudaMemcpy(h_C_cublas.data(), d_C, h_C_cublas.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H cublas");

    gclp::ScopedWallTimer cpu_timer;
    matmul_cpu_naive(h_A, h_B, h_C_cpu, m, n, k);
    const double cpu_naive_ms = cpu_timer.elapsed_ms();

    auto max_abs_error = [](const std::vector<float> & lhs, const std::vector<float> & rhs) {
      double max_err = 0.0;
      for (std::size_t i = 0; i < lhs.size(); ++i) {
        max_err = std::max(max_err, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
      }
      return max_err;
    };

    const double err_cpu = max_abs_error(h_C_cpu, h_C_cublas);
    const double err_naive = max_abs_error(h_C_naive, h_C_cublas);
    const double err_tiled = max_abs_error(h_C_tiled, h_C_cublas);

    std::cout << "Lesson 11: MatMul comparison CPU vs GPU naive vs GPU tiled vs cuBLAS\n";
    std::cout << "Dims: " << m << "x" << n << " * " << n << "x" << k << "\n";
    std::cout << "CPU naive ms:   " << cpu_naive_ms << "\n";
    std::cout << "GPU naive ms:   " << gpu_naive_ms << "\n";
    std::cout << "GPU tiled ms:   " << gpu_tiled_ms << "\n";
    std::cout << "GPU cuBLAS ms:  " << gpu_cublas_ms << "\n";
    std::cout << "Speedup vs CPU (naive/tiled/cublas): "
              << (cpu_naive_ms / std::max<double>(gpu_naive_ms, 1.0e-9)) << "x / "
              << (cpu_naive_ms / std::max<double>(gpu_tiled_ms, 1.0e-9)) << "x / "
              << (cpu_naive_ms / std::max<double>(gpu_cublas_ms, 1.0e-9)) << "x\n";
    std::cout << "Max abs error vs cuBLAS (cpu/naive/tiled): "
              << err_cpu << " / " << err_naive << " / " << err_tiled << "\n";

    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
