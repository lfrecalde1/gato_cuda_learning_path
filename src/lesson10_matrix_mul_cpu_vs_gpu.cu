#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void matmul_naive_kernel(const float * A, const float * B, float * C, int m, int n, int k)
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

void matmul_cpu(
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

int main()
{
  try {
    constexpr int m = 512;
    constexpr int n = 512;
    constexpr int k = 512;

    std::vector<float> h_A(m * n), h_B(n * k), h_C_gpu(m * k), h_C_cpu(m * k);

    for (int i = 0; i < m * n; ++i) {
      h_A[i] = 0.01F * static_cast<float>(i % 29);
    }
    for (int i = 0; i < n * k; ++i) {
      h_B[i] = 0.01F * static_cast<float>((i + 3) % 31);
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_C = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, h_B.size() * sizeof(float)), "cudaMalloc d_B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_C, h_C_gpu.size() * sizeof(float)), "cudaMalloc d_C");

    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    constexpr dim3 block(16, 16);
    const dim3 grid(
      static_cast<unsigned>((k + block.x - 1) / block.x),
      static_cast<unsigned>((m + block.y - 1) / block.y));

    gclp::ScopedWallTimer gpu_timer;
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    gclp::throw_on_cuda_error(cudaGetLastError(), "matmul_naive_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize GPU");
    const double gpu_ms = gpu_timer.elapsed_ms();

    gclp::throw_on_cuda_error(cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    gclp::ScopedWallTimer cpu_timer;
    matmul_cpu(h_A, h_B, h_C_cpu, m, n, k);
    const double cpu_ms = cpu_timer.elapsed_ms();

    double max_err = 0.0;
    for (std::size_t i = 0; i < h_C_cpu.size(); ++i) {
      const double err = std::abs(static_cast<double>(h_C_gpu[i]) - static_cast<double>(h_C_cpu[i]));
      if (err > max_err) {
        max_err = err;
      }
    }

    const double speedup = cpu_ms / std::max(gpu_ms, 1.0e-9);

    std::cout << "Lesson 10: Matrix multiplication CPU vs GPU\n";
    std::cout << "shape=" << m << "x" << n << " * " << n << "x" << k << "\n";
    std::cout << "gpu_ms=" << gpu_ms << " cpu_ms=" << cpu_ms << " speedup=" << speedup << "x max_err=" << max_err << "\n";
    std::cout << "This demonstrates why GPU is powerful for dense linear algebra workloads.\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
