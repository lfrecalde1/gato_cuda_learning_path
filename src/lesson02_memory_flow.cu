#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void vector_add_kernel(const float * a, const float * b, float * c, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main()
{
  try {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
      h_a[i] = 0.001F * static_cast<float>(i);
      h_b[i] = 2.0F * std::sin(0.002F * static_cast<float>(i));
    }

    float * d_a = nullptr;
    float * d_b = nullptr;
    float * d_c = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_a, n * sizeof(float)), "cudaMalloc d_a");
    gclp::throw_on_cuda_error(cudaMalloc(&d_b, n * sizeof(float)), "cudaMalloc d_b");
    gclp::throw_on_cuda_error(cudaMalloc(&d_c, n * sizeof(float)), "cudaMalloc d_c");

    gclp::ScopedWallTimer timer;

    gclp::throw_on_cuda_error(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    gclp::throw_on_cuda_error(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    gclp::throw_on_cuda_error(cudaGetLastError(), "vector_add_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    gclp::throw_on_cuda_error(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H c");

    const double elapsed_ms = timer.elapsed_ms();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
      const double ref = static_cast<double>(h_a[i]) + static_cast<double>(h_b[i]);
      const double err = std::abs(ref - static_cast<double>(h_c[i]));
      if (err > max_abs_err) {
        max_abs_err = err;
      }
    }

    std::cout << "Lesson 02: Host->Device, kernel, Device->Host flow\n";
    std::cout << "n=" << n << " elapsed_ms=" << elapsed_ms << " max_abs_err=" << max_abs_err << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
