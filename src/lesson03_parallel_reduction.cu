#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void sum_squares_reduction(const float * x, float * partial, int n)
{
  __shared__ float cache[256];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;

  float value = 0.0F;
  if (gid < n) {
    value = x[gid] * x[gid];
  }
  cache[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      cache[tid] += cache[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial[blockIdx.x] = cache[0];
  }
}

int main()
{
  try {
    constexpr int n = 1 << 20;
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    std::vector<float> h_x(n);
    for (int i = 0; i < n; ++i) {
      h_x[i] = std::sin(0.001F * static_cast<float>(i));
    }

    float * d_x = nullptr;
    float * d_partial = nullptr;
    std::vector<float> h_partial(blocks, 0.0F);

    gclp::throw_on_cuda_error(cudaMalloc(&d_x, n * sizeof(float)), "cudaMalloc d_x");
    gclp::throw_on_cuda_error(cudaMalloc(&d_partial, blocks * sizeof(float)), "cudaMalloc d_partial");

    gclp::throw_on_cuda_error(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    sum_squares_reduction<<<blocks, threads>>>(d_x, d_partial, n);
    gclp::throw_on_cuda_error(cudaGetLastError(), "sum_squares_reduction launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    gclp::throw_on_cuda_error(
      cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost),
      "D2H partial");

    double gpu_sum = 0.0;
    for (float v : h_partial) {
      gpu_sum += static_cast<double>(v);
    }

    double cpu_sum = 0.0;
    for (float v : h_x) {
      cpu_sum += static_cast<double>(v) * static_cast<double>(v);
    }

    std::cout << "Lesson 03: Parallel reduction\n";
    std::cout << "gpu_sum=" << gpu_sum << " cpu_sum=" << cpu_sum << " abs_err=" << std::abs(gpu_sum - cpu_sum) << "\n";

    cudaFree(d_x);
    cudaFree(d_partial);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
