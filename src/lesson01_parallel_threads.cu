#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void thread_map_kernel(int * out_global, int * out_block, int * out_thread, int total)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total) {
    out_global[gid] = gid;
    out_block[gid] = static_cast<int>(blockIdx.x);
    out_thread[gid] = static_cast<int>(threadIdx.x);
  }
}

int main()
{
  try {
    constexpr int total = 48;
    constexpr int threads_per_block = 16;
    const int blocks = (total + threads_per_block - 1) / threads_per_block;

    std::vector<int> h_global(total), h_block(total), h_thread(total);
    int * d_global = nullptr;
    int * d_block = nullptr;
    int * d_thread = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_global, total * sizeof(int)), "cudaMalloc d_global");
    gclp::throw_on_cuda_error(cudaMalloc(&d_block, total * sizeof(int)), "cudaMalloc d_block");
    gclp::throw_on_cuda_error(cudaMalloc(&d_thread, total * sizeof(int)), "cudaMalloc d_thread");

    thread_map_kernel<<<blocks, threads_per_block>>>(d_global, d_block, d_thread, total);
    gclp::throw_on_cuda_error(cudaGetLastError(), "thread_map_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    gclp::throw_on_cuda_error(
      cudaMemcpy(h_global.data(), d_global, total * sizeof(int), cudaMemcpyDeviceToHost),
      "cudaMemcpy global");
    gclp::throw_on_cuda_error(
      cudaMemcpy(h_block.data(), d_block, total * sizeof(int), cudaMemcpyDeviceToHost),
      "cudaMemcpy block");
    gclp::throw_on_cuda_error(
      cudaMemcpy(h_thread.data(), d_thread, total * sizeof(int), cudaMemcpyDeviceToHost),
      "cudaMemcpy thread");

    std::cout << "Lesson 01: CUDA thread model\n";
    std::cout << "grid=" << blocks << " blocks, blockDim=" << threads_per_block << "\n\n";
    std::cout << " idx | block | thread\n";
    for (int i = 0; i < total; ++i) {
      std::cout << std::setw(4) << h_global[i] << " | " << std::setw(5) << h_block[i] << " | " <<
        std::setw(6) << h_thread[i] << "\n";
    }

    cudaFree(d_global);
    cudaFree(d_block);
    cudaFree(d_thread);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
