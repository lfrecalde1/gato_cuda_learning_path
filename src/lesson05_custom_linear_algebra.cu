#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

void throw_on_cublas(cublasStatus_t status, const char * context)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS call failed");
  }
}

__global__ void custom_gramian_kernel(const float * A, float * G, int rows, int cols)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < cols && col < cols) {
    float sum = 0.0F;
    for (int k = 0; k < rows; ++k) {
      sum += A[k * cols + row] * A[k * cols + col];
    }
    G[row * cols + col] = sum;
  }
}

int main()
{
  try {
    constexpr int rows = 512;
    constexpr int cols = 128;

    std::vector<float> h_A(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
      h_A[i] = 0.03F * std::sin(0.001F * static_cast<float>(i));
    }

    float * d_A = nullptr;
    float * d_G_custom = nullptr;
    float * d_G_cublas = nullptr;
    std::vector<float> h_G_custom(cols * cols), h_G_cublas(cols * cols);

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_G_custom, h_G_custom.size() * sizeof(float)), "cudaMalloc G_custom");
    gclp::throw_on_cuda_error(cudaMalloc(&d_G_cublas, h_G_cublas.size() * sizeof(float)), "cudaMalloc G_cublas");
    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");

    constexpr dim3 block(16, 16);
    const dim3 grid(
      static_cast<unsigned>((cols + block.x - 1) / block.x),
      static_cast<unsigned>((cols + block.y - 1) / block.y));

    gclp::ScopedWallTimer custom_timer;
    custom_gramian_kernel<<<grid, block>>>(d_A, d_G_custom, rows, cols);
    gclp::throw_on_cuda_error(cudaGetLastError(), "custom_gramian_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize custom");
    const double custom_ms = custom_timer.elapsed_ms();

    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // Compute G = A^T A. A is row-major; use swapped operands in cuBLAS.
    gclp::ScopedWallTimer cublas_timer;
    throw_on_cublas(
      cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        cols,
        cols,
        rows,
        &alpha,
        d_A,
        cols,
        d_A,
        cols,
        &beta,
        d_G_cublas,
        cols),
      "cublasSgemm A^T A");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize cublas");
    const double cublas_ms = cublas_timer.elapsed_ms();

    gclp::throw_on_cuda_error(cudaMemcpy(h_G_custom.data(), d_G_custom, h_G_custom.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H G_custom");
    gclp::throw_on_cuda_error(cudaMemcpy(h_G_cublas.data(), d_G_cublas, h_G_cublas.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H G_cublas");

    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < h_G_custom.size(); ++i) {
      const double err = std::abs(static_cast<double>(h_G_custom[i]) - static_cast<double>(h_G_cublas[i]));
      if (err > max_abs_err) {
        max_abs_err = err;
      }
    }

    std::cout << "Lesson 05: Custom linear algebra vs cuBLAS\n";
    std::cout << "custom_ms=" << custom_ms << " cublas_ms=" << cublas_ms << " max_abs_err=" << max_abs_err << "\n";
    std::cout << "Why custom kernels still matter: structure-aware operators (block-tridiagonal, Schur, PCG) can beat dense GEMM for optimizer-specific math.\n";

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_G_custom);
    cudaFree(d_G_cublas);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
