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

int main()
{
  try {
    constexpr int m = 256;
    constexpr int n = 256;
    constexpr int k = 256;

    std::vector<float> h_A(m * k), h_B(k * n), h_C(m * n, 0.0F);
    for (int i = 0; i < m * k; ++i) {
      h_A[i] = 0.01F * static_cast<float>(i % 31);
    }
    for (int i = 0; i < k * n; ++i) {
      h_B[i] = 0.02F * static_cast<float>(i % 17);
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_C = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, h_B.size() * sizeof(float)), "cudaMalloc B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_C, h_C.size() * sizeof(float)), "cudaMalloc C");

    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // row-major C = A * B using cuBLAS column-major by swapping operands.
    gclp::ScopedWallTimer timer;
    throw_on_cublas(
      cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        d_B,
        n,
        d_A,
        k,
        &beta,
        d_C,
        n),
      "cublasSgemm");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const double elapsed_ms = timer.elapsed_ms();

    gclp::throw_on_cuda_error(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    double checksum = 0.0;
    for (int i = 0; i < 64; ++i) {
      checksum += h_C[static_cast<std::size_t>(i)];
    }

    std::cout << "Lesson 04: cuBLAS GEMM\n";
    std::cout << "shape=" << m << "x" << k << " * " << k << "x" << n << " elapsed_ms=" << elapsed_ms << " checksum64=" << checksum << "\n";

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
