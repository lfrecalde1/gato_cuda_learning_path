#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void elementwise_ops_kernel(
  const float * A,
  const float * B,
  float * C_add,
  float * C_sub,
  float * C_hadamard,
  int rows,
  int cols)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx < total) {
    C_add[idx] = A[idx] + B[idx];
    C_sub[idx] = A[idx] - B[idx];
    C_hadamard[idx] = A[idx] * B[idx];
  }
}

void print_matrix(const std::vector<float> & M, int rows, int cols, const char * name)
{
  std::cout << name << "\n";
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << std::setw(8) << M[static_cast<std::size_t>(r * cols + c)] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main()
{
  try {
    constexpr int rows = 4;
    constexpr int cols = 4;
    constexpr int total = rows * cols;

    std::vector<float> h_A(total), h_B(total), h_add(total), h_sub(total), h_hadamard(total);

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        const int idx = r * cols + c;
        h_A[static_cast<std::size_t>(idx)] = static_cast<float>(r + c);
        h_B[static_cast<std::size_t>(idx)] = static_cast<float>(2 * r - c);
      }
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_add = nullptr;
    float * d_sub = nullptr;
    float * d_hadamard = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, total * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, total * sizeof(float)), "cudaMalloc d_B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_add, total * sizeof(float)), "cudaMalloc d_add");
    gclp::throw_on_cuda_error(cudaMalloc(&d_sub, total * sizeof(float)), "cudaMalloc d_sub");
    gclp::throw_on_cuda_error(cudaMalloc(&d_hadamard, total * sizeof(float)), "cudaMalloc d_hadamard");

    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    elementwise_ops_kernel<<<1, 32>>>(d_A, d_B, d_add, d_sub, d_hadamard, rows, cols);
    gclp::throw_on_cuda_error(cudaGetLastError(), "elementwise_ops_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    gclp::throw_on_cuda_error(cudaMemcpy(h_add.data(), d_add, total * sizeof(float), cudaMemcpyDeviceToHost), "D2H add");
    gclp::throw_on_cuda_error(cudaMemcpy(h_sub.data(), d_sub, total * sizeof(float), cudaMemcpyDeviceToHost), "D2H sub");
    gclp::throw_on_cuda_error(
      cudaMemcpy(h_hadamard.data(), d_hadamard, total * sizeof(float), cudaMemcpyDeviceToHost),
      "D2H hadamard");

    std::cout << "Lesson 08: Matrix basics (variables and element operations)\n\n";
    print_matrix(h_A, rows, cols, "Matrix A:");
    print_matrix(h_B, rows, cols, "Matrix B:");
    print_matrix(h_add, rows, cols, "A + B:");
    print_matrix(h_sub, rows, cols, "A - B:");
    print_matrix(h_hadamard, rows, cols, "A .* B (elementwise product):");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_add);
    cudaFree(d_sub);
    cudaFree(d_hadamard);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
