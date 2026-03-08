#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void matrix_vector_mul_kernel(const float * A, const float * x, float * y, int rows, int cols)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float sum = 0.0F;
    for (int j = 0; j < cols; ++j) {
      sum += A[row * cols + j] * x[j];
    }
    y[row] = sum;
  }
}

__global__ void matrix_add_kernel(const float * A, const float * B, float * C, int total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    C[idx] = A[idx] + B[idx];
  }
}

int main()
{
  try {
    constexpr int rows = 256;
    constexpr int cols = 128;
    constexpr int total = rows * cols;
    constexpr int threads = 256;

    std::vector<float> h_A(total), h_B(total), h_C(total), h_x(cols), h_y(rows), h_y_ref(rows, 0.0F);

    for (int i = 0; i < total; ++i) {
      h_A[i] = 0.01F * static_cast<float>(i % 19);
      h_B[i] = 0.02F * static_cast<float>((i + 7) % 23);
    }
    for (int i = 0; i < cols; ++i) {
      h_x[i] = std::sin(0.01F * static_cast<float>(i));
    }

    for (int r = 0; r < rows; ++r) {
      float sum = 0.0F;
      for (int c = 0; c < cols; ++c) {
        sum += h_A[static_cast<std::size_t>(r * cols + c)] * h_x[static_cast<std::size_t>(c)];
      }
      h_y_ref[static_cast<std::size_t>(r)] = sum;
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_C = nullptr;
    float * d_x = nullptr;
    float * d_y = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, total * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, total * sizeof(float)), "cudaMalloc d_B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_C, total * sizeof(float)), "cudaMalloc d_C");
    gclp::throw_on_cuda_error(cudaMalloc(&d_x, cols * sizeof(float)), "cudaMalloc d_x");
    gclp::throw_on_cuda_error(cudaMalloc(&d_y, rows * sizeof(float)), "cudaMalloc d_y");

    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
    gclp::throw_on_cuda_error(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    const int blocks_total = (total + threads - 1) / threads;
    const int blocks_rows = (rows + threads - 1) / threads;

    matrix_add_kernel<<<blocks_total, threads>>>(d_A, d_B, d_C, total);
    matrix_vector_mul_kernel<<<blocks_rows, threads>>>(d_A, d_x, d_y, rows, cols);
    gclp::throw_on_cuda_error(cudaGetLastError(), "matrix kernels launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    gclp::throw_on_cuda_error(cudaMemcpy(h_C.data(), d_C, total * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");
    gclp::throw_on_cuda_error(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost), "D2H y");

    double max_err = 0.0;
    for (int i = 0; i < rows; ++i) {
      const double err = std::abs(static_cast<double>(h_y[i]) - static_cast<double>(h_y_ref[i]));
      if (err > max_err) {
        max_err = err;
      }
    }

    std::cout << "Lesson 09: Matrix-vector and matrix-matrix elementwise ops\n";
    std::cout << "C[0] (A+B)=" << h_C[0] << " y[0]=" << h_y[0] << " max_err_vs_cpu=" << max_err << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
