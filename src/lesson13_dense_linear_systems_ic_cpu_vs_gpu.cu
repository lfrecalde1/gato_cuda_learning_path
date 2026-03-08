#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

inline std::size_t cm_index(int row, int col, int rows)
{
  return static_cast<std::size_t>(col * rows + row);
}

void fill_stable_dense_system(std::vector<float> & A, int n)
{
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      float value = 0.0006F * std::sin(0.09F * static_cast<float>((r + 1) * (c + 1)));
      if (r == c) {
        value += 0.92F;
      }
      A[cm_index(r, c, n)] = value;
    }
  }
}

void fill_initial_conditions(std::vector<float> & X0, int n, int batch)
{
  for (int ic = 0; ic < batch; ++ic) {
    const float family_shift = 0.18F * static_cast<float>((ic % 8) - 4);
    for (int r = 0; r < n; ++r) {
      const float sin_part = std::sin(0.03F * static_cast<float>((r + 1) * (ic + 1)));
      const float cos_part = std::cos(0.015F * static_cast<float>((r + 2) * (ic + 3)));
      X0[cm_index(r, ic, n)] = 0.5F * sin_part + 0.3F * cos_part + family_shift;
    }
  }
}

void cpu_propagate_batch(
  const std::vector<float> & A,
  const std::vector<float> & X0,
  std::vector<float> & X_out,
  int n,
  int batch,
  int steps)
{
  std::vector<float> X_curr = X0;
  std::vector<float> X_next(static_cast<std::size_t>(n * batch), 0.0F);

  for (int step = 0; step < steps; ++step) {
    for (int ic = 0; ic < batch; ++ic) {
      for (int r = 0; r < n; ++r) {
        float sum = 0.0F;
        for (int c = 0; c < n; ++c) {
          sum += A[cm_index(r, c, n)] * X_curr[cm_index(c, ic, n)];
        }
        X_next[cm_index(r, ic, n)] = sum;
      }
    }
    std::swap(X_curr, X_next);
  }

  X_out = X_curr;
}

float column_l2_norm(const std::vector<float> & X, int n, int col)
{
  double sum_sq = 0.0;
  for (int r = 0; r < n; ++r) {
    const double v = static_cast<double>(X[cm_index(r, col, n)]);
    sum_sq += v * v;
  }
  return static_cast<float>(std::sqrt(sum_sq));
}

}  // namespace

int main()
{
  try {
    constexpr int n = 96;
    constexpr int batch = 256;
    constexpr int steps = 80;

    std::vector<float> h_A(static_cast<std::size_t>(n * n), 0.0F);
    std::vector<float> h_X0(static_cast<std::size_t>(n * batch), 0.0F);
    std::vector<float> h_X_cpu(static_cast<std::size_t>(n * batch), 0.0F);
    std::vector<float> h_X_gpu(static_cast<std::size_t>(n * batch), 0.0F);

    fill_stable_dense_system(h_A, n);
    fill_initial_conditions(h_X0, n, batch);

    gclp::ScopedWallTimer cpu_timer;
    cpu_propagate_batch(h_A, h_X0, h_X_cpu, n, batch, steps);
    const double cpu_ms = cpu_timer.elapsed_ms();

    float * d_A = nullptr;
    float * d_X_curr = nullptr;
    float * d_X_next = nullptr;
    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_X_curr, h_X0.size() * sizeof(float)), "cudaMalloc d_X_curr");
    gclp::throw_on_cuda_error(cudaMalloc(&d_X_next, h_X0.size() * sizeof(float)), "cudaMalloc d_X_next");

    gclp::throw_on_cuda_error(
      cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(
      cudaMemcpy(d_X_curr, h_X0.data(), h_X0.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D X0");

    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");

    cudaStream_t stream = nullptr;
    gclp::throw_on_cuda_error(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    throw_on_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_start), "cudaEventCreate start");
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_stop), "cudaEventCreate stop");

    gclp::throw_on_cuda_error(cudaEventRecord(gpu_start, stream), "cudaEventRecord start");
    for (int step = 0; step < steps; ++step) {
      // Column-major batched propagation: X_next = A * X_curr
      throw_on_cublas(
        cublasSgemm(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, n, batch, n,
          &alpha, d_A, n, d_X_curr, n, &beta, d_X_next, n),
        "cublasSgemm propagate");
      std::swap(d_X_curr, d_X_next);
    }
    gclp::throw_on_cuda_error(cudaEventRecord(gpu_stop, stream), "cudaEventRecord stop");
    gclp::throw_on_cuda_error(cudaEventSynchronize(gpu_stop), "cudaEventSynchronize stop");

    float gpu_ms = 0.0F;
    gclp::throw_on_cuda_error(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop), "cudaEventElapsedTime");

    gclp::throw_on_cuda_error(
      cudaMemcpy(h_X_gpu.data(), d_X_curr, h_X_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H X");

    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < h_X_cpu.size(); ++i) {
      max_abs_error = std::max(
        max_abs_error,
        std::abs(static_cast<double>(h_X_cpu[i]) - static_cast<double>(h_X_gpu[i])));
    }

    std::cout << "Lesson 13: Dense linear system simulation with multiple initial conditions\n";
    std::cout << "Dynamics: x(k+1) = A*x(k), A is dense " << n << "x" << n << "\n";
    std::cout << "Batch size (initial conditions): " << batch << ", steps: " << steps << "\n";
    std::cout << "CPU simulation time (ms): " << cpu_ms << "\n";
    std::cout << "GPU cuBLAS simulation time (ms): " << gpu_ms << "\n";
    std::cout << "Speedup GPU/CPU: " << (cpu_ms / std::max<double>(gpu_ms, 1.0e-9)) << "x\n";
    std::cout << "Max |CPU - GPU| on final state batch: " << max_abs_error << "\n\n";

    std::cout << "Final state norms for sample initial conditions (CPU vs GPU):\n";
    for (int ic = 0; ic < 6; ++ic) {
      const float cpu_norm = column_l2_norm(h_X_cpu, n, ic);
      const float gpu_norm = column_l2_norm(h_X_gpu, n, ic);
      std::cout << "  ic#" << ic << ": " << cpu_norm << " vs " << gpu_norm << "\n";
    }

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_X_curr);
    cudaFree(d_X_next);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
