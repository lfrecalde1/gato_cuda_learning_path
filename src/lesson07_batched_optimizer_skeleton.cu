#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void batched_rollout_cost_kernel(
  const float * x0_batch,
  const float * u_batch,
  float * cost_batch,
  int state_dim,
  int control_dim,
  int horizon)
{
  const int b = blockIdx.x;
  if (threadIdx.x == 0) {
    const float * x0 = x0_batch + b * state_dim;
    const float * u = u_batch + b * horizon * control_dim;

    float x_scalar = x0[0];
    float cost = 0.0F;

    for (int k = 0; k < horizon; ++k) {
      const float uk = u[k * control_dim + 0];
      x_scalar = 0.98F * x_scalar + 0.05F * uk;
      cost += x_scalar * x_scalar + 0.01F * uk * uk;
    }

    cost_batch[b] = cost;
  }
}

int main()
{
  try {
    constexpr int batch_size = 64;
    constexpr int state_dim = 12;
    constexpr int control_dim = 4;
    constexpr int horizon = 20;

    std::vector<float> h_x0(batch_size * state_dim);
    std::vector<float> h_u(batch_size * horizon * control_dim);
    std::vector<float> h_cost(batch_size);

    for (int b = 0; b < batch_size; ++b) {
      for (int i = 0; i < state_dim; ++i) {
        h_x0[static_cast<std::size_t>(b * state_dim + i)] = 0.1F * std::sin(0.1F * static_cast<float>(b + i));
      }
      for (int k = 0; k < horizon; ++k) {
        for (int j = 0; j < control_dim; ++j) {
          const std::size_t idx = static_cast<std::size_t>(b * horizon * control_dim + k * control_dim + j);
          h_u[idx] = 0.2F * std::cos(0.05F * static_cast<float>(b + k + j));
        }
      }
    }

    float * d_x0 = nullptr;
    float * d_u = nullptr;
    float * d_cost = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_x0, h_x0.size() * sizeof(float)), "cudaMalloc d_x0");
    gclp::throw_on_cuda_error(cudaMalloc(&d_u, h_u.size() * sizeof(float)), "cudaMalloc d_u");
    gclp::throw_on_cuda_error(cudaMalloc(&d_cost, h_cost.size() * sizeof(float)), "cudaMalloc d_cost");

    gclp::throw_on_cuda_error(cudaMemcpy(d_x0, h_x0.data(), h_x0.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D x0");
    gclp::throw_on_cuda_error(cudaMemcpy(d_u, h_u.data(), h_u.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D u");

    gclp::ScopedWallTimer timer;
    batched_rollout_cost_kernel<<<batch_size, 32>>>(d_x0, d_u, d_cost, state_dim, control_dim, horizon);
    gclp::throw_on_cuda_error(cudaGetLastError(), "batched_rollout_cost_kernel launch");
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const double elapsed_ms = timer.elapsed_ms();

    gclp::throw_on_cuda_error(cudaMemcpy(h_cost.data(), d_cost, h_cost.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H cost");

    double mean_cost = 0.0;
    for (float c : h_cost) {
      mean_cost += c;
    }
    mean_cost /= static_cast<double>(batch_size);

    std::cout << "Lesson 07: Batched optimizer skeleton (GATO-style batching concept)\n";
    std::cout << "batch_size=" << batch_size << " horizon=" << horizon << " elapsed_ms=" << elapsed_ms << " mean_cost=" << mean_cost << "\n";

    cudaFree(d_x0);
    cudaFree(d_u);
    cudaFree(d_cost);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
