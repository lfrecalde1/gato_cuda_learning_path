#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

__global__ void projected_gradient_step(
  float * u,
  const float * grad,
  const float * u_min,
  const float * u_max,
  float step,
  int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float next = u[idx] - step * grad[idx];
    if (next < u_min[idx]) {
      next = u_min[idx];
    }
    if (next > u_max[idx]) {
      next = u_max[idx];
    }
    u[idx] = next;
  }
}

__global__ void merit_kernel(const float * x, const float * u, float * out, int nx, int nu)
{
  __shared__ float cache[256];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float value = 0.0F;
  if (idx < nx) {
    value += 0.5F * x[idx] * x[idx];
  }
  if (idx < nu) {
    value += 0.01F * u[idx] * u[idx];
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
    atomicAdd(out, cache[0]);
  }
}

int main()
{
  try {
    constexpr int nx = 4096;
    constexpr int nu = 1024;
    constexpr int threads = 256;

    std::vector<float> h_x(nx), h_u(nu), h_grad(nu), h_u_min(nu, -0.7F), h_u_max(nu, 0.7F);
    for (int i = 0; i < nx; ++i) {
      h_x[i] = std::sin(0.003F * static_cast<float>(i));
    }
    for (int i = 0; i < nu; ++i) {
      h_u[i] = 0.4F * std::cos(0.01F * static_cast<float>(i));
      h_grad[i] = 0.2F * std::sin(0.02F * static_cast<float>(i));
    }

    float * d_x = nullptr;
    float * d_u = nullptr;
    float * d_grad = nullptr;
    float * d_u_min = nullptr;
    float * d_u_max = nullptr;
    float * d_merit = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_x, nx * sizeof(float)), "cudaMalloc d_x");
    gclp::throw_on_cuda_error(cudaMalloc(&d_u, nu * sizeof(float)), "cudaMalloc d_u");
    gclp::throw_on_cuda_error(cudaMalloc(&d_grad, nu * sizeof(float)), "cudaMalloc d_grad");
    gclp::throw_on_cuda_error(cudaMalloc(&d_u_min, nu * sizeof(float)), "cudaMalloc d_u_min");
    gclp::throw_on_cuda_error(cudaMalloc(&d_u_max, nu * sizeof(float)), "cudaMalloc d_u_max");
    gclp::throw_on_cuda_error(cudaMalloc(&d_merit, sizeof(float)), "cudaMalloc d_merit");

    gclp::throw_on_cuda_error(cudaMemcpy(d_x, h_x.data(), nx * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    gclp::throw_on_cuda_error(cudaMemcpy(d_u, h_u.data(), nu * sizeof(float), cudaMemcpyHostToDevice), "H2D u");
    gclp::throw_on_cuda_error(cudaMemcpy(d_grad, h_grad.data(), nu * sizeof(float), cudaMemcpyHostToDevice), "H2D grad");
    gclp::throw_on_cuda_error(cudaMemcpy(d_u_min, h_u_min.data(), nu * sizeof(float), cudaMemcpyHostToDevice), "H2D u_min");
    gclp::throw_on_cuda_error(cudaMemcpy(d_u_max, h_u_max.data(), nu * sizeof(float), cudaMemcpyHostToDevice), "H2D u_max");

    const int blocks_u = (nu + threads - 1) / threads;
    projected_gradient_step<<<blocks_u, threads>>>(d_u, d_grad, d_u_min, d_u_max, 0.05F, nu);

    const int max_len = (nx > nu) ? nx : nu;
    const int blocks_m = (max_len + threads - 1) / threads;
    gclp::throw_on_cuda_error(cudaMemset(d_merit, 0, sizeof(float)), "cudaMemset merit");
    merit_kernel<<<blocks_m, threads>>>(d_x, d_u, d_merit, nx, nu);

    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    float merit = 0.0F;
    gclp::throw_on_cuda_error(cudaMemcpy(&merit, d_merit, sizeof(float), cudaMemcpyDeviceToHost), "D2H merit");
    gclp::throw_on_cuda_error(cudaMemcpy(h_u.data(), d_u, nu * sizeof(float), cudaMemcpyDeviceToHost), "D2H u");

    std::cout << "Lesson 06: Optimizer step primitives\n";
    std::cout << "merit=" << merit << " u[0..3]={" << h_u[0] << ", " << h_u[1] << ", " << h_u[2] << ", " << h_u[3] << "}\n";

    cudaFree(d_x);
    cudaFree(d_u);
    cudaFree(d_grad);
    cudaFree(d_u_min);
    cudaFree(d_u_max);
    cudaFree(d_merit);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
