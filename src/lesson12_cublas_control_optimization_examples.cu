#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

__global__ void projected_step_kernel(float * u, const float * grad, float step, float u_min, float u_max, int n)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float next = u[idx] - step * grad[idx];
    if (next < u_min) {
      next = u_min;
    }
    if (next > u_max) {
      next = u_max;
    }
    u[idx] = next;
  }
}

}  // namespace

int main()
{
  try {
    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");

    // Example A: linear control model x_{k+1} = A*x_k + B*u_k
    constexpr int nx = 12;
    constexpr int nu = 4;
    std::vector<float> h_A(nx * nx, 0.0F);
    std::vector<float> h_B(nx * nu, 0.0F);
    std::vector<float> h_x(nx, 0.0F);
    std::vector<float> h_u(nu, 0.0F);
    std::vector<float> h_x_next(nx, 0.0F);

    for (int r = 0; r < nx; ++r) {
      for (int c = 0; c < nx; ++c) {
        h_A[cm_index(r, c, nx)] = (r == c) ? 0.98F : 0.002F * static_cast<float>((r + c) % 5);
      }
    }
    for (int r = 0; r < nx; ++r) {
      for (int c = 0; c < nu; ++c) {
        h_B[cm_index(r, c, nx)] = 0.01F * static_cast<float>((1 + r + c) % 7);
      }
    }
    for (int i = 0; i < nx; ++i) {
      h_x[static_cast<std::size_t>(i)] = 0.2F * std::sin(0.1F * static_cast<float>(i));
    }
    for (int i = 0; i < nu; ++i) {
      h_u[static_cast<std::size_t>(i)] = 0.15F * std::cos(0.3F * static_cast<float>(i));
    }

    float * d_A = nullptr;
    float * d_B = nullptr;
    float * d_x = nullptr;
    float * d_u = nullptr;
    float * d_x_next = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_A, h_A.size() * sizeof(float)), "cudaMalloc d_A");
    gclp::throw_on_cuda_error(cudaMalloc(&d_B, h_B.size() * sizeof(float)), "cudaMalloc d_B");
    gclp::throw_on_cuda_error(cudaMalloc(&d_x, h_x.size() * sizeof(float)), "cudaMalloc d_x");
    gclp::throw_on_cuda_error(cudaMalloc(&d_u, h_u.size() * sizeof(float)), "cudaMalloc d_u");
    gclp::throw_on_cuda_error(cudaMalloc(&d_x_next, h_x_next.size() * sizeof(float)), "cudaMalloc d_x_next");

    gclp::throw_on_cuda_error(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    gclp::throw_on_cuda_error(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");
    gclp::throw_on_cuda_error(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    gclp::throw_on_cuda_error(cudaMemcpy(d_u, h_u.data(), h_u.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D u");

    constexpr float one = 1.0F;
    constexpr float zero = 0.0F;
    throw_on_cublas(cublasSgemv(handle, CUBLAS_OP_N, nx, nx, &one, d_A, nx, d_x, 1, &zero, d_x_next, 1), "gemv A*x");
    throw_on_cublas(cublasSgemv(handle, CUBLAS_OP_N, nx, nu, &one, d_B, nx, d_u, 1, &one, d_x_next, 1), "gemv B*u");
    gclp::throw_on_cuda_error(cudaMemcpy(h_x_next.data(), d_x_next, h_x_next.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H x_next");

    std::cout << "Lesson 12A: cuBLAS for control linear dynamics\n";
    std::cout << "x_next[0..3] = {" << h_x_next[0] << ", " << h_x_next[1] << ", " << h_x_next[2] << ", " << h_x_next[3] << "}\n\n";

    // Example B: optimization gradient g = H^T(Hu - b) + lambda*u
    constexpr int m = 96;
    constexpr int n = 48;
    constexpr int iters = 25;
    constexpr float lambda = 1.0e-2F;
    constexpr float step = 8.0e-2F;

    std::vector<float> h_H(m * n, 0.0F);
    std::vector<float> h_b(m, 0.0F);
    std::vector<float> h_uopt(n, 0.0F);

    for (int r = 0; r < m; ++r) {
      h_b[static_cast<std::size_t>(r)] = std::sin(0.05F * static_cast<float>(r));
      for (int c = 0; c < n; ++c) {
        h_H[cm_index(r, c, m)] = 0.02F * std::cos(0.01F * static_cast<float>(r * (c + 1)));
      }
    }
    for (int i = 0; i < n; ++i) {
      h_uopt[static_cast<std::size_t>(i)] = 0.1F * std::sin(0.2F * static_cast<float>(i));
    }

    float * d_H = nullptr;
    float * d_b = nullptr;
    float * d_uopt = nullptr;
    float * d_y = nullptr;    // y = H*u - b
    float * d_grad = nullptr; // grad = H^T*y + lambda*u

    gclp::throw_on_cuda_error(cudaMalloc(&d_H, h_H.size() * sizeof(float)), "cudaMalloc d_H");
    gclp::throw_on_cuda_error(cudaMalloc(&d_b, h_b.size() * sizeof(float)), "cudaMalloc d_b");
    gclp::throw_on_cuda_error(cudaMalloc(&d_uopt, h_uopt.size() * sizeof(float)), "cudaMalloc d_uopt");
    gclp::throw_on_cuda_error(cudaMalloc(&d_y, m * sizeof(float)), "cudaMalloc d_y");
    gclp::throw_on_cuda_error(cudaMalloc(&d_grad, n * sizeof(float)), "cudaMalloc d_grad");

    gclp::throw_on_cuda_error(cudaMemcpy(d_H, h_H.data(), h_H.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D H");
    gclp::throw_on_cuda_error(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D b");
    gclp::throw_on_cuda_error(cudaMemcpy(d_uopt, h_uopt.data(), h_uopt.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D uopt");

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    auto evaluate_objective = [&]() {
      throw_on_cublas(cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, d_H, m, d_uopt, 1, &zero, d_y, 1), "gemv H*u");
      const float minus_one = -1.0F;
      throw_on_cublas(cublasSaxpy(handle, m, &minus_one, d_b, 1, d_y, 1), "axpy y-b");

      float y_norm = 0.0F;
      float u_norm = 0.0F;
      throw_on_cublas(cublasSnrm2(handle, m, d_y, 1, &y_norm), "nrm2 y");
      throw_on_cublas(cublasSnrm2(handle, n, d_uopt, 1, &u_norm), "nrm2 u");

      return 0.5F * y_norm * y_norm + 0.5F * lambda * u_norm * u_norm;
    };

    float objective0 = evaluate_objective();
    for (int iter = 0; iter < iters; ++iter) {
      // y = H*u - b
      throw_on_cublas(cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, d_H, m, d_uopt, 1, &zero, d_y, 1), "gemv H*u iter");
      const float minus_one = -1.0F;
      throw_on_cublas(cublasSaxpy(handle, m, &minus_one, d_b, 1, d_y, 1), "axpy -b iter");

      // grad = H^T * y
      throw_on_cublas(cublasSgemv(handle, CUBLAS_OP_T, m, n, &one, d_H, m, d_y, 1, &zero, d_grad, 1), "gemv H^T*y");

      // grad += lambda * u
      throw_on_cublas(cublasSaxpy(handle, n, &lambda, d_uopt, 1, d_grad, 1), "axpy lambda*u");

      // projected gradient step
      projected_step_kernel<<<blocks, threads>>>(d_uopt, d_grad, step, -0.8F, 0.8F, n);
      gclp::throw_on_cuda_error(cudaGetLastError(), "projected_step_kernel launch");
    }
    gclp::throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize opt");

    float objectiveN = evaluate_objective();
    gclp::throw_on_cuda_error(cudaMemcpy(h_uopt.data(), d_uopt, h_uopt.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H uopt");

    std::cout << "Lesson 12B: cuBLAS in optimization loop (least-squares + L2 + box projection)\n";
    std::cout << "J(initial)=" << objective0 << " J(final)=" << objectiveN << " improvement=" << (objective0 - objectiveN) << "\n";
    std::cout << "u*[0..3] = {" << h_uopt[0] << ", " << h_uopt[1] << ", " << h_uopt[2] << ", " << h_uopt[3] << "}\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_x);
    cudaFree(d_u);
    cudaFree(d_x_next);
    cudaFree(d_H);
    cudaFree(d_b);
    cudaFree(d_uopt);
    cudaFree(d_y);
    cudaFree(d_grad);
    cublasDestroy(handle);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
