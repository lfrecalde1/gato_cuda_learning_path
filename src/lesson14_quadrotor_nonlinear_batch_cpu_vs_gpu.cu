#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace gclp = gato_cuda_learning_path;

namespace
{

constexpr int kStateDim = 12;  // p(3), v(3), euler(3), omega(3)
constexpr int kVec3 = 3;

struct QuadParams
{
  float m{1.4F};
  float g{9.81F};
  float Jx{0.02F};
  float Jy{0.025F};
  float Jz{0.03F};
  float inv_Jx{1.0F / Jx};
  float inv_Jy{1.0F / Jy};
  float inv_Jz{1.0F / Jz};
};

void throw_on_cublas(cublasStatus_t status, const char * context)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS call failed");
  }
}

__host__ __device__ inline std::size_t idx_state(int batch_i, int state_i, int batch)
{
  return static_cast<std::size_t>(state_i * batch + batch_i);
}

__host__ __device__ inline std::size_t idx_col_major(int row, int col, int rows)
{
  return static_cast<std::size_t>(col * rows + row);
}

void seed_initial_states_and_inputs(
  std::vector<float> & X0,
  std::vector<float> & U,
  int batch,
  const QuadParams & p)
{
  for (int b = 0; b < batch; ++b) {
    // position
    X0[idx_state(b, 0, batch)] = 0.8F * std::sin(0.03F * static_cast<float>(b));
    X0[idx_state(b, 1, batch)] = 0.7F * std::cos(0.02F * static_cast<float>(b));
    X0[idx_state(b, 2, batch)] = 1.0F + 0.2F * std::sin(0.015F * static_cast<float>(b));
    // velocity
    X0[idx_state(b, 3, batch)] = 0.1F * std::sin(0.07F * static_cast<float>(b + 1));
    X0[idx_state(b, 4, batch)] = 0.1F * std::cos(0.05F * static_cast<float>(b + 2));
    X0[idx_state(b, 5, batch)] = -0.05F * std::sin(0.04F * static_cast<float>(b + 3));
    // euler angles
    X0[idx_state(b, 6, batch)] = 0.07F * std::sin(0.017F * static_cast<float>(b));
    X0[idx_state(b, 7, batch)] = 0.08F * std::cos(0.019F * static_cast<float>(b));
    X0[idx_state(b, 8, batch)] = 0.12F * std::sin(0.011F * static_cast<float>(b));
    // body rates
    X0[idx_state(b, 9, batch)] = 0.02F * std::sin(0.031F * static_cast<float>(b + 1));
    X0[idx_state(b, 10, batch)] = -0.025F * std::cos(0.029F * static_cast<float>(b + 2));
    X0[idx_state(b, 11, batch)] = 0.02F * std::sin(0.021F * static_cast<float>(b + 3));

    const float hover = p.m * p.g;
    U[idx_col_major(0, b, 4)] = hover * (1.0F + 0.08F * std::sin(0.013F * static_cast<float>(b)));
    U[idx_col_major(1, b, 4)] = 0.02F * std::sin(0.023F * static_cast<float>(b + 1));
    U[idx_col_major(2, b, 4)] = 0.02F * std::cos(0.021F * static_cast<float>(b + 2));
    U[idx_col_major(3, b, 4)] = 0.015F * std::sin(0.018F * static_cast<float>(b + 3));
  }
}

void cpu_step(
  std::vector<float> & X,
  const std::vector<float> & U,
  int batch,
  float dt,
  const QuadParams & p)
{
  for (int b = 0; b < batch; ++b) {
    const float thrust = U[idx_col_major(0, b, 4)];
    const float tau_x = U[idx_col_major(1, b, 4)];
    const float tau_y = U[idx_col_major(2, b, 4)];
    const float tau_z = U[idx_col_major(3, b, 4)];

    float px = X[idx_state(b, 0, batch)];
    float py = X[idx_state(b, 1, batch)];
    float pz = X[idx_state(b, 2, batch)];
    float vx = X[idx_state(b, 3, batch)];
    float vy = X[idx_state(b, 4, batch)];
    float vz = X[idx_state(b, 5, batch)];
    float roll = X[idx_state(b, 6, batch)];
    float pitch = X[idx_state(b, 7, batch)];
    float yaw = X[idx_state(b, 8, batch)];
    float wx = X[idx_state(b, 9, batch)];
    float wy = X[idx_state(b, 10, batch)];
    float wz = X[idx_state(b, 11, batch)];

    const float cr = std::cos(roll);
    const float sr = std::sin(roll);
    const float cp = std::cos(pitch);
    const float sp = std::sin(pitch);
    const float cy = std::cos(yaw);
    const float sy = std::sin(yaw);
    const float tp = std::tan(pitch);
    const float secp = 1.0F / std::max(0.2F, cp);

    // Third column of Rz(yaw)*Ry(pitch)*Rx(roll), used for thrust direction.
    const float r13 = cy * sp * cr + sy * sr;
    const float r23 = sy * sp * cr - cy * sr;
    const float r33 = cp * cr;

    const float ax = (thrust / p.m) * r13;
    const float ay = (thrust / p.m) * r23;
    const float az = (thrust / p.m) * r33 - p.g;

    // Euler angle kinematics: euler_dot = E(roll,pitch) * omega
    const float roll_dot = wx + sr * tp * wy + cr * tp * wz;
    const float pitch_dot = cr * wy - sr * wz;
    const float yaw_dot = sr * secp * wy + cr * secp * wz;

    // Rigid body rates
    const float wx_dot = p.inv_Jx * (tau_x - (p.Jz - p.Jy) * wy * wz);
    const float wy_dot = p.inv_Jy * (tau_y - (p.Jx - p.Jz) * wx * wz);
    const float wz_dot = p.inv_Jz * (tau_z - (p.Jy - p.Jx) * wx * wy);

    px += dt * vx;
    py += dt * vy;
    pz += dt * vz;
    vx += dt * ax;
    vy += dt * ay;
    vz += dt * az;
    roll += dt * roll_dot;
    pitch += dt * pitch_dot;
    yaw += dt * yaw_dot;
    wx += dt * wx_dot;
    wy += dt * wy_dot;
    wz += dt * wz_dot;

    X[idx_state(b, 0, batch)] = px;
    X[idx_state(b, 1, batch)] = py;
    X[idx_state(b, 2, batch)] = pz;
    X[idx_state(b, 3, batch)] = vx;
    X[idx_state(b, 4, batch)] = vy;
    X[idx_state(b, 5, batch)] = vz;
    X[idx_state(b, 6, batch)] = roll;
    X[idx_state(b, 7, batch)] = pitch;
    X[idx_state(b, 8, batch)] = yaw;
    X[idx_state(b, 9, batch)] = wx;
    X[idx_state(b, 10, batch)] = wy;
    X[idx_state(b, 11, batch)] = wz;
  }
}

__global__ void build_batch_matrices_kernel(
  const float * X,
  const float * U,
  float * R_batch,
  float * E_batch,
  float * thrust_body_batch,
  float * omega_batch,
  int batch)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) {
    return;
  }

  const float roll = X[idx_state(b, 6, batch)];
  const float pitch = X[idx_state(b, 7, batch)];
  const float yaw = X[idx_state(b, 8, batch)];
  const float wx = X[idx_state(b, 9, batch)];
  const float wy = X[idx_state(b, 10, batch)];
  const float wz = X[idx_state(b, 11, batch)];
  const float thrust = U[idx_col_major(0, b, 4)];

  const float cr = cosf(roll);
  const float sr = sinf(roll);
  const float cp = cosf(pitch);
  const float sp = sinf(pitch);
  const float cy = cosf(yaw);
  const float sy = sinf(yaw);
  const float tp = tanf(pitch);
  const float secp = 1.0F / fmaxf(0.2F, cp);

  // R = Rz(yaw)*Ry(pitch)*Rx(roll), column-major 3x3 per batch item
  float * R = R_batch + static_cast<std::size_t>(b) * 9;
  R[0] = cy * cp;
  R[1] = sy * cp;
  R[2] = -sp;
  R[3] = cy * sp * sr - sy * cr;
  R[4] = sy * sp * sr + cy * cr;
  R[5] = cp * sr;
  R[6] = cy * sp * cr + sy * sr;
  R[7] = sy * sp * cr - cy * sr;
  R[8] = cp * cr;

  // E matrix for euler_dot = E * omega
  float * E = E_batch + static_cast<std::size_t>(b) * 9;
  E[0] = 1.0F;
  E[1] = 0.0F;
  E[2] = 0.0F;
  E[3] = sr * tp;
  E[4] = cr;
  E[5] = sr * secp;
  E[6] = cr * tp;
  E[7] = -sr;
  E[8] = cr * secp;

  float * thrust_body = thrust_body_batch + static_cast<std::size_t>(b) * 3;
  thrust_body[0] = 0.0F;
  thrust_body[1] = 0.0F;
  thrust_body[2] = thrust;

  float * omega = omega_batch + static_cast<std::size_t>(b) * 3;
  omega[0] = wx;
  omega[1] = wy;
  omega[2] = wz;
}

__global__ void integrate_batch_kernel(
  float * X,
  const float * U,
  const float * accel_world_batch,
  const float * euler_dot_batch,
  int batch,
  float dt,
  QuadParams p)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) {
    return;
  }

  float px = X[idx_state(b, 0, batch)];
  float py = X[idx_state(b, 1, batch)];
  float pz = X[idx_state(b, 2, batch)];
  float vx = X[idx_state(b, 3, batch)];
  float vy = X[idx_state(b, 4, batch)];
  float vz = X[idx_state(b, 5, batch)];
  float roll = X[idx_state(b, 6, batch)];
  float pitch = X[idx_state(b, 7, batch)];
  float yaw = X[idx_state(b, 8, batch)];
  float wx = X[idx_state(b, 9, batch)];
  float wy = X[idx_state(b, 10, batch)];
  float wz = X[idx_state(b, 11, batch)];

  const float tau_x = U[idx_col_major(1, b, 4)];
  const float tau_y = U[idx_col_major(2, b, 4)];
  const float tau_z = U[idx_col_major(3, b, 4)];

  const float ax = accel_world_batch[static_cast<std::size_t>(b) * 3] / p.m;
  const float ay = accel_world_batch[static_cast<std::size_t>(b) * 3 + 1] / p.m;
  const float az = accel_world_batch[static_cast<std::size_t>(b) * 3 + 2] / p.m - p.g;

  const float roll_dot = euler_dot_batch[static_cast<std::size_t>(b) * 3];
  const float pitch_dot = euler_dot_batch[static_cast<std::size_t>(b) * 3 + 1];
  const float yaw_dot = euler_dot_batch[static_cast<std::size_t>(b) * 3 + 2];

  const float wx_dot = p.inv_Jx * (tau_x - (p.Jz - p.Jy) * wy * wz);
  const float wy_dot = p.inv_Jy * (tau_y - (p.Jx - p.Jz) * wx * wz);
  const float wz_dot = p.inv_Jz * (tau_z - (p.Jy - p.Jx) * wx * wy);

  px += dt * vx;
  py += dt * vy;
  pz += dt * vz;
  vx += dt * ax;
  vy += dt * ay;
  vz += dt * az;
  roll += dt * roll_dot;
  pitch += dt * pitch_dot;
  yaw += dt * yaw_dot;
  wx += dt * wx_dot;
  wy += dt * wy_dot;
  wz += dt * wz_dot;

  X[idx_state(b, 0, batch)] = px;
  X[idx_state(b, 1, batch)] = py;
  X[idx_state(b, 2, batch)] = pz;
  X[idx_state(b, 3, batch)] = vx;
  X[idx_state(b, 4, batch)] = vy;
  X[idx_state(b, 5, batch)] = vz;
  X[idx_state(b, 6, batch)] = roll;
  X[idx_state(b, 7, batch)] = pitch;
  X[idx_state(b, 8, batch)] = yaw;
  X[idx_state(b, 9, batch)] = wx;
  X[idx_state(b, 10, batch)] = wy;
  X[idx_state(b, 11, batch)] = wz;
}

}  // namespace

int main()
{
  try {
    constexpr float dt = 0.01F;
    constexpr float sim_time = 10.0F;
    constexpr int steps = static_cast<int>(sim_time / dt);  // 1000
    constexpr int batch = 512;

    const QuadParams params{};

    std::vector<float> h_X0(static_cast<std::size_t>(kStateDim * batch), 0.0F);
    std::vector<float> h_U(static_cast<std::size_t>(4 * batch), 0.0F);
    seed_initial_states_and_inputs(h_X0, h_U, batch, params);

    std::vector<float> h_X_cpu = h_X0;
    gclp::ScopedWallTimer cpu_timer;
    for (int k = 0; k < steps; ++k) {
      cpu_step(h_X_cpu, h_U, batch, dt, params);
    }
    const double cpu_ms = cpu_timer.elapsed_ms();

    float * d_X = nullptr;
    float * d_U = nullptr;
    float * d_R_batch = nullptr;
    float * d_E_batch = nullptr;
    float * d_thrust_body_batch = nullptr;
    float * d_omega_batch = nullptr;
    float * d_accel_world_batch = nullptr;
    float * d_euler_dot_batch = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_X, h_X0.size() * sizeof(float)), "cudaMalloc d_X");
    gclp::throw_on_cuda_error(cudaMalloc(&d_U, h_U.size() * sizeof(float)), "cudaMalloc d_U");
    gclp::throw_on_cuda_error(cudaMalloc(&d_R_batch, static_cast<std::size_t>(9 * batch) * sizeof(float)), "cudaMalloc d_R_batch");
    gclp::throw_on_cuda_error(cudaMalloc(&d_E_batch, static_cast<std::size_t>(9 * batch) * sizeof(float)), "cudaMalloc d_E_batch");
    gclp::throw_on_cuda_error(cudaMalloc(&d_thrust_body_batch, static_cast<std::size_t>(3 * batch) * sizeof(float)), "cudaMalloc d_thrust");
    gclp::throw_on_cuda_error(cudaMalloc(&d_omega_batch, static_cast<std::size_t>(3 * batch) * sizeof(float)), "cudaMalloc d_omega");
    gclp::throw_on_cuda_error(cudaMalloc(&d_accel_world_batch, static_cast<std::size_t>(3 * batch) * sizeof(float)), "cudaMalloc d_accel");
    gclp::throw_on_cuda_error(cudaMalloc(&d_euler_dot_batch, static_cast<std::size_t>(3 * batch) * sizeof(float)), "cudaMalloc d_euler_dot");

    gclp::throw_on_cuda_error(cudaMemcpy(d_X, h_X0.data(), h_X0.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D X0");
    gclp::throw_on_cuda_error(cudaMemcpy(d_U, h_U.data(), h_U.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D U");

    cublasHandle_t handle = nullptr;
    throw_on_cublas(cublasCreate(&handle), "cublasCreate");

    cudaStream_t stream = nullptr;
    gclp::throw_on_cuda_error(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    throw_on_cublas(cublasSetStream(handle, stream), "cublasSetStream");

    constexpr int threads = 256;
    const int blocks = (batch + threads - 1) / threads;

    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_start), "cudaEventCreate start");
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_stop), "cudaEventCreate stop");

    gclp::throw_on_cuda_error(cudaEventRecord(gpu_start, stream), "cudaEventRecord start");
    for (int k = 0; k < steps; ++k) {
      build_batch_matrices_kernel<<<blocks, threads, 0, stream>>>(
        d_X, d_U, d_R_batch, d_E_batch, d_thrust_body_batch, d_omega_batch, batch);
      gclp::throw_on_cuda_error(cudaGetLastError(), "build_batch_matrices_kernel launch");

      throw_on_cublas(
        cublasSgemmStridedBatched(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          kVec3, 1, kVec3,
          &alpha,
          d_R_batch, 3, 9,
          d_thrust_body_batch, 3, 3,
          &beta,
          d_accel_world_batch, 3, 3,
          batch),
        "cublasSgemmStridedBatched R*u");

      throw_on_cublas(
        cublasSgemmStridedBatched(
          handle, CUBLAS_OP_N, CUBLAS_OP_N,
          kVec3, 1, kVec3,
          &alpha,
          d_E_batch, 3, 9,
          d_omega_batch, 3, 3,
          &beta,
          d_euler_dot_batch, 3, 3,
          batch),
        "cublasSgemmStridedBatched E*omega");

      integrate_batch_kernel<<<blocks, threads, 0, stream>>>(
        d_X, d_U, d_accel_world_batch, d_euler_dot_batch, batch, dt, params);
      gclp::throw_on_cuda_error(cudaGetLastError(), "integrate_batch_kernel launch");
    }
    gclp::throw_on_cuda_error(cudaEventRecord(gpu_stop, stream), "cudaEventRecord stop");
    gclp::throw_on_cuda_error(cudaEventSynchronize(gpu_stop), "cudaEventSynchronize stop");

    float gpu_ms = 0.0F;
    gclp::throw_on_cuda_error(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop), "cudaEventElapsedTime");

    std::vector<float> h_X_gpu(static_cast<std::size_t>(kStateDim * batch), 0.0F);
    gclp::throw_on_cuda_error(cudaMemcpy(h_X_gpu.data(), d_X, h_X_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H X");

    double max_abs_error = 0.0;
    double rms_error = 0.0;
    for (std::size_t i = 0; i < h_X_gpu.size(); ++i) {
      const double e = std::abs(static_cast<double>(h_X_cpu[i]) - static_cast<double>(h_X_gpu[i]));
      max_abs_error = std::max(max_abs_error, e);
      rms_error += e * e;
    }
    rms_error = std::sqrt(rms_error / static_cast<double>(h_X_gpu.size()));

    std::cout << "Lesson 14: Nonlinear quadrotor batch simulation CPU vs CUDA+cuBLAS\n";
    std::cout << "Batch: " << batch << ", dt: " << dt << " s, sim_time: " << sim_time
              << " s, steps: " << steps << "\n";
    std::cout << "CPU time (ms): " << cpu_ms << "\n";
    std::cout << "GPU time (ms): " << gpu_ms << "\n";
    std::cout << "Speedup CPU/GPU: " << (cpu_ms / std::max<double>(gpu_ms, 1.0e-9)) << "x\n";
    std::cout << "Final state max abs error: " << max_abs_error << "\n";
    std::cout << "Final state RMS error: " << rms_error << "\n\n";

    std::cout << "Sample final positions z (CPU vs GPU):\n";
    for (int b = 0; b < 6; ++b) {
      const float z_cpu = h_X_cpu[idx_state(b, 2, batch)];
      const float z_gpu = h_X_gpu[idx_state(b, 2, batch)];
      std::cout << "  batch#" << b << ": " << z_cpu << " vs " << z_gpu << "\n";
    }

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    cudaFree(d_X);
    cudaFree(d_U);
    cudaFree(d_R_batch);
    cudaFree(d_E_batch);
    cudaFree(d_thrust_body_batch);
    cudaFree(d_omega_batch);
    cudaFree(d_accel_world_batch);
    cudaFree(d_euler_dot_batch);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
