#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#if GCLP_HAVE_CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif

namespace gclp = gato_cuda_learning_path;

namespace
{

constexpr int kRotSize = 9;

struct Vec3
{
  float x;
  float y;
  float z;
};

__host__ __device__ inline std::size_t idx_pt(int b, int p, int c, int points)
{
  return static_cast<std::size_t>((b * points + p) * 3 + c);
}

__host__ __device__ inline Vec3 mat3_mul_vec(const float * R, const Vec3 & v)
{
  Vec3 out{};
  out.x = R[0] * v.x + R[3] * v.y + R[6] * v.z;
  out.y = R[1] * v.x + R[4] * v.y + R[7] * v.z;
  out.z = R[2] * v.x + R[5] * v.y + R[8] * v.z;
  return out;
}

__host__ __device__ inline void mat3_mul(const float * A, const float * B, float * C)
{
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      C[c * 3 + r] = A[0 * 3 + r] * B[c * 3 + 0]
        + A[1 * 3 + r] * B[c * 3 + 1]
        + A[2 * 3 + r] * B[c * 3 + 2];
    }
  }
}

__host__ __device__ inline float clamp_abs(float x, float min_abs)
{
  return (fabsf(x) < min_abs) ? ((x >= 0.0F) ? min_abs : -min_abs) : x;
}

__host__ __device__ inline void exp_so3(float wx, float wy, float wz, float * R)
{
  const float theta = sqrtf(wx * wx + wy * wy + wz * wz);
  const float eps = 1.0e-6F;
  const float ax = (theta > eps) ? wx / theta : wx;
  const float ay = (theta > eps) ? wy / theta : wy;
  const float az = (theta > eps) ? wz / theta : wz;
  const float s = (theta > eps) ? sinf(theta) : theta;
  const float c = (theta > eps) ? cosf(theta) : (1.0F - 0.5F * theta * theta);
  const float one_c = 1.0F - c;

  R[0] = c + ax * ax * one_c;
  R[1] = ay * ax * one_c + az * s;
  R[2] = az * ax * one_c - ay * s;
  R[3] = ax * ay * one_c - az * s;
  R[4] = c + ay * ay * one_c;
  R[5] = az * ay * one_c + ax * s;
  R[6] = ax * az * one_c + ay * s;
  R[7] = ay * az * one_c - ax * s;
  R[8] = c + az * az * one_c;
}

void solve_6x6(float * A, float * b)
{
  float aug[6][7];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      aug[r][c] = A[r * 6 + c];
    }
    aug[r][6] = b[r];
  }

  for (int i = 0; i < 6; ++i) {
    int piv = i;
    float max_abs = fabsf(aug[i][i]);
    for (int r = i + 1; r < 6; ++r) {
      const float v = fabsf(aug[r][i]);
      if (v > max_abs) {
        max_abs = v;
        piv = r;
      }
    }
    if (piv != i) {
      for (int c = i; c < 7; ++c) {
        std::swap(aug[i][c], aug[piv][c]);
      }
    }
    const float diag = clamp_abs(aug[i][i], 1.0e-6F);
    for (int c = i; c < 7; ++c) {
      aug[i][c] /= diag;
    }
    for (int r = i + 1; r < 6; ++r) {
      const float f = aug[r][i];
      for (int c = i; c < 7; ++c) {
        aug[r][c] -= f * aug[i][c];
      }
    }
  }
  for (int i = 5; i >= 0; --i) {
    for (int r = 0; r < i; ++r) {
      aug[r][6] -= aug[r][i] * aug[i][6];
    }
  }
  for (int i = 0; i < 6; ++i) {
    b[i] = aug[i][6];
  }
}

void generate_problem(
  std::vector<float> & src,
  std::vector<float> & tgt,
  std::vector<float> & t_true,
  int batch,
  int points)
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.0F, 1.0F);
  std::normal_distribution<float> noise(0.0F, 0.002F);

  std::vector<Vec3> base(static_cast<std::size_t>(points));
  for (int p = 0; p < points; ++p) {
    base[static_cast<std::size_t>(p)] = Vec3{u(rng), u(rng), u(rng)};
  }

  for (int b = 0; b < batch; ++b) {
    const float tx = 0.2F * std::sin(0.01F * static_cast<float>(b));
    const float ty = -0.15F * std::cos(0.013F * static_cast<float>(b));
    const float tz = 0.1F * std::sin(0.009F * static_cast<float>(b));
    const float wx = 0.15F * std::sin(0.008F * static_cast<float>(b));
    const float wy = -0.12F * std::cos(0.006F * static_cast<float>(b));
    const float wz = 0.1F * std::sin(0.01F * static_cast<float>(b));
    float Rb[9];
    exp_so3(wx, wy, wz, Rb);

    t_true[static_cast<std::size_t>(b) * 3] = tx;
    t_true[static_cast<std::size_t>(b) * 3 + 1] = ty;
    t_true[static_cast<std::size_t>(b) * 3 + 2] = tz;

    for (int p = 0; p < points; ++p) {
      const Vec3 s = base[static_cast<std::size_t>(p)];
      src[idx_pt(b, p, 0, points)] = s.x;
      src[idx_pt(b, p, 1, points)] = s.y;
      src[idx_pt(b, p, 2, points)] = s.z;

      Vec3 q = mat3_mul_vec(Rb, s);
      q.x += tx + noise(rng);
      q.y += ty + noise(rng);
      q.z += tz + noise(rng);
      tgt[idx_pt(b, p, 0, points)] = q.x;
      tgt[idx_pt(b, p, 1, points)] = q.y;
      tgt[idx_pt(b, p, 2, points)] = q.z;
    }
  }
}

void compute_normals_cpu(
  const std::vector<float> & src,
  const std::vector<float> & tgt,
  const std::vector<float> & R_batch,
  const std::vector<float> & t_batch,
  std::vector<float> & H_batch,
  std::vector<float> & g_batch,
  int batch,
  int points)
{
  std::fill(H_batch.begin(), H_batch.end(), 0.0F);
  std::fill(g_batch.begin(), g_batch.end(), 0.0F);

  for (int b = 0; b < batch; ++b) {
    const float * R = &R_batch[static_cast<std::size_t>(b) * 9];
    const float tx = t_batch[static_cast<std::size_t>(b) * 3];
    const float ty = t_batch[static_cast<std::size_t>(b) * 3 + 1];
    const float tz = t_batch[static_cast<std::size_t>(b) * 3 + 2];
    float * H = &H_batch[static_cast<std::size_t>(b) * 36];
    float * g = &g_batch[static_cast<std::size_t>(b) * 6];

    for (int p = 0; p < points; ++p) {
      const Vec3 s{
        src[idx_pt(b, p, 0, points)],
        src[idx_pt(b, p, 1, points)],
        src[idx_pt(b, p, 2, points)]};
      const Vec3 q{
        tgt[idx_pt(b, p, 0, points)],
        tgt[idx_pt(b, p, 1, points)],
        tgt[idx_pt(b, p, 2, points)]};

      Vec3 rp = mat3_mul_vec(R, s);
      rp.x += tx;
      rp.y += ty;
      rp.z += tz;
      const Vec3 r{rp.x - q.x, rp.y - q.y, rp.z - q.z};

      const float J[3][6] = {
        {1.0F, 0.0F, 0.0F, 0.0F, rp.z - tz, -(rp.y - ty)},
        {0.0F, 1.0F, 0.0F, -(rp.z - tz), 0.0F, rp.x - tx},
        {0.0F, 0.0F, 1.0F, rp.y - ty, -(rp.x - tx), 0.0F}
      };
      const float rv[3] = {r.x, r.y, r.z};

      for (int i = 0; i < 6; ++i) {
        float jtr = 0.0F;
        for (int rr = 0; rr < 3; ++rr) {
          jtr += J[rr][i] * rv[rr];
        }
        g[i] += jtr;
        for (int j = 0; j < 6; ++j) {
          float v = 0.0F;
          for (int rr = 0; rr < 3; ++rr) {
            v += J[rr][i] * J[rr][j];
          }
          H[i * 6 + j] += v;
        }
      }
    }
  }
}

__global__ void zero_pose_kernel(float * R_batch, float * t_batch, int batch)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) {
    return;
  }
  float * R = &R_batch[static_cast<std::size_t>(b) * 9];
  for (int i = 0; i < 9; ++i) {
    R[i] = 0.0F;
  }
  R[0] = 1.0F;
  R[4] = 1.0F;
  R[8] = 1.0F;
  t_batch[static_cast<std::size_t>(b) * 3] = 0.0F;
  t_batch[static_cast<std::size_t>(b) * 3 + 1] = 0.0F;
  t_batch[static_cast<std::size_t>(b) * 3 + 2] = 0.0F;
}

__global__ void build_normal_equations_kernel(
  const float * src,
  const float * tgt,
  const float * R_batch,
  const float * t_batch,
  float * H_batch,
  float * g_batch,
  int points)
{
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int kTerms = 42;  // 36 H + 6 g
  __shared__ float sdata[128 * kTerms];

  float local[kTerms];
  for (int i = 0; i < kTerms; ++i) {
    local[i] = 0.0F;
  }

  const float * R = &R_batch[static_cast<std::size_t>(b) * 9];
  const float tx = t_batch[static_cast<std::size_t>(b) * 3];
  const float ty = t_batch[static_cast<std::size_t>(b) * 3 + 1];
  const float tz = t_batch[static_cast<std::size_t>(b) * 3 + 2];

  for (int p = tid; p < points; p += blockDim.x) {
    const Vec3 s{
      src[idx_pt(b, p, 0, points)],
      src[idx_pt(b, p, 1, points)],
      src[idx_pt(b, p, 2, points)]};
    const Vec3 q{
      tgt[idx_pt(b, p, 0, points)],
      tgt[idx_pt(b, p, 1, points)],
      tgt[idx_pt(b, p, 2, points)]};

    Vec3 rp = mat3_mul_vec(R, s);
    rp.x += tx;
    rp.y += ty;
    rp.z += tz;
    const Vec3 r{rp.x - q.x, rp.y - q.y, rp.z - q.z};

    const float J[3][6] = {
      {1.0F, 0.0F, 0.0F, 0.0F, rp.z - tz, -(rp.y - ty)},
      {0.0F, 1.0F, 0.0F, -(rp.z - tz), 0.0F, rp.x - tx},
      {0.0F, 0.0F, 1.0F, rp.y - ty, -(rp.x - tx), 0.0F}
    };
    const float rv[3] = {r.x, r.y, r.z};

    for (int i = 0; i < 6; ++i) {
      float gi = 0.0F;
      for (int rr = 0; rr < 3; ++rr) {
        gi += J[rr][i] * rv[rr];
      }
      local[36 + i] += gi;
      for (int j = 0; j < 6; ++j) {
        float hij = 0.0F;
        for (int rr = 0; rr < 3; ++rr) {
          hij += J[rr][i] * J[rr][j];
        }
        local[i * 6 + j] += hij;
      }
    }
  }

  float * sb = &sdata[tid * kTerms];
  for (int i = 0; i < kTerms; ++i) {
    sb[i] = local[i];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float * a = &sdata[tid * kTerms];
      float * bptr = &sdata[(tid + stride) * kTerms];
      for (int i = 0; i < kTerms; ++i) {
        a[i] += bptr[i];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float * H = &H_batch[static_cast<std::size_t>(b) * 36];
    float * g = &g_batch[static_cast<std::size_t>(b) * 6];
    const float * out = &sdata[0];
    for (int i = 0; i < 36; ++i) {
      H[i] = out[i];
    }
    for (int i = 0; i < 6; ++i) {
      g[i] = out[36 + i];
    }
  }
}

__device__ inline void solve_6x6_device(float * A, float * b)
{
  float aug[6][7];
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      aug[r][c] = A[r * 6 + c];
    }
    aug[r][6] = b[r];
  }
  for (int i = 0; i < 6; ++i) {
    int piv = i;
    float max_abs = fabsf(aug[i][i]);
    for (int r = i + 1; r < 6; ++r) {
      const float v = fabsf(aug[r][i]);
      if (v > max_abs) {
        max_abs = v;
        piv = r;
      }
    }
    if (piv != i) {
      for (int c = i; c < 7; ++c) {
        const float tmp = aug[i][c];
        aug[i][c] = aug[piv][c];
        aug[piv][c] = tmp;
      }
    }
    const float diag = clamp_abs(aug[i][i], 1.0e-6F);
    for (int c = i; c < 7; ++c) {
      aug[i][c] /= diag;
    }
    for (int r = i + 1; r < 6; ++r) {
      const float f = aug[r][i];
      for (int c = i; c < 7; ++c) {
        aug[r][c] -= f * aug[i][c];
      }
    }
  }
  for (int i = 5; i >= 0; --i) {
    for (int r = 0; r < i; ++r) {
      aug[r][6] -= aug[r][i] * aug[i][6];
    }
  }
  for (int i = 0; i < 6; ++i) {
    b[i] = aug[i][6];
  }
}

__global__ void solve_and_update_kernel(
  float * H_batch, float * g_batch, float * R_batch, float * t_batch, float damping, int batch)
{
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch) {
    return;
  }

  float * H = &H_batch[static_cast<std::size_t>(b) * 36];
  float * g = &g_batch[static_cast<std::size_t>(b) * 6];
  for (int i = 0; i < 6; ++i) {
    H[i * 6 + i] += damping;
    g[i] = -g[i];
  }
  solve_6x6_device(H, g);

  t_batch[static_cast<std::size_t>(b) * 3] += g[0];
  t_batch[static_cast<std::size_t>(b) * 3 + 1] += g[1];
  t_batch[static_cast<std::size_t>(b) * 3 + 2] += g[2];

  float dR[9];
  exp_so3(g[3], g[4], g[5], dR);
  float Rold[9];
  float * R = &R_batch[static_cast<std::size_t>(b) * 9];
  for (int i = 0; i < 9; ++i) {
    Rold[i] = R[i];
  }
  mat3_mul(dR, Rold, R);
}

#if GCLP_HAVE_CERES
struct CeresPointResidual
{
  CeresPointResidual(const Vec3 & s, const Vec3 & q)
  : s_(s), q_(q) {}

  template<typename T>
  bool operator()(const T * const pose, T * residual) const
  {
    // pose = [tx, ty, tz, rx, ry, rz] (angle-axis rotation)
    const T p[3] = {T(s_.x), T(s_.y), T(s_.z)};
    T rp[3];
    ceres::AngleAxisRotatePoint(pose + 3, p, rp);
    residual[0] = rp[0] + pose[0] - T(q_.x);
    residual[1] = rp[1] + pose[1] - T(q_.y);
    residual[2] = rp[2] + pose[2] - T(q_.z);
    return true;
  }

  Vec3 s_;
  Vec3 q_;
};
#endif

}  // namespace

int main()
{
  try {
    constexpr int batch = 256;
    constexpr int points = 256;
    constexpr int iterations = 10;
    constexpr float damping = 1.0e-3F;

    std::vector<float> h_src(static_cast<std::size_t>(batch * points * 3));
    std::vector<float> h_tgt(static_cast<std::size_t>(batch * points * 3));
    std::vector<float> h_t_true(static_cast<std::size_t>(batch * 3));
    generate_problem(h_src, h_tgt, h_t_true, batch, points);

    // Custom CPU Gauss-Newton
    std::vector<float> h_R_cpu(static_cast<std::size_t>(batch * 9), 0.0F);
    std::vector<float> h_t_cpu(static_cast<std::size_t>(batch * 3), 0.0F);
    for (int b = 0; b < batch; ++b) {
      h_R_cpu[static_cast<std::size_t>(b) * 9 + 0] = 1.0F;
      h_R_cpu[static_cast<std::size_t>(b) * 9 + 4] = 1.0F;
      h_R_cpu[static_cast<std::size_t>(b) * 9 + 8] = 1.0F;
    }
    std::vector<float> h_H(static_cast<std::size_t>(batch * 36));
    std::vector<float> h_g(static_cast<std::size_t>(batch * 6));

    gclp::ScopedWallTimer cpu_custom_timer;
    for (int it = 0; it < iterations; ++it) {
      compute_normals_cpu(h_src, h_tgt, h_R_cpu, h_t_cpu, h_H, h_g, batch, points);
      for (int b = 0; b < batch; ++b) {
        float * Hb = &h_H[static_cast<std::size_t>(b) * 36];
        float * gb = &h_g[static_cast<std::size_t>(b) * 6];
        for (int i = 0; i < 6; ++i) {
          Hb[i * 6 + i] += damping;
          gb[i] = -gb[i];
        }
        solve_6x6(Hb, gb);
        h_t_cpu[static_cast<std::size_t>(b) * 3] += gb[0];
        h_t_cpu[static_cast<std::size_t>(b) * 3 + 1] += gb[1];
        h_t_cpu[static_cast<std::size_t>(b) * 3 + 2] += gb[2];
        float dR[9];
        exp_so3(gb[3], gb[4], gb[5], dR);
        float Rold[9];
        float * Rb = &h_R_cpu[static_cast<std::size_t>(b) * 9];
        for (int i = 0; i < 9; ++i) {
          Rold[i] = Rb[i];
        }
        mat3_mul(dR, Rold, Rb);
      }
    }
    const double cpu_custom_ms = cpu_custom_timer.elapsed_ms();

    // Custom GPU Gauss-Newton
    float * d_src = nullptr;
    float * d_tgt = nullptr;
    float * d_R = nullptr;
    float * d_t = nullptr;
    float * d_H = nullptr;
    float * d_g = nullptr;

    gclp::throw_on_cuda_error(cudaMalloc(&d_src, h_src.size() * sizeof(float)), "cudaMalloc d_src");
    gclp::throw_on_cuda_error(cudaMalloc(&d_tgt, h_tgt.size() * sizeof(float)), "cudaMalloc d_tgt");
    gclp::throw_on_cuda_error(cudaMalloc(&d_R, static_cast<std::size_t>(batch * 9) * sizeof(float)), "cudaMalloc d_R");
    gclp::throw_on_cuda_error(cudaMalloc(&d_t, static_cast<std::size_t>(batch * 3) * sizeof(float)), "cudaMalloc d_t");
    gclp::throw_on_cuda_error(cudaMalloc(&d_H, static_cast<std::size_t>(batch * 36) * sizeof(float)), "cudaMalloc d_H");
    gclp::throw_on_cuda_error(cudaMalloc(&d_g, static_cast<std::size_t>(batch * 6) * sizeof(float)), "cudaMalloc d_g");

    gclp::throw_on_cuda_error(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D src");
    gclp::throw_on_cuda_error(cudaMemcpy(d_tgt, h_tgt.data(), h_tgt.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D tgt");

    constexpr int threads_pose = 256;
    const int blocks_pose = (batch + threads_pose - 1) / threads_pose;
    zero_pose_kernel<<<blocks_pose, threads_pose>>>(d_R, d_t, batch);
    gclp::throw_on_cuda_error(cudaGetLastError(), "zero_pose_kernel launch");

    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_start), "cudaEventCreate start");
    gclp::throw_on_cuda_error(cudaEventCreate(&gpu_stop), "cudaEventCreate stop");

    constexpr int threads_reduce = 128;
    gclp::throw_on_cuda_error(cudaEventRecord(gpu_start), "cudaEventRecord start");
    for (int it = 0; it < iterations; ++it) {
      build_normal_equations_kernel<<<batch, threads_reduce>>>(d_src, d_tgt, d_R, d_t, d_H, d_g, points);
      gclp::throw_on_cuda_error(cudaGetLastError(), "build_normal_equations_kernel launch");
      solve_and_update_kernel<<<blocks_pose, threads_pose>>>(d_H, d_g, d_R, d_t, damping, batch);
      gclp::throw_on_cuda_error(cudaGetLastError(), "solve_and_update_kernel launch");
    }
    gclp::throw_on_cuda_error(cudaEventRecord(gpu_stop), "cudaEventRecord stop");
    gclp::throw_on_cuda_error(cudaEventSynchronize(gpu_stop), "cudaEventSynchronize stop");

    float gpu_custom_ms = 0.0F;
    gclp::throw_on_cuda_error(cudaEventElapsedTime(&gpu_custom_ms, gpu_start, gpu_stop), "cudaEventElapsedTime");

    std::vector<float> h_t_gpu(static_cast<std::size_t>(batch * 3));
    gclp::throw_on_cuda_error(cudaMemcpy(h_t_gpu.data(), d_t, h_t_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H t");

    // Ceres CPU (if available)
    double ceres_cpu_ms = -1.0;
    std::vector<float> h_t_ceres(static_cast<std::size_t>(batch * 3), 0.0F);
#if GCLP_HAVE_CERES
    {
      gclp::ScopedWallTimer ceres_timer;
      for (int b = 0; b < batch; ++b) {
        double pose[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ceres::Problem problem;
        for (int p = 0; p < points; ++p) {
          Vec3 s{
            h_src[idx_pt(b, p, 0, points)],
            h_src[idx_pt(b, p, 1, points)],
            h_src[idx_pt(b, p, 2, points)]};
          Vec3 q{
            h_tgt[idx_pt(b, p, 0, points)],
            h_tgt[idx_pt(b, p, 1, points)],
            h_tgt[idx_pt(b, p, 2, points)]};
          ceres::CostFunction * cost =
            new ceres::AutoDiffCostFunction<CeresPointResidual, 3, 6>(new CeresPointResidual(s, q));
          problem.AddResidualBlock(cost, nullptr, pose);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = iterations;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        h_t_ceres[static_cast<std::size_t>(b) * 3] = static_cast<float>(pose[0]);
        h_t_ceres[static_cast<std::size_t>(b) * 3 + 1] = static_cast<float>(pose[1]);
        h_t_ceres[static_cast<std::size_t>(b) * 3 + 2] = static_cast<float>(pose[2]);
      }
      ceres_cpu_ms = ceres_timer.elapsed_ms();
    }
#endif

    auto mean_t_err = [&](const std::vector<float> & t_est) {
      double mean = 0.0;
      for (int b = 0; b < batch; ++b) {
        const float ex = t_est[static_cast<std::size_t>(b) * 3] - h_t_true[static_cast<std::size_t>(b) * 3];
        const float ey = t_est[static_cast<std::size_t>(b) * 3 + 1] - h_t_true[static_cast<std::size_t>(b) * 3 + 1];
        const float ez = t_est[static_cast<std::size_t>(b) * 3 + 2] - h_t_true[static_cast<std::size_t>(b) * 3 + 2];
        mean += std::sqrt(static_cast<double>(ex * ex + ey * ey + ez * ez));
      }
      return mean / static_cast<double>(batch);
    };

    const double err_cpu_custom = mean_t_err(h_t_cpu);
    const double err_gpu_custom = mean_t_err(h_t_gpu);

    std::cout << "Lesson 16: ICP pose estimation benchmark (same problem)\n";
    std::cout << "Solvers: Ceres CPU, custom CPU Gauss-Newton, custom GPU Gauss-Newton\n";
    std::cout << "Batch=" << batch << ", points=" << points << ", iters=" << iterations << "\n";
    std::cout << "Custom CPU time (ms): " << cpu_custom_ms << "\n";
    std::cout << "Custom GPU time (ms): " << gpu_custom_ms << "\n";
    std::cout << "Custom CPU/GPU speedup: " << (cpu_custom_ms / std::max<double>(gpu_custom_ms, 1.0e-9)) << "x\n";
    std::cout << "Mean translation error (custom CPU / custom GPU): "
              << err_cpu_custom << " / " << err_gpu_custom << "\n";
#if GCLP_HAVE_CERES
    const double err_ceres = mean_t_err(h_t_ceres);
    std::cout << "Ceres CPU time (ms): " << ceres_cpu_ms << "\n";
    std::cout << "Mean translation error (Ceres): " << err_ceres << "\n";
    std::cout << "Speedup custom GPU vs Ceres CPU: "
              << (ceres_cpu_ms / std::max<double>(gpu_custom_ms, 1.0e-9)) << "x\n";
#else
    std::cout << "Ceres CPU: unavailable at build time (install Ceres and rebuild to enable).\n";
#endif

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cudaFree(d_src);
    cudaFree(d_tgt);
    cudaFree(d_R);
    cudaFree(d_t);
    cudaFree(d_H);
    cudaFree(d_g);
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
