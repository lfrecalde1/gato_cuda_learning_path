#include "gato_cuda_learning_path/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if GCLP_HAVE_CERES
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif

namespace gclp = gato_cuda_learning_path;

namespace
{

struct Vec3
{
  float x;
  float y;
  float z;
};

struct Pose
{
  float R[9];  // column-major
  float t[3];
};

struct GtSample
{
  int64_t t;
  Pose pose;
};

Pose pose_identity()
{
  Pose p{};
  p.R[0] = 1.0F; p.R[3] = 0.0F; p.R[6] = 0.0F;
  p.R[1] = 0.0F; p.R[4] = 1.0F; p.R[7] = 0.0F;
  p.R[2] = 0.0F; p.R[5] = 0.0F; p.R[8] = 1.0F;
  p.t[0] = 0.0F; p.t[1] = 0.0F; p.t[2] = 0.0F;
  return p;
}

__host__ __device__ inline std::size_t idx_pt(int b, int p, int c, int points)
{
  return static_cast<std::size_t>((b * points + p) * 3 + c);
}

__host__ __device__ inline Vec3 mat3_mul_vec(const float * R, const Vec3 & v)
{
  return Vec3{
    R[0] * v.x + R[3] * v.y + R[6] * v.z,
    R[1] * v.x + R[4] * v.y + R[7] * v.z,
    R[2] * v.x + R[5] * v.y + R[8] * v.z};
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

__host__ __device__ inline void mat3_transpose(const float * A, float * At)
{
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      At[c * 3 + r] = A[r * 3 + c];
    }
  }
}

__host__ __device__ inline float clamp_abs(float x, float min_abs)
{
  return (fabsf(x) < min_abs) ? ((x >= 0.0F) ? min_abs : -min_abs) : x;
}

Pose pose_inverse(const Pose & p)
{
  Pose inv{};
  mat3_transpose(p.R, inv.R);
  const Vec3 tp{p.t[0], p.t[1], p.t[2]};
  const Vec3 tinv = mat3_mul_vec(inv.R, Vec3{-tp.x, -tp.y, -tp.z});
  inv.t[0] = tinv.x;
  inv.t[1] = tinv.y;
  inv.t[2] = tinv.z;
  return inv;
}

Pose pose_mul(const Pose & a, const Pose & b)
{
  Pose c{};
  mat3_mul(a.R, b.R, c.R);
  const Vec3 bt{b.t[0], b.t[1], b.t[2]};
  const Vec3 abt = mat3_mul_vec(a.R, bt);
  c.t[0] = abt.x + a.t[0];
  c.t[1] = abt.y + a.t[1];
  c.t[2] = abt.z + a.t[2];
  return c;
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

void quat_to_rot(float qx, float qy, float qz, float qw, float * R)
{
  const float n = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (n < 1.0e-8F) {
    R[0] = 1.0F; R[3] = 0.0F; R[6] = 0.0F;
    R[1] = 0.0F; R[4] = 1.0F; R[7] = 0.0F;
    R[2] = 0.0F; R[5] = 0.0F; R[8] = 1.0F;
    return;
  }
  qx /= n; qy /= n; qz /= n; qw /= n;
  const float xx = qx * qx;
  const float yy = qy * qy;
  const float zz = qz * qz;
  const float xy = qx * qy;
  const float xz = qx * qz;
  const float yz = qy * qz;
  const float wx = qw * qx;
  const float wy = qw * qy;
  const float wz = qw * qz;
  R[0] = 1.0F - 2.0F * (yy + zz);
  R[3] = 2.0F * (xy - wz);
  R[6] = 2.0F * (xz + wy);
  R[1] = 2.0F * (xy + wz);
  R[4] = 1.0F - 2.0F * (xx + zz);
  R[7] = 2.0F * (yz - wx);
  R[2] = 2.0F * (xz - wy);
  R[5] = 2.0F * (yz + wx);
  R[8] = 1.0F - 2.0F * (xx + yy);
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
    float max_abs = std::abs(aug[i][i]);
    for (int r = i + 1; r < 6; ++r) {
      const float v = std::abs(aug[r][i]);
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

void nearest_correspondences(
  const std::vector<Vec3> & src,
  const std::vector<Vec3> & tgt,
  std::vector<Vec3> & tgt_match)
{
  tgt_match.resize(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    float best_d2 = std::numeric_limits<float>::max();
    int best_j = 0;
    for (std::size_t j = 0; j < tgt.size(); ++j) {
      const float dx = src[i].x - tgt[j].x;
      const float dy = src[i].y - tgt[j].y;
      const float dz = src[i].z - tgt[j].z;
      const float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_j = static_cast<int>(j);
      }
    }
    tgt_match[i] = tgt[static_cast<std::size_t>(best_j)];
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
      const Vec3 s{src[idx_pt(b, p, 0, points)], src[idx_pt(b, p, 1, points)], src[idx_pt(b, p, 2, points)]};
      const Vec3 q{tgt[idx_pt(b, p, 0, points)], tgt[idx_pt(b, p, 1, points)], tgt[idx_pt(b, p, 2, points)]};
      Vec3 rp = mat3_mul_vec(R, s);
      rp.x += tx; rp.y += ty; rp.z += tz;
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
          float hij = 0.0F;
          for (int rr = 0; rr < 3; ++rr) {
            hij += J[rr][i] * J[rr][j];
          }
          H[i * 6 + j] += hij;
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
  const float * src, const float * tgt, const float * R_batch, const float * t_batch,
  float * H_batch, float * g_batch, int points)
{
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int kTerms = 42;
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
    const Vec3 s{src[idx_pt(b, p, 0, points)], src[idx_pt(b, p, 1, points)], src[idx_pt(b, p, 2, points)]};
    const Vec3 q{tgt[idx_pt(b, p, 0, points)], tgt[idx_pt(b, p, 1, points)], tgt[idx_pt(b, p, 2, points)]};
    Vec3 rp = mat3_mul_vec(R, s);
    rp.x += tx; rp.y += ty; rp.z += tz;
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
  CeresPointResidual(const Vec3 & s, const Vec3 & q) : s_(s), q_(q) {}
  template<typename T>
  bool operator()(const T * const pose, T * residual) const
  {
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

float rotation_angle_error_rad(const float * R_est, const float * R_gt)
{
  float Rt[9];
  mat3_transpose(R_est, Rt);
  float E[9];
  mat3_mul(Rt, R_gt, E);
  const float tr = E[0] + E[4] + E[8];
  const float cos_ang = std::max(-1.0F, std::min(1.0F, 0.5F * (tr - 1.0F)));
  return std::acos(cos_ang);
}

Pose rel_from_world_poses(const Pose & wT0, const Pose & wT1)
{
  return pose_mul(pose_inverse(wT0), wT1);
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const std::string bag_path =
      (argc > 1) ? std::string(argv[1]) :
      "/home/fer/payload_transportation_ws/feb_19_lidar_vicon_liss_6_5_final";
    const std::string lidar_topic = "/eagle11/livox/points";
    const std::string gt_topic = "/eagle11/odom";

    constexpr int points_per_pair = 256;
    constexpr int max_pairs = 256;
    constexpr int gn_iters = 10;
    constexpr float damping = 1.0e-3F;

    rosbag2_cpp::Reader reader;
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = bag_path;
    storage_options.storage_id = "sqlite3";
    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";
    reader.open(storage_options, converter_options);

    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc_ser;
    rclcpp::Serialization<nav_msgs::msg::Odometry> odom_ser;

    struct TimedCloud {int64_t t; std::vector<Vec3> pts;};
    struct TimedPose {int64_t t; Pose pose;};
    std::vector<TimedCloud> clouds;
    std::vector<TimedPose> gt_poses;

    while (reader.has_next()) {
      auto msg = reader.read_next();
      if (msg->topic_name == lidar_topic) {
        rclcpp::SerializedMessage sm(*msg->serialized_data);
        sensor_msgs::msg::PointCloud2 pc;
        pc_ser.deserialize_message(&sm, &pc);

        int ox = -1, oy = -1, oz = -1;
        for (const auto & f : pc.fields) {
          if (f.name == "x") {ox = static_cast<int>(f.offset);}
          if (f.name == "y") {oy = static_cast<int>(f.offset);}
          if (f.name == "z") {oz = static_cast<int>(f.offset);}
        }
        if (ox < 0 || oy < 0 || oz < 0 || pc.point_step == 0) {
          continue;
        }
        const std::size_t npts = static_cast<std::size_t>(pc.width) * static_cast<std::size_t>(pc.height);
        std::vector<Vec3> pts;
        pts.reserve(npts);
        for (std::size_t i = 0; i < npts; ++i) {
          const std::size_t base = i * pc.point_step;
          if (base + static_cast<std::size_t>(std::max({ox, oy, oz}) + 4) > pc.data.size()) {
            break;
          }
          const float x = *reinterpret_cast<const float *>(&pc.data[base + static_cast<std::size_t>(ox)]);
          const float y = *reinterpret_cast<const float *>(&pc.data[base + static_cast<std::size_t>(oy)]);
          const float z = *reinterpret_cast<const float *>(&pc.data[base + static_cast<std::size_t>(oz)]);
          if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            pts.push_back(Vec3{x, y, z});
          }
        }
        if (!pts.empty()) {
          clouds.push_back(TimedCloud{static_cast<int64_t>(msg->time_stamp), std::move(pts)});
        }
      } else if (msg->topic_name == gt_topic) {
        rclcpp::SerializedMessage sm(*msg->serialized_data);
        nav_msgs::msg::Odometry odom;
        odom_ser.deserialize_message(&sm, &odom);
        Pose p{};
        quat_to_rot(
          static_cast<float>(odom.pose.pose.orientation.x),
          static_cast<float>(odom.pose.pose.orientation.y),
          static_cast<float>(odom.pose.pose.orientation.z),
          static_cast<float>(odom.pose.pose.orientation.w),
          p.R);
        p.t[0] = static_cast<float>(odom.pose.pose.position.x);
        p.t[1] = static_cast<float>(odom.pose.pose.position.y);
        p.t[2] = static_cast<float>(odom.pose.pose.position.z);
        gt_poses.push_back(TimedPose{static_cast<int64_t>(msg->time_stamp), p});
      }
    }

    if (clouds.size() < 2 || gt_poses.size() < 2) {
      std::cerr << "Not enough lidar or odom data in bag.\n";
      return 1;
    }

    auto nearest_pose = [&](int64_t t) -> const GtSample {
      auto it = std::lower_bound(
        gt_poses.begin(), gt_poses.end(), t,
        [](const TimedPose & a, int64_t v) {return a.t < v;});
      if (it == gt_poses.end()) {
        return GtSample{gt_poses.back().t, gt_poses.back().pose};
      }
      if (it == gt_poses.begin()) {
        return GtSample{it->t, it->pose};
      }
      auto prev = it - 1;
      if (std::llabs(it->t - t) < std::llabs(prev->t - t)) {
        return GtSample{it->t, it->pose};
      }
      return GtSample{prev->t, prev->pose};
    };

    int batch = 0;
    std::vector<float> src_batch;
    std::vector<float> tgt_batch;
    std::vector<Pose> gt_rel;
    std::vector<int64_t> pair_t0;
    std::vector<int64_t> pair_t1;
    std::vector<int64_t> gt_t0;
    std::vector<int64_t> gt_t1;
    src_batch.reserve(static_cast<std::size_t>(max_pairs * points_per_pair * 3));
    tgt_batch.reserve(static_cast<std::size_t>(max_pairs * points_per_pair * 3));
    gt_rel.reserve(static_cast<std::size_t>(max_pairs));
    pair_t0.reserve(static_cast<std::size_t>(max_pairs));
    pair_t1.reserve(static_cast<std::size_t>(max_pairs));
    gt_t0.reserve(static_cast<std::size_t>(max_pairs));
    gt_t1.reserve(static_cast<std::size_t>(max_pairs));

    for (std::size_t i = 0; i + 1 < clouds.size() && batch < max_pairs; ++i) {
      const auto & c0 = clouds[i];
      const auto & c1 = clouds[i + 1];
      if (c0.pts.size() < static_cast<std::size_t>(points_per_pair) ||
          c1.pts.size() < static_cast<std::size_t>(points_per_pair)) {
        continue;
      }

      std::vector<Vec3> s(points_per_pair), t(points_per_pair);
      const std::size_t step0 = c0.pts.size() / static_cast<std::size_t>(points_per_pair);
      const std::size_t step1 = c1.pts.size() / static_cast<std::size_t>(points_per_pair);
      for (int p = 0; p < points_per_pair; ++p) {
        s[static_cast<std::size_t>(p)] = c0.pts[static_cast<std::size_t>(p) * step0];
        t[static_cast<std::size_t>(p)] = c1.pts[static_cast<std::size_t>(p) * step1];
      }

      std::vector<Vec3> t_match;
      nearest_correspondences(s, t, t_match);

      for (int p = 0; p < points_per_pair; ++p) {
        src_batch.push_back(s[static_cast<std::size_t>(p)].x);
        src_batch.push_back(s[static_cast<std::size_t>(p)].y);
        src_batch.push_back(s[static_cast<std::size_t>(p)].z);
        tgt_batch.push_back(t_match[static_cast<std::size_t>(p)].x);
        tgt_batch.push_back(t_match[static_cast<std::size_t>(p)].y);
        tgt_batch.push_back(t_match[static_cast<std::size_t>(p)].z);
      }

      const GtSample wT0 = nearest_pose(c0.t);
      const GtSample wT1 = nearest_pose(c1.t);
      Pose rel = rel_from_world_poses(wT0.pose, wT1.pose);
      gt_rel.push_back(rel);
      pair_t0.push_back(c0.t);
      pair_t1.push_back(c1.t);
      gt_t0.push_back(wT0.t);
      gt_t1.push_back(wT1.t);
      ++batch;
    }

    if (batch == 0) {
      std::cerr << "No usable cloud pairs found.\n";
      return 1;
    }

    std::cout << "Loaded bag: " << bag_path << "\n";
    std::cout << "Cloud frames: " << clouds.size() << ", GT poses: " << gt_poses.size() << "\n";
    std::cout << "Evaluation pairs: " << batch << ", points/pair: " << points_per_pair << "\n";

    // Custom CPU
    std::vector<float> R_cpu(static_cast<std::size_t>(batch * 9), 0.0F);
    std::vector<float> t_cpu(static_cast<std::size_t>(batch * 3), 0.0F);
    for (int b = 0; b < batch; ++b) {
      R_cpu[static_cast<std::size_t>(b) * 9 + 0] = 1.0F;
      R_cpu[static_cast<std::size_t>(b) * 9 + 4] = 1.0F;
      R_cpu[static_cast<std::size_t>(b) * 9 + 8] = 1.0F;
    }
    std::vector<float> H_cpu(static_cast<std::size_t>(batch * 36));
    std::vector<float> g_cpu(static_cast<std::size_t>(batch * 6));
    gclp::ScopedWallTimer cpu_timer;
    for (int it = 0; it < gn_iters; ++it) {
      compute_normals_cpu(src_batch, tgt_batch, R_cpu, t_cpu, H_cpu, g_cpu, batch, points_per_pair);
      for (int b = 0; b < batch; ++b) {
        float * H = &H_cpu[static_cast<std::size_t>(b) * 36];
        float * g = &g_cpu[static_cast<std::size_t>(b) * 6];
        for (int i = 0; i < 6; ++i) {
          H[i * 6 + i] += damping;
          g[i] = -g[i];
        }
        solve_6x6(H, g);
        t_cpu[static_cast<std::size_t>(b) * 3] += g[0];
        t_cpu[static_cast<std::size_t>(b) * 3 + 1] += g[1];
        t_cpu[static_cast<std::size_t>(b) * 3 + 2] += g[2];
        float dR[9];
        exp_so3(g[3], g[4], g[5], dR);
        float Rold[9];
        float * R = &R_cpu[static_cast<std::size_t>(b) * 9];
        for (int i = 0; i < 9; ++i) {
          Rold[i] = R[i];
        }
        mat3_mul(dR, Rold, R);
      }
    }
    const double cpu_custom_ms = cpu_timer.elapsed_ms();

    // Custom GPU (optional if CUDA runtime/device is available)
    bool gpu_ok = true;
    float gpu_custom_ms = -1.0F;
    std::string gpu_error;
    std::vector<float> R_gpu(static_cast<std::size_t>(batch * 9), 0.0F);
    std::vector<float> t_gpu(static_cast<std::size_t>(batch * 3), 0.0F);
    float * d_src = nullptr;
    float * d_tgt = nullptr;
    float * d_R = nullptr;
    float * d_t = nullptr;
    float * d_H = nullptr;
    float * d_g = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    auto cuda_try = [&](cudaError_t status, const char * context) {
      if (status != cudaSuccess && gpu_ok) {
        gpu_ok = false;
        gpu_error = std::string(context) + ": " + cudaGetErrorString(status);
      }
    };

    int device_count = 0;
    cuda_try(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
      gpu_ok = false;
      if (gpu_error.empty()) {
        gpu_error = "No CUDA device detected";
      }
    }

    if (gpu_ok) {
      cuda_try(cudaMalloc(&d_src, src_batch.size() * sizeof(float)), "cudaMalloc d_src");
      cuda_try(cudaMalloc(&d_tgt, tgt_batch.size() * sizeof(float)), "cudaMalloc d_tgt");
      cuda_try(cudaMalloc(&d_R, static_cast<std::size_t>(batch * 9) * sizeof(float)), "cudaMalloc d_R");
      cuda_try(cudaMalloc(&d_t, static_cast<std::size_t>(batch * 3) * sizeof(float)), "cudaMalloc d_t");
      cuda_try(cudaMalloc(&d_H, static_cast<std::size_t>(batch * 36) * sizeof(float)), "cudaMalloc d_H");
      cuda_try(cudaMalloc(&d_g, static_cast<std::size_t>(batch * 6) * sizeof(float)), "cudaMalloc d_g");
    }
    if (gpu_ok) {
      cuda_try(cudaMemcpy(d_src, src_batch.data(), src_batch.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D src");
      cuda_try(cudaMemcpy(d_tgt, tgt_batch.data(), tgt_batch.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D tgt");
    }
    if (gpu_ok) {
      const int blocks_pose = (batch + 255) / 256;
      zero_pose_kernel<<<blocks_pose, 256>>>(d_R, d_t, batch);
      cuda_try(cudaGetLastError(), "zero_pose_kernel");
      cuda_try(cudaEventCreate(&gpu_start), "cudaEventCreate start");
      cuda_try(cudaEventCreate(&gpu_stop), "cudaEventCreate stop");
      cuda_try(cudaEventRecord(gpu_start), "cudaEventRecord start");
      for (int it = 0; it < gn_iters && gpu_ok; ++it) {
        build_normal_equations_kernel<<<batch, 128>>>(d_src, d_tgt, d_R, d_t, d_H, d_g, points_per_pair);
        cuda_try(cudaGetLastError(), "build_normal_equations_kernel");
        solve_and_update_kernel<<<blocks_pose, 256>>>(d_H, d_g, d_R, d_t, damping, batch);
        cuda_try(cudaGetLastError(), "solve_and_update_kernel");
      }
      cuda_try(cudaEventRecord(gpu_stop), "cudaEventRecord stop");
      cuda_try(cudaEventSynchronize(gpu_stop), "cudaEventSync stop");
      if (gpu_ok) {
        cuda_try(cudaEventElapsedTime(&gpu_custom_ms, gpu_start, gpu_stop), "cudaEventElapsedTime");
      }
      if (gpu_ok) {
        cuda_try(cudaMemcpy(R_gpu.data(), d_R, R_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H R");
        cuda_try(cudaMemcpy(t_gpu.data(), d_t, t_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H t");
      }
    }

    // Ceres CPU
    double ceres_ms = -1.0;
    std::vector<float> R_ceres(static_cast<std::size_t>(batch * 9), 0.0F);
    std::vector<float> t_ceres(static_cast<std::size_t>(batch * 3), 0.0F);
#if GCLP_HAVE_CERES
    {
      gclp::ScopedWallTimer ceres_timer;
      for (int b = 0; b < batch; ++b) {
        double pose[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ceres::Problem problem;
        for (int p = 0; p < points_per_pair; ++p) {
          Vec3 s{src_batch[idx_pt(b, p, 0, points_per_pair)], src_batch[idx_pt(b, p, 1, points_per_pair)], src_batch[idx_pt(b, p, 2, points_per_pair)]};
          Vec3 q{tgt_batch[idx_pt(b, p, 0, points_per_pair)], tgt_batch[idx_pt(b, p, 1, points_per_pair)], tgt_batch[idx_pt(b, p, 2, points_per_pair)]};
          ceres::CostFunction * cost = new ceres::AutoDiffCostFunction<CeresPointResidual, 3, 6>(new CeresPointResidual(s, q));
          problem.AddResidualBlock(cost, nullptr, pose);
        }
        ceres::Solver::Options options;
        options.max_num_iterations = gn_iters;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        t_ceres[static_cast<std::size_t>(b) * 3] = static_cast<float>(pose[0]);
        t_ceres[static_cast<std::size_t>(b) * 3 + 1] = static_cast<float>(pose[1]);
        t_ceres[static_cast<std::size_t>(b) * 3 + 2] = static_cast<float>(pose[2]);
        float Rc[9];
        exp_so3(static_cast<float>(pose[3]), static_cast<float>(pose[4]), static_cast<float>(pose[5]), Rc);
        for (int i = 0; i < 9; ++i) {
          R_ceres[static_cast<std::size_t>(b) * 9 + i] = Rc[i];
        }
      }
      ceres_ms = ceres_timer.elapsed_ms();
    }
#endif

    auto compute_errors = [&](const std::vector<float> & R_est, const std::vector<float> & t_est) {
      double mean_t = 0.0;
      double mean_r = 0.0;
      for (int b = 0; b < batch; ++b) {
        const Pose & gt = gt_rel[static_cast<std::size_t>(b)];
        const float ex = t_est[static_cast<std::size_t>(b) * 3] - gt.t[0];
        const float ey = t_est[static_cast<std::size_t>(b) * 3 + 1] - gt.t[1];
        const float ez = t_est[static_cast<std::size_t>(b) * 3 + 2] - gt.t[2];
        mean_t += std::sqrt(static_cast<double>(ex * ex + ey * ey + ez * ez));
        mean_r += rotation_angle_error_rad(&R_est[static_cast<std::size_t>(b) * 9], gt.R);
      }
      mean_t /= static_cast<double>(batch);
      mean_r /= static_cast<double>(batch);
      return std::pair<double, double>{mean_t, mean_r};
    };

    const auto [t_err_cpu, r_err_cpu] = compute_errors(R_cpu, t_cpu);
    double t_err_gpu = -1.0;
    double r_err_gpu = -1.0;
    if (gpu_ok) {
      const auto errs = compute_errors(R_gpu, t_gpu);
      t_err_gpu = errs.first;
      r_err_gpu = errs.second;
    }

    std::cout << "Custom CPU time (ms): " << cpu_custom_ms << "\n";
    if (gpu_ok) {
      std::cout << "Custom GPU time (ms): " << gpu_custom_ms << "\n";
      std::cout << "Custom CPU/GPU speedup: " << (cpu_custom_ms / std::max<double>(gpu_custom_ms, 1.0e-9)) << "x\n";
    } else {
      std::cout << "Custom GPU: unavailable (" << gpu_error << ")\n";
    }
    std::cout << "Custom CPU mean errors: t=" << t_err_cpu << " m, r=" << r_err_cpu << " rad\n";
    if (gpu_ok) {
      std::cout << "Custom GPU mean errors: t=" << t_err_gpu << " m, r=" << r_err_gpu << " rad\n";
    }

#if GCLP_HAVE_CERES
    const auto [t_err_ceres, r_err_ceres] = compute_errors(R_ceres, t_ceres);
    std::cout << "Ceres CPU time (ms): " << ceres_ms << "\n";
    std::cout << "Ceres mean errors: t=" << t_err_ceres << " m, r=" << r_err_ceres << " rad\n";
    if (gpu_ok) {
      std::cout << "Speedup GPU custom vs Ceres: " << (ceres_ms / std::max<double>(gpu_custom_ms, 1.0e-9)) << "x\n";
    }
#else
    std::cout << "Ceres unavailable at build time.\n";
#endif

    std::vector<Pose> gt_abs(static_cast<std::size_t>(batch));
    std::vector<Pose> cpu_abs(static_cast<std::size_t>(batch));
    std::vector<Pose> ceres_abs(static_cast<std::size_t>(batch));
    std::vector<Pose> gpu_abs(static_cast<std::size_t>(batch));
    Pose cur_gt = pose_identity();
    Pose cur_cpu = pose_identity();
    Pose cur_ceres = pose_identity();
    Pose cur_gpu = pose_identity();
    for (int b = 0; b < batch; ++b) {
      cur_gt = pose_mul(cur_gt, gt_rel[static_cast<std::size_t>(b)]);
      gt_abs[static_cast<std::size_t>(b)] = cur_gt;

      Pose rel_cpu{};
      for (int i = 0; i < 9; ++i) {
        rel_cpu.R[i] = R_cpu[static_cast<std::size_t>(b) * 9 + i];
      }
      rel_cpu.t[0] = t_cpu[static_cast<std::size_t>(b) * 3];
      rel_cpu.t[1] = t_cpu[static_cast<std::size_t>(b) * 3 + 1];
      rel_cpu.t[2] = t_cpu[static_cast<std::size_t>(b) * 3 + 2];
      cur_cpu = pose_mul(cur_cpu, rel_cpu);
      cpu_abs[static_cast<std::size_t>(b)] = cur_cpu;

      Pose rel_ceres{};
      for (int i = 0; i < 9; ++i) {
        rel_ceres.R[i] = R_ceres[static_cast<std::size_t>(b) * 9 + i];
      }
      rel_ceres.t[0] = t_ceres[static_cast<std::size_t>(b) * 3];
      rel_ceres.t[1] = t_ceres[static_cast<std::size_t>(b) * 3 + 1];
      rel_ceres.t[2] = t_ceres[static_cast<std::size_t>(b) * 3 + 2];
      cur_ceres = pose_mul(cur_ceres, rel_ceres);
      ceres_abs[static_cast<std::size_t>(b)] = cur_ceres;

      if (gpu_ok) {
        Pose rel_gpu{};
        for (int i = 0; i < 9; ++i) {
          rel_gpu.R[i] = R_gpu[static_cast<std::size_t>(b) * 9 + i];
        }
        rel_gpu.t[0] = t_gpu[static_cast<std::size_t>(b) * 3];
        rel_gpu.t[1] = t_gpu[static_cast<std::size_t>(b) * 3 + 1];
        rel_gpu.t[2] = t_gpu[static_cast<std::size_t>(b) * 3 + 2];
        cur_gpu = pose_mul(cur_gpu, rel_gpu);
        gpu_abs[static_cast<std::size_t>(b)] = cur_gpu;
      } else {
        Pose p = pose_identity();
        p.t[0] = std::numeric_limits<float>::quiet_NaN();
        p.t[1] = std::numeric_limits<float>::quiet_NaN();
        p.t[2] = std::numeric_limits<float>::quiet_NaN();
        gpu_abs[static_cast<std::size_t>(b)] = p;
      }
    }

    const std::string csv_path = "lesson17_pose_comparison.csv";
    {
      std::ofstream csv(csv_path);
      csv << "pair_index,pair_t0_ns,pair_t1_ns,gt_t0_ns,gt_t1_ns,"
          << "gt_tx,gt_ty,gt_tz,gt_rot_err_rad_ref0,"
          << "gt_abs_tx,gt_abs_ty,gt_abs_tz,"
          << "cpu_abs_tx,cpu_abs_ty,cpu_abs_tz,cpu_abs_t_err_m,cpu_abs_r_err_rad,"
          << "ceres_tx,ceres_ty,ceres_tz,ceres_t_err_m,ceres_r_err_rad,"
          << "ceres_abs_tx,ceres_abs_ty,ceres_abs_tz,ceres_abs_t_err_m,ceres_abs_r_err_rad,"
          << "gpu_tx,gpu_ty,gpu_tz,gpu_t_err_m,gpu_r_err_rad,"
          << "gpu_abs_tx,gpu_abs_ty,gpu_abs_tz,gpu_abs_t_err_m,gpu_abs_r_err_rad,gpu_available\n";
      for (int b = 0; b < batch; ++b) {
        const Pose & gt = gt_rel[static_cast<std::size_t>(b)];
        const Pose & gt_abs_p = gt_abs[static_cast<std::size_t>(b)];
        const Pose & cpu_abs_p = cpu_abs[static_cast<std::size_t>(b)];
        const Pose & ceres_abs_p = ceres_abs[static_cast<std::size_t>(b)];
        const float ceres_tx = t_ceres[static_cast<std::size_t>(b) * 3];
        const float ceres_ty = t_ceres[static_cast<std::size_t>(b) * 3 + 1];
        const float ceres_tz = t_ceres[static_cast<std::size_t>(b) * 3 + 2];
        const float ceres_ex = ceres_tx - gt.t[0];
        const float ceres_ey = ceres_ty - gt.t[1];
        const float ceres_ez = ceres_tz - gt.t[2];
        const float ceres_t_err = std::sqrt(ceres_ex * ceres_ex + ceres_ey * ceres_ey + ceres_ez * ceres_ez);
        const float ceres_r_err = rotation_angle_error_rad(&R_ceres[static_cast<std::size_t>(b) * 9], gt.R);
        const float ceres_abs_ex = ceres_abs_p.t[0] - gt_abs_p.t[0];
        const float ceres_abs_ey = ceres_abs_p.t[1] - gt_abs_p.t[1];
        const float ceres_abs_ez = ceres_abs_p.t[2] - gt_abs_p.t[2];
        const float ceres_abs_t_err =
          std::sqrt(ceres_abs_ex * ceres_abs_ex + ceres_abs_ey * ceres_abs_ey + ceres_abs_ez * ceres_abs_ez);
        const float ceres_abs_r_err = rotation_angle_error_rad(ceres_abs_p.R, gt_abs_p.R);

        const float cpu_abs_ex = cpu_abs_p.t[0] - gt_abs_p.t[0];
        const float cpu_abs_ey = cpu_abs_p.t[1] - gt_abs_p.t[1];
        const float cpu_abs_ez = cpu_abs_p.t[2] - gt_abs_p.t[2];
        const float cpu_abs_t_err =
          std::sqrt(cpu_abs_ex * cpu_abs_ex + cpu_abs_ey * cpu_abs_ey + cpu_abs_ez * cpu_abs_ez);
        const float cpu_abs_r_err = rotation_angle_error_rad(cpu_abs_p.R, gt_abs_p.R);

        float gpu_tx = std::numeric_limits<float>::quiet_NaN();
        float gpu_ty = std::numeric_limits<float>::quiet_NaN();
        float gpu_tz = std::numeric_limits<float>::quiet_NaN();
        float gpu_t_err = std::numeric_limits<float>::quiet_NaN();
        float gpu_r_err = std::numeric_limits<float>::quiet_NaN();
        float gpu_abs_tx = std::numeric_limits<float>::quiet_NaN();
        float gpu_abs_ty = std::numeric_limits<float>::quiet_NaN();
        float gpu_abs_tz = std::numeric_limits<float>::quiet_NaN();
        float gpu_abs_t_err = std::numeric_limits<float>::quiet_NaN();
        float gpu_abs_r_err = std::numeric_limits<float>::quiet_NaN();
        if (gpu_ok) {
          const Pose & gpu_abs_p = gpu_abs[static_cast<std::size_t>(b)];
          gpu_tx = t_gpu[static_cast<std::size_t>(b) * 3];
          gpu_ty = t_gpu[static_cast<std::size_t>(b) * 3 + 1];
          gpu_tz = t_gpu[static_cast<std::size_t>(b) * 3 + 2];
          const float gx = gpu_tx - gt.t[0];
          const float gy = gpu_ty - gt.t[1];
          const float gz = gpu_tz - gt.t[2];
          gpu_t_err = std::sqrt(gx * gx + gy * gy + gz * gz);
          gpu_r_err = rotation_angle_error_rad(&R_gpu[static_cast<std::size_t>(b) * 9], gt.R);
          gpu_abs_tx = gpu_abs_p.t[0];
          gpu_abs_ty = gpu_abs_p.t[1];
          gpu_abs_tz = gpu_abs_p.t[2];
          const float gax = gpu_abs_tx - gt_abs_p.t[0];
          const float gay = gpu_abs_ty - gt_abs_p.t[1];
          const float gaz = gpu_abs_tz - gt_abs_p.t[2];
          gpu_abs_t_err = std::sqrt(gax * gax + gay * gay + gaz * gaz);
          gpu_abs_r_err = rotation_angle_error_rad(gpu_abs_p.R, gt_abs_p.R);
        }

        csv << b << ","
            << pair_t0[static_cast<std::size_t>(b)] << ","
            << pair_t1[static_cast<std::size_t>(b)] << ","
            << gt_t0[static_cast<std::size_t>(b)] << ","
            << gt_t1[static_cast<std::size_t>(b)] << ","
            << gt.t[0] << "," << gt.t[1] << "," << gt.t[2] << "," << 0.0F << ","
            << gt_abs_p.t[0] << "," << gt_abs_p.t[1] << "," << gt_abs_p.t[2] << ","
            << cpu_abs_p.t[0] << "," << cpu_abs_p.t[1] << "," << cpu_abs_p.t[2] << ","
            << cpu_abs_t_err << "," << cpu_abs_r_err << ","
            << ceres_tx << "," << ceres_ty << "," << ceres_tz << "," << ceres_t_err << "," << ceres_r_err << ","
            << ceres_abs_p.t[0] << "," << ceres_abs_p.t[1] << "," << ceres_abs_p.t[2] << ","
            << ceres_abs_t_err << "," << ceres_abs_r_err << ","
            << gpu_tx << "," << gpu_ty << "," << gpu_tz << "," << gpu_t_err << "," << gpu_r_err << ","
            << gpu_abs_tx << "," << gpu_abs_ty << "," << gpu_abs_tz << "," << gpu_abs_t_err << "," << gpu_abs_r_err << ","
            << (gpu_ok ? 1 : 0) << "\n";
      }
    }
    std::cout << "Saved per-pair comparison CSV: " << csv_path << "\n";

    if (gpu_start != nullptr) {
      cudaEventDestroy(gpu_start);
    }
    if (gpu_stop != nullptr) {
      cudaEventDestroy(gpu_stop);
    }
    if (d_src != nullptr) {
      cudaFree(d_src);
    }
    if (d_tgt != nullptr) {
      cudaFree(d_tgt);
    }
    if (d_R != nullptr) {
      cudaFree(d_R);
    }
    if (d_t != nullptr) {
      cudaFree(d_t);
    }
    if (d_H != nullptr) {
      cudaFree(d_H);
    }
    if (d_g != nullptr) {
      cudaFree(d_g);
    }
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
