#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace gato_cuda_learning_path
{

inline void throw_on_cuda_error(cudaError_t status, const char * context)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

class ScopedWallTimer
{
public:
  ScopedWallTimer()
  : start_(std::chrono::steady_clock::now())
  {}

  double elapsed_ms() const
  {
    return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start_).count();
  }

private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace gato_cuda_learning_path
