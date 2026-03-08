# gato_cuda_learning_path

A progressive CUDA package to learn from zero to GATO-style optimizer implementation.

## Learning Sequence

1. `lesson01_parallel_threads`: CUDA thread/block indexing and parallel launch model.
2. `lesson02_memory_flow`: host/device memory flow and first elementwise kernel.
3. `lesson03_parallel_reduction`: block-level reduction for sums (merit/cost style primitive).
4. `lesson04_cublas_gemm`: dense linear algebra with cuBLAS.
5. `lesson05_custom_linear_algebra`: custom matrix kernel vs cuBLAS and when custom wins.
6. `lesson06_optimizer_step`: projected gradient + merit accumulation.
7. `lesson07_batched_optimizer_skeleton`: batch-wise rollout/cost skeleton like GATO batching.
8. `lesson08_matrix_basics`: matrix variables and elementwise operations (`+`, `-`, Hadamard product).
9. `lesson09_matrix_vector_ops`: matrix-vector multiply and matrix addition on GPU.
10. `lesson10_matrix_mul_cpu_vs_gpu`: naive dense matrix multiply benchmark and speedup view.
11. `lesson11_matmul_all_comparison`: compare CPU naive, GPU naive kernel, GPU tiled kernel, and cuBLAS GEMM.
12. `lesson12_cublas_control_optimization_examples`: cuBLAS examples for control dynamics and optimization gradients.
13. `lesson13_dense_linear_systems_ic_cpu_vs_gpu`: dense linear system simulation over many initial conditions, comparing CPU and GPU/cuBLAS execution.
14. `lesson14_quadrotor_nonlinear_batch_cpu_vs_gpu`: batched nonlinear quadrotor simulation (`10 s`, `dt=0.01`) with CUDA + cuBLAS vs CPU.
15. `lesson15_icp_pose_estimation_custom_solver_cpu_vs_gpu`: batched point-to-point ICP pose estimation with a custom Gauss-Newton solver (no Ceres), comparing CPU vs GPU.
16. `lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu`: same point-cloud pose problem solved with Ceres CPU, custom CPU Gauss-Newton, and custom GPU Gauss-Newton.

## Build

```bash
colcon build --packages-select gato_cuda_learning_path --symlink-install
source install/setup.bash
```

## Run Lessons

```bash
ros2 run gato_cuda_learning_path lesson01_parallel_threads
ros2 run gato_cuda_learning_path lesson02_memory_flow
ros2 run gato_cuda_learning_path lesson03_parallel_reduction
ros2 run gato_cuda_learning_path lesson04_cublas_gemm
ros2 run gato_cuda_learning_path lesson05_custom_linear_algebra
ros2 run gato_cuda_learning_path lesson06_optimizer_step
ros2 run gato_cuda_learning_path lesson07_batched_optimizer_skeleton
ros2 run gato_cuda_learning_path lesson08_matrix_basics
ros2 run gato_cuda_learning_path lesson09_matrix_vector_ops
ros2 run gato_cuda_learning_path lesson10_matrix_mul_cpu_vs_gpu
ros2 run gato_cuda_learning_path lesson11_matmul_all_comparison
ros2 run gato_cuda_learning_path lesson12_cublas_control_optimization_examples
ros2 run gato_cuda_learning_path lesson13_dense_linear_systems_ic_cpu_vs_gpu
ros2 run gato_cuda_learning_path lesson14_quadrotor_nonlinear_batch_cpu_vs_gpu
ros2 run gato_cuda_learning_path lesson15_icp_pose_estimation_custom_solver_cpu_vs_gpu
ros2 run gato_cuda_learning_path lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu
```

## How this maps to GATO concepts

1. Parallel model and memory discipline: base required for every GATO kernel.
2. Reduction and matrix operations: used in KKT/merit/line-search components.
3. cuBLAS: fast dense baselines and validation reference.
4. Custom linear algebra: needed for structure-aware operators (block tridiagonal, Schur, PCG), where generic dense kernels are suboptimal.
5. Batched solve pattern: mirrors the optimizer batch execution strategy.
6. Matrix fundamentals path: builds intuition from elementwise ops to dense multiply acceleration.
7. Practical cuBLAS path: demonstrates control model propagation and optimization gradient loops.

## Your next implementation milestones

1. Add a custom batched block-tridiagonal matrix-vector product.
2. Add a PCG loop with fixed iteration budget.
3. Add line-search kernel and merit comparison.
4. Add deterministic timing instrumentation and no-allocation runtime loop.
5. Integrate as a ROS 2 real-time node once kernel primitives are stable.
# gato_cuda_learning_path
