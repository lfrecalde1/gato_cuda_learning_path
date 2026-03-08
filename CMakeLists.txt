cmake_minimum_required(VERSION 3.18)
project(gato_cuda_learning_path LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 86 75)
endif()

find_package(ament_cmake REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Ubuntu libceres-dev installs config files in /usr/lib/cmake/Ceres.
if(NOT Ceres_DIR AND EXISTS "/usr/lib/cmake/Ceres/CeresConfig.cmake")
  set(Ceres_DIR "/usr/lib/cmake/Ceres" CACHE PATH "Path to CeresConfig.cmake")
endif()
find_package(Ceres QUIET)

set(LESSON_TARGETS
  lesson01_parallel_threads
  lesson02_memory_flow
  lesson03_parallel_reduction
  lesson04_cublas_gemm
  lesson05_custom_linear_algebra
  lesson06_optimizer_step
  lesson07_batched_optimizer_skeleton
  lesson08_matrix_basics
  lesson09_matrix_vector_ops
  lesson10_matrix_mul_cpu_vs_gpu
  lesson11_matmul_all_comparison
  lesson12_cublas_control_optimization_examples
  lesson13_dense_linear_systems_ic_cpu_vs_gpu
  lesson14_quadrotor_nonlinear_batch_cpu_vs_gpu
  lesson15_icp_pose_estimation_custom_solver_cpu_vs_gpu
  lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu
)

foreach(target IN LISTS LESSON_TARGETS)
  add_executable(${target} src/${target}.cu)
  target_include_directories(${target} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${target} CUDA::cudart)
  set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

target_link_libraries(lesson04_cublas_gemm CUDA::cublas)

target_link_libraries(lesson05_custom_linear_algebra CUDA::cublas)
target_link_libraries(lesson11_matmul_all_comparison CUDA::cublas)
target_link_libraries(lesson12_cublas_control_optimization_examples CUDA::cublas)
target_link_libraries(lesson13_dense_linear_systems_ic_cpu_vs_gpu CUDA::cublas)
target_link_libraries(lesson14_quadrotor_nonlinear_batch_cpu_vs_gpu CUDA::cublas)

if(Ceres_FOUND)
  target_link_libraries(lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu Ceres::ceres)
  target_compile_definitions(lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu PRIVATE GCLP_HAVE_CERES=1)
else()
  target_compile_definitions(lesson16_icp_pose_estimation_ceres_cpu_vs_custom_cpu_vs_gpu PRIVATE GCLP_HAVE_CERES=0)
  message(WARNING "Ceres not found: lesson16 will run custom CPU/GPU methods and report Ceres as unavailable.")
endif()

install(TARGETS ${LESSON_TARGETS}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)

ament_package()
