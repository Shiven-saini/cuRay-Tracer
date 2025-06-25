cmake_minimum_required(VERSION 3.18)
project(CudaRayTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)
enable_language(CUDA)

# Include directories
include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Source files
set(SOURCES
    src/main.cpp
    src/Window.cpp
    src/Camera.cpp
    src/Scene.cpp
    src/RayTracer.cpp
    src/FPSCounter.cpp
    src/cuda/raytracing_kernel.cu
    src/cuda/cuda_utils.cu
)

# Header files
set(HEADERS
    include/Window.h
    include/Camera.h
    include/Scene.h
    include/RayTracer.h
    include/FPSCounter.h
    include/cuda/raytracing_kernel.h
    include/cuda/cuda_utils.h
    include/math/Vec3.h
    include/math/Ray.h
    include/primitives/Sphere.h
    include/primitives/Plane.h
    include/primitives/Material.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Set CUDA properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    OpenGL::GL
    glfw
    GLEW::GLEW
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Compiler-specific options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --ptxas-options=-v
        -Wno-deprecated-gpu-targets
        --expt-relaxed-constexpr
    >
)

# Debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
endif()