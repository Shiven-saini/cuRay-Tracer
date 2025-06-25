# CUDA Ray Tracing Engine

**Author:** Shiven Saini  
**Email:** shiven.career@proton.me  
**Date:** June 2025

## Overview

A high-performance real-time ray tracing engine built with CUDA and OpenGL, featuring GPU-accelerated rendering with advanced lighting effects including reflections, refractions, and shadows. The engine uses CUDA-OpenGL interoperability for optimal performance and supports real-time camera movement and interaction.

## Features

### Core Rendering Features
- **GPU-Accelerated Ray Tracing**: Utilizes NVIDIA CUDA for parallel ray computation
- **Real-time Performance**: Optimized for 720p resolution with 30+ FPS on modern GPUs
- **Advanced Materials**: Support for diffuse, reflective, and refractive materials
- **Complex Scene Rendering**: Room-like environment with multiple geometric primitives

### Visual Effects
- **Reflections**: High-quality mirror-like reflections with configurable reflectivity
- **Refractions**: Glass-like materials with realistic light bending
- **Shadows**: Hard shadows with proper occlusion testing
- **Multi-bounce Lighting**: Configurable ray depth for global illumination
- **Anti-aliasing**: Multi-sample anti-aliasing for smooth edges
- **Tone Mapping**: HDR tone mapping with gamma correction

### Interactive Features
- **Real-time Camera Control**: WASD movement + mouse look
- **FPS Counter**: Real-time performance monitoring
- **Responsive Design**: Immediate visual feedback to camera movement

### Technical Features
- **CUDA-OpenGL Interop**: Seamless GPU memory sharing
- **Modern C++17**: Clean, maintainable codebase
- **CMake Build System**: Cross-platform build support
- **Modular Architecture**: Well-organized component-based design

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 3.0+
- Minimum 2GB VRAM recommended
- Modern multi-core CPU

### Software
- Linux (Ubuntu 18.04+ recommended)
- CUDA Toolkit 11.0+
- CMake 3.18+
- GCC 7.0+ with C++17 support

### Dependencies
- OpenGL 3.3+
- GLFW 3.3+
- GLEW 2.0+
- CUDA Runtime & Libraries

## Installation

### 1. Install CUDA Toolkit
```bash
# Download from NVIDIA Developer website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. Install Dependencies
```bash
sudo apt update
sudo apt install -y build-essential cmake
sudo apt install -y libglfw3-dev libglew-dev libgl1-mesa-dev
```

### 3. Build the Project
```bash
git clone <repository-url>
cd CudaRayTracer
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 4. Run the Engine
```bash
./CudaRayTracer
```

## Controls

| Control | Action |
|---------|--------|
| `W` | Move forward |
| `S` | Move backward |
| `A` | Move left |
| `D` | Move right |
| `Space` | Move up |
| `Shift` | Move down |
| `Mouse` | Look around |
| `ESC` | Exit application |

## Project Structure

```
CudaRayTracer/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/                    # Header files
│   ├── math/                   # Mathematical utilities
│   │   ├── Vec3.h             # 3D vector class
│   │   └── Ray.h              # Ray class
│   ├── primitives/            # Geometric primitives
│   │   ├── Sphere.h           # Sphere primitive
│   │   ├── Plane.h            # Plane primitive
│   │   └── Material.h         # Material definitions
│   ├── cuda/                  # CUDA-specific headers
│   │   ├── raytracing_kernel.h # Kernel declarations
│   │   └── cuda_utils.h       # CUDA utilities
│   ├── Window.h               # Window management
│   ├── Camera.h               # Camera system
│   ├── Scene.h                # Scene management
│   ├── RayTracer.h            # Main ray tracer
│   └── FPSCounter.h           # Performance monitoring
└── src/                       # Source files
    ├── main.cpp               # Application entry point
    ├── Window.cpp             # Window implementation
    ├── Camera.cpp             # Camera implementation
    ├── Scene.cpp              # Scene setup
    ├── RayTracer.cpp          # Ray tracer implementation
    ├── FPSCounter.cpp         # FPS counter implementation
    └── cuda/                  # CUDA implementations
        ├── raytracing_kernel.cu # Ray tracing kernels
        └── cuda_utils.cu      # CUDA utility functions
```

## Architecture

### Core Components

1. **Window Management**: GLFW-based window creation and input handling
2. **Camera System**: First-person camera with smooth movement and mouse look
3. **Scene Management**: Hierarchical scene representation with materials
4. **Ray Tracer**: CUDA-accelerated ray tracing with OpenGL interop
5. **Performance Monitoring**: Real-time FPS and timing information

### Rendering Pipeline

1. **Scene Setup**: Initialize geometric primitives and materials
2. **Camera Update**: Process input and update camera matrices
3. **CUDA Kernel Launch**: Generate rays and trace through scene
4. **Lighting Calculation**: Compute direct and indirect lighting
5. **Material Interaction**: Handle reflections, refractions, and shadows
6. **Post-processing**: Apply tone mapping and gamma correction
7. **Display**: Present final image via OpenGL texture

### Memory Management

- **GPU Memory**: Efficient allocation and deallocation
- **CUDA-OpenGL Interop**: Zero-copy data sharing
- **Resource Management**: RAII-based cleanup and error handling

## Performance Optimization

### CUDA Optimizations
- **Coalesced Memory Access**: Optimized memory access patterns
- **Shared Memory Usage**: Minimize global memory bandwidth
- **Occupancy Optimization**: Balanced thread block sizes
- **Fast Math**: Enabled for performance-critical operations

### Algorithmic Optimizations
- **Early Ray Termination**: Stop tracing when contribution is minimal
- **Adaptive Sampling**: Variable sample counts based on scene complexity
- **Efficient Intersection Testing**: Optimized ray-primitive intersections

## Troubleshooting

### Common Issues

**CUDA Device Not Found**
```bash
nvidia-smi  # Check if NVIDIA driver is installed
nvcc --version  # Check CUDA installation
```

**Build Errors**
```bash
# Ensure all dependencies are installed
sudo apt install build-essential cmake libglfw3-dev libglew-dev
```

**Low Performance**
- Check GPU memory usage: `nvidia-smi`
- Verify CUDA compute capability: Must be 3.0+
- Monitor thermal throttling

### Debug Mode
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./CudaRayTracer
```

## Future Enhancements

- [ ] Denoising algorithms for real-time quality
- [ ] Volumetric lighting and fog effects
- [ ] Temporal anti-aliasing (TAA)
- [ ] Scene loading from external files
- [ ] Material editor interface
- [ ] Multi-GPU support
- [ ] RTX hardware acceleration integration

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## Contact

**Shiven Saini**  
Email: shiven.career@proton.me  

For questions, suggestions, or collaboration opportunities, please don't hesitate to reach out!

---

*Built with passion for real-time graphics and GPU computing.*