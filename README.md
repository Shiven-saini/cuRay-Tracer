# CUDA Ray Tracing Engine

**Author:** Shiven Saini  
**Email:** shiven.career@proton.me  

A real-time CUDA-accelerated ray tracer built with OpenGL for interactive 3D rendering.

## Features

- **CUDA-accelerated ray tracing** for real-time performance
- **Interactive camera control** with mouse navigation
- **Dynamic light source control** using keyboard input
- **Real-time rendering** with OpenGL display
- **Cross-platform support** (Linux, Windows, macOS)

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (version 11.0 or higher)
- **CMake** (version 3.18 or higher)
- **OpenGL** development libraries
- **GLFW3** for window management
- **GLEW** for OpenGL extension loading

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install cmake build-essential
sudo apt install libglfw3-dev libglew-dev libgl1-mesa-dev
```

### Installing CUDA Toolkit

Download and install the CUDA Toolkit from [NVIDIA's official website](https://developer.nvidia.com/cuda-downloads).

## Building the Project

### Using Build Scripts (Recommended)

Make the scripts executable:
```bash
chmod +x clean.sh build.sh run.sh build_and_run.sh
```

**Clean build directory:**
```bash
./clean.sh
```

**Build the project:**
```bash
./build.sh
```

**Run the executable:**
```bash
./run.sh
```

**Build and run in one command:**
```bash
./build_and_run.sh
```

### Manual Build

1. **Create build directory:**
   ```bash
   mkdir build && cd build
   ```

2. **Configure with CMake:**
   ```bash
   cmake ..
   ```

3. **Build the project:**
   ```bash
   make -j$(nproc)
   ```

4. **Run the executable:**
   ```bash
   ./cuRayTracer
   ```

## Controls

### Camera Control (Mouse)
- **Mouse Movement**: Look around (first-person camera)
- **Mouse Scroll**: Zoom in/out
- **Left Click + Drag**: Rotate camera view
- **Right Click + Drag**: Pan camera

### Light Source Control (Keyboard)
- **W**: Move light forward
- **S**: Move light backward
- **A**: Move light left
- **D**: Move light right
- **Q**: Move light up
- **E**: Move light down

### General Controls
- **ESC**: Exit application
- **R**: Reset camera and light positions
- **F**: Toggle fullscreen mode
- **Space**: Pause/resume animation

## Project Structure

```
cuRay-Tracer/
├── src/                    # Source code
│   ├── main.cpp           # Main application entry point
│   ├── renderer.cu        # CUDA ray tracing kernels
│   ├── camera.cpp         # Camera management
│   └── scene.cpp          # Scene setup and management
├── include/               # Header files
├── shaders/              # OpenGL shaders
├── CMakeLists.txt        # CMake configuration
├── clean.sh              # Clean build directory
├── build.sh              # Build project
├── run.sh                # Run executable
├── build_and_run.sh      # Build and run combined
└── README.md             # This file
```

## Performance Tips

- Ensure your NVIDIA GPU drivers are up to date
- For best performance, use a GPU with compute capability 6.0 or higher
- Adjust ray tracing parameters in the source code for optimal performance/quality balance
- Use Release build configuration for maximum performance:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release ..
  ```

## Troubleshooting

### Common Issues

**CUDA not found:**
- Ensure CUDA Toolkit is properly installed
- Add CUDA to your PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- Add CUDA libraries to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

**OpenGL libraries not found:**
- Install development packages: `sudo apt install libgl1-mesa-dev libglu1-mesa-dev`

**Build fails with compute capability errors:**
- Check your GPU's compute capability and update CMakeLists.txt accordingly
- Modern GPUs typically support compute capability 6.0 or higher

**Runtime performance issues:**
- Verify GPU is being used: `nvidia-smi` while running
- Reduce ray tracing resolution or depth for better performance
- Ensure adequate GPU memory is available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA CUDA documentation and samples
- John Hopkin's GPU Programming specialization materials
- OpenGL and GLFW communities
- Ray tracing algorithm references and tutorials