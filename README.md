# GPU-Accelerated N-Body Gravity Simulator

A high-performance astrophysical simulation that leverages CUDA-OpenGL interop to simulate gravitational interactions between thousands of particles in real-time. This project demonstrates massive parallelism, memory hierarchy optimization, and zero-copy rendering techniques for Linux systems.

## 🚀 Features

* **Parallel Physics Engine:** $O(n^2)$ brute-force gravity calculation implemented in CUDA C++.
* **Shared Memory Tiling:** Optimized memory throughput by leveraging GPU Shared Memory to reduce Global Memory latency.
* **Zero-Copy Rendering:** Direct CUDA-OpenGL Interop using Vertex Buffer Objects (VBOs) for real-time visualization without CPU-GPU bottlenecks.
* **Real-time Control UI:** Integrated **Dear ImGui** dashboard to adjust time-steps and reset simulation parameters on the fly.
* **Velocity-based Coloring:** Custom GLSL Shaders to map particle kinetic energy to a dynamic color ramp.

## 🛠 Tech Stack

* **OS:** Linux (X11 or Wayland)
* **Language:** CUDA C++ (NVCC)
* **Graphics:** OpenGL 3.0+ (GLEW, GLFW)
* **GUI:** Dear ImGui
* **Build System:** Makefile

## 📦 Project Structure

* `src/`: Core CUDA logic and main application loop.
* `include/`: Header files for graphics and GUI backends.
* `external/`: Third-party libraries (ImGui).
* `build/`: Target directory for compiled binaries.

## ⚙️ Prerequisites

Ensure you have the NVIDIA drivers and CUDA Toolkit installed.

### Install Dependencies

**Arch Linux:**
`sudo pacman -S cuda glfw-wayland glew nvidia-prime`

**Ubuntu/Debian:**
`sudo apt-get install nvidia-cuda-toolkit libglew-dev libglfw3-dev nvidia-prime`

**Fedora:**
`sudo dnf install cuda-toolkit glfw-devel glew-devel`

## 🔨 Build & Run

1. **Clone the repository:**
`git clone https://github.com/Ricky-131/GPU-Accelerated-N-Body-Gravity-Simulator.git`
2. **Build:**
`make`
3. **Run:**
For systems with dedicated GPUs: `./build/nbody_viz`
For hybrid laptops (NVIDIA + Intel/AMD): `prime-run ./build/nbody_viz`

## 📈 Technical Implementation

The simulator uses **Shared Memory Tiling**. To avoid the "Memory Wall," the kernel loads tiles of particle data into the GPU's on-chip shared memory. This reduces Global Memory traffic by a factor proportional to the thread block size, significantly increasing GFLOPS performance.

---

Developed by [Ricky S](https://www.google.com/search?q=https://github.com/Ricky-131)

---
