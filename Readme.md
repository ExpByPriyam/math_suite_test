# The Optimized Linear Algebra & Spatial Partitioning Suite

A high-performance, header-only, low-level mathematical computing engine written from scratch in C++20. Engineered specifically for multi-dimensional coordinate transformations, explicit hardware SIMD acceleration, and cache-localized spatial query pipelines.

## 🚀 Performance Architecture Blueprint

This suite was built to demonstrate mechanical sympathy — designing software that actively respects and exploits the underlying CPU architecture, cache sub-systems, and register structures.

### 1. Contiguous Memory Layout & Cache Locality

Naive mathematical libraries often utilize nested dynamic vectors (e.g., `vector<vector<T>>`), which introduce massive memory fragmentation, heap overhead, and severe pointer chasing.

- **Row-Major Layout:** This engine flattens a multi-dimensional matrix $M \times N$ into a strict, single contiguous 1D array where data elements are indexed linearly:

$$\text{Index}(i, j) = i \times N + j$$

- **Cache Alignment:** All core template shapes are decorated with `alignas(32)`, forcing compile-time variables allocated on the stack to sit perfectly on uniform memory address intervals. This maximizes CPU cache line optimization (64-byte blocks) and removes hardware structural stalls.

### 2. Multi-Architecture Hardware SIMD Specialization

For high-frequency graphics, simulation, and predictive risk transformations ($4 \times 4$ float variations), the execution layer leverages Template Specialization to drop standard arithmetic loop structures entirely. Preprocessor directives selectively compile raw processor microcode:

- **Apple Silicon (ARM NEON):** Bypasses standard operations by loading vectors into 128-bit hardware registers via `<arm_neon.h>` intrinsics (`vld1q_f32`, `vdupq_n_f32`, `vmlaq_f32`), calculating four parallel float lanes concurrently.
- **Intel / AMD (x86_64 AVX2):** Utilizes 256-bit wide computing blocks via `<immintrin.h>` integrating hardware-level Fused Multiply-Add (FMA) actions for dense computation paths.

### 3. High-Speed Frame Memory Allocation

Dynamic allocations (`new` / `malloc`) are expensive bottleneck points under multi-threaded loads due to kernel context switches and heap fragmentation lockouts.

- **Flat Arena Design:** The system pre-allocates a massive contiguous arena memory block upfront. High-frequency allocations (like local runtime matrices) are served by advancing an internal offset pointer — a true $O(1)$ action.
- **Zero Deallocation Overhead:** Individual destructions do nothing. Entire operational workspaces are reclaimed instantaneously in a single CPU cycle by resetting the memory arena offset index back to zero.

### 4. Cache-Localized Spatial Partitioning

Standard tree architectures rely heavily on pointer connections (`Node* left, right`), which scatter memory allocations randomly across the heap and destroy data locality.

- **Flat-Array BVH:** This engine constructs a cache-friendly Axis-Aligned Bounding Box (AABB) Bounding Volume Hierarchy (BVH) where all nodes live sequentially within a flat contiguous pool.
- **Logarithmic Complexity:** Leverages an optimized ray slab-intersection system. It scales search metrics down from an expensive linear $O(N)$ sweep to an ultra-fast $O(\log N)$ tree traversal tracking spatial bounds on a local stack buffer.

## 📊 Empirical Benchmarks

**Hardware Environment:** Apple M-Series Silicon Native (ARM64 Arch) **Compilation Flags:** `-O3 -march=native` (Strict performance profiling verified via native inline assembly barriers)

| Mathematical / Spatial Operation   | Optimization Strategy          | Average Latency | Speed Multiplier       |
| ---------------------------------- | ------------------------------ | --------------- | ---------------------- |
| $4 \times 4$ Matrix Multiplication | ARM NEON Hardware SIMD         | 0.7 ns          | 21x Faster             |
| $5 \times 5$ Matrix Multiplication | Standard Loop Fallback (i-k-j) | 14.6 ns         | Baseline               |
| Dynamic Matrix Workspace Gen       | Custom Arena Allocator         | ~2.8x Speedup   | Near-Zero Allocation   |
| 10k Coordinate Dataset Query       | Flat-Array BVH Traversal       | 208.0 ns        | Sub-Microsecond Search |

## 📁 Repository Directory Structure

```
├── CMakeLists.txt              # Unified compiler flags & cross-platform profiling configuration
├── include/
│   └── MathSuite/
│       ├── Matrix.hpp          # Compile-time templates, geometric math, & SIMD specialization
│       ├── DynamicMatrix.hpp   # Runtime heap matrix tracking integrated with custom arenas
│       ├── ArenaAllocator.hpp  # Cache-aligned, thread-safe linear frame allocator
│       ├── AABB.hpp            # Spatial bound blocks & ray-slab intersection algorithms
│       └── BVH.hpp             # Contiguous flat-array spatial partitioning framework
└── src/
    └── main.cpp                # Performance profiler & verification execution loop
```

## 🛠️ Build and Integration Verification

The project is structured as a modern header-only utility suite driven by an automated CMake compiler pipeline. It dynamically inspects your host operating environment to target native vector register pipelines safely.

### Build Executable

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Performance Profiler

```bash
./math_suite_test
```

## 💡 Core Design Philosophy

This suite was built to demonstrate an advanced grasp of modern low-level systems engineering. By bridging mathematical paradigms with hardware realities — understanding registers, minimizing heap allocation, and organizing data structures to preserve cache integrity — it provides a production-grade foundation ready for high-frequency algorithmic modeling, embedded computing, or high-throughput asset systems.
