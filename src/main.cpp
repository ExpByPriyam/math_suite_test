#include "Matrix.hpp"
#include "ArenaAllocator.hpp"
#include "DynamicMatrix.hpp"
#include "BVH.hpp"
#include <chrono>
#include <random>
#include <vector>

void run_matrix_benchmark() {
    using namespace MathSuite;
    std::cout << "\n--- BENCHMARK 1: 4x4 MATRIX MULTIPLICATION (SIMD VS LOOP) ---\n";

    Matrix4x4 A4{
        1.5f, 0.5f, 2.0f, 4.0f,
        0.0f, 2.2f, 1.1f, 3.5f,
        3.0f, 0.0f, 1.0f, 0.5f,
        0.0f, 1.0f, 0.2f, 2.0f
    };
    Matrix4x4 B4{
        2.0f, 0.0f, 1.0f, 0.0f,
        0.5f, 1.0f, 0.0f, 3.0f,
        0.0f, 2.5f, 1.5f, 1.0f,
        1.0f, 0.0f, 2.0f, 4.0f
    };
    Matrix4x4 result4{};

    Matrix<float, 5, 5> A5{};
    Matrix<float, 5, 5> B5{};
    Matrix<float, 5, 5> result5{};
    
    for (std::size_t i = 0; i < 5; ++i) {
        A5(i, i) = 1.5f;
        B5(i, i) = 2.0f;
    }

    const std::size_t iterations = 10'000'000;
    std::cout << "Executing " << iterations << " iterations...\n";

    auto start_simd = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        asm volatile("" : : "g"(&A4), "g"(&B4) : "memory");
        result4 = A4 * B4;
    }
    auto end_simd = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration_cast<std::chrono::milliseconds>(end_simd - start_simd).count();

    auto start_standard = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        asm volatile("" : : "g"(&A5), "g"(&B5) : "memory");
        result5 = A5 * B5;
    }
    auto end_standard = std::chrono::high_resolution_clock::now();
    auto duration_standard = std::chrono::duration_cast<std::chrono::milliseconds>(end_standard - start_standard).count();

    std::cout << "-> 4x4 Matrix (Hardware SIMD Path):       " << duration_simd << " ms (" 
              << (static_cast<double>(duration_simd) * 1'000'000.0) / static_cast<double>(iterations) << " ns/op)\n";
    std::cout << "-> 5x5 Matrix (Standard Loop Fallback):  " << duration_standard << " ms (" 
              << (static_cast<double>(duration_standard) * 1'000'000.0) / static_cast<double>(iterations) << " ns/op)\n";
}

void run_allocator_benchmark() {
    using namespace MathSuite;
    std::cout << "\n--- BENCHMARK 2: CUSTOM ARENA VS HEAP SYSTEM ALLOCATIONS ---\n";

    const std::size_t alloc_iterations = 100'000;
    Arena memory_arena(1024 * 1024); // 1MB

    // Profile Arena Allocator using DynamicMatrix
    auto start_arena = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < alloc_iterations; ++i) {
        DynamicMatrix<float> mat(16, 16, memory_arena);
        asm volatile("" : : "g"(&mat) : "memory");
        memory_arena.reset(); // Instantaneous O(1) frame reset
    }
    auto end_arena = std::chrono::high_resolution_clock::now();
    auto duration_arena = std::chrono::duration_cast<std::chrono::microseconds>(end_arena - start_arena).count();

    // Profile standard heap allocations
    auto start_heap = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < alloc_iterations; ++i) {
        std::vector<float> heap_mat(16 * 16, 0.0f);
        asm volatile("" : : "g"(heap_mat.data()) : "memory");
        // Implicit standard heap deallocation occurs here
    }
    auto end_heap = std::chrono::high_resolution_clock::now();
    auto duration_heap = std::chrono::duration_cast<std::chrono::microseconds>(end_heap - start_heap).count();

    std::cout << "Executed " << alloc_iterations << " dynamic matrix generations (16x16):\n";
    std::cout << "-> Custom Arena Allocator Path: " << duration_arena << " us\n";
    std::cout << "-> Standard Heap Allocator Path: " << duration_heap << " us\n";
}

void run_spatial_benchmark() {
    using namespace MathSuite;
    std::cout << "\n--- BENCHMARK 3: FLAT-ARRAY BVH TRAVERSAL LOGARITHMIC SEARCH ---\n";

    std::vector<Vector3> points;
    points.reserve(10000);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    for (std::size_t i = 0; i < 10000; ++i) {
        points.push_back(Vector3{dist(rng), dist(rng), dist(rng)});
    }

    auto start_build = std::chrono::high_resolution_clock::now();
    BVH bvh(points);
    auto end_build = std::chrono::high_resolution_clock::now();
    auto duration_build = std::chrono::duration_cast<std::chrono::microseconds>(end_build - start_build).count();

    std::cout << "Generated 10,000 Spatial Points.\n";
    std::cout << "BVH Contiguous Architecture Generation: " << duration_build << " us\n";

    Ray search_ray(Vector3{0.0f, 0.0f, -100.0f}, Vector3{0.0f, 0.0f, 1.0f});
    std::size_t matched_index = 0;

    auto start_query = std::chrono::high_resolution_clock::now();
    bvh.intersect(search_ray, matched_index);
    auto end_query = std::chrono::high_resolution_clock::now();
    auto duration_query = std::chrono::duration_cast<std::chrono::nanoseconds>(end_query - start_query).count();

    std::cout << "-> Logarithmic Flat Tree Intersection Latency: " << duration_query << " ns\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "   LOW-LEVEL ENGINE SUITE PRODUCTION PROFILER     \n";
    std::cout << "==================================================\n";

    run_matrix_benchmark();
    run_allocator_benchmark();
    run_spatial_benchmark();

    std::cout << "\n==================================================\n";
    return 0;
}