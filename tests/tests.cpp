#include "Matrix.hpp"
#include "AABB.hpp"
#include <cassert>
#include <iostream>

void test_matrix_properties() {
    using namespace MathSuite;
    
    // Matrix identity multiplication stability test
    Matrix4x4 A{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    Matrix4x4 identity{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    
    auto result = A * identity;
    assert(std::abs(result(0, 0) - 1.0f) < 1e-5f);
    assert(std::abs(result(3, 3) - 16.0f) < 1e-5f);
    std::cout << "[PASS] Matrix Identity Multiplication Verified.\n";
}

void test_normalization_bounds() {
    using namespace MathSuite;
    Vector3 zero_vector{0.f, 0.f, 0.f};
    
    bool exception_thrown = false;
    try {
        auto fail = zero_vector.normalized();
    } catch (const std::runtime_error&) {
        exception_thrown = true;
    }
    
    assert(exception_thrown && "Engine must throw an explicit error on near-zero division boundaries.");
    std::cout << "[PASS] Vector Normalization Edge Case Guard Verified.\n";
}

int main() {
    std::cout << "Executing Automated Integrity Check Suite...\n";
    test_matrix_properties();
    test_normalization_bounds();
    std::cout << "All core system modules stable.\n";
    return 0;
}