#pragma once

#include <cstddef>
#include <iostream>
#include <array>
#include <type_traits>
#include <stdexcept>
#include <cmath>

// Intrinsic headers for hardware vectorization
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

namespace MathSuite {

template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix {
    static_assert(std::is_floating_point_v<T>, "Matrix underlying type must be a floating-point type.");
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be greater than zero.");

public:
    using value_type = T;
    static constexpr std::size_t row_count = Rows;
    static constexpr std::size_t col_count = Cols;
    static constexpr std::size_t total_elements = Rows * Cols;

private:
    // Flat 1D array enforcing row-major contiguous memory allocation.
    alignas(32) std::array<T, total_elements> m_data{};

public:
    // Constructors
    constexpr Matrix() = default;
    
    template<typename... Args>
    constexpr Matrix(Args... args) : m_data{static_cast<T>(args)...} {
        static_assert(sizeof...(args) == total_elements, "Argument count must match total matrix elements.");
    }

    // Accessors (Row-Major Indexing: Row * Cols + Col)
    constexpr T& operator()(std::size_t r, std::size_t c) {
        return m_data[r * Cols + c];
    }

    constexpr const T& operator()(std::size_t r, std::size_t c) const {
        return m_data[r * Cols + c];
    }

    // Raw data access for low-level memory operations and SIMD loading
    constexpr T* data() noexcept { return m_data.data(); }
    constexpr const T* data() const noexcept { return m_data.data(); }

    // --- Fundamental Arithmetic Operations ---

    // Element-wise Addition
    constexpr Matrix operator+(const Matrix& other) const noexcept {
        Matrix result = *this;
        for (std::size_t i = 0; i < total_elements; ++i) {
            result.m_data[i] += other.m_data[i];
        }
        return result;
    }

    // Element-wise Subtraction
    constexpr Matrix operator-(const Matrix& other) const noexcept {
        Matrix result = *this;
        for (std::size_t i = 0; i < total_elements; ++i) {
            result.m_data[i] -= other.m_data[i];
        }
        return result;
    }

    // Scalar Multiplication
    constexpr Matrix& operator*=(T scalar) noexcept {
        for (auto& val : m_data) {
            val *= scalar;
        }
        return *this;
    }

    constexpr Matrix operator*(T scalar) const noexcept {
        Matrix result = *this;
        result *= scalar;
        return result;
    }

    // Matrix-Matrix Multiplication (General Loop Fallback)
    template <std::size_t OtherCols>
    constexpr Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const {
        Matrix<T, Rows, OtherCols> result{};
        for (std::size_t i = 0; i < Rows; ++i) {
            for (std::size_t k = 0; k < Cols; ++k) {
                T temp = (*this)(i, k);
                for (std::size_t j = 0; j < OtherCols; ++j) {
                    result(i, j) += temp * other(k, j);
                }
            }
        }
        return result;
    }

    // --- Vector Specific Geometric Operations ---
    
    constexpr T dot(const Matrix& other) const noexcept {
        static_assert(Cols == 1, "Dot product is only defined for column vectors (N x 1).");
        T total = 0;
        for (std::size_t i = 0; i < Rows; ++i) {
            total += (*this)(i, 0) * other(i, 0);
        }
        return total;
    }

    constexpr T magnitude_squared() const noexcept {
        static_assert(Cols == 1, "Magnitude is only defined for column vectors.");
        return this->dot(*this);
    }

    T magnitude() const noexcept {
        static_assert(Cols == 1, "Magnitude is only defined for column vectors.");
        return std::sqrt(magnitude_squared());
    }

    Matrix normalized() const {
        static_assert(Cols == 1, "Normalization is only defined for column vectors.");
        T mag = magnitude();
        if (mag < static_cast<T>(1e-6)) {
            throw std::runtime_error("Cannot normalize a near-zero vector.");
        }
        return (*this) * (static_cast<T>(1.0) / mag);
    }

    void print() const {
        for (std::size_t i = 0; i < Rows; ++i) {
            std::cout << "[ ";
            for (std::size_t j = 0; j < Cols; ++j) {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << "]\n";
        }
    }
};

template <typename T, std::size_t N>
using Vector = Matrix<T, N, 1>;

using Vector3 = Vector<float, 3>;
using Vector4 = Vector<float, 4>;
using Matrix4x4 = Matrix<float, 4, 4>;

template <typename T>
constexpr Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& u, const Matrix<T, 3, 1>& v) noexcept {
    return Matrix<T, 3, 1>{
        u(1, 0) * v(2, 0) - u(2, 0) * v(1, 0),
        u(2, 0) * v(0, 0) - u(0, 0) * v(2, 0),
        u(0, 0) * v(1, 0) - u(1, 0) * v(0, 0)
    };
}

// ============================================================================
// Explicit Template Specialization for Ultra-Fast 4x4 float Matrix Multiplication
// ============================================================================
template <>
template <>
inline Matrix<float, 4, 4> Matrix<float, 4, 4>::operator*<4>(const Matrix<float, 4, 4>& other) const {
    Matrix<float, 4, 4> result{};
    
    const float* a = this->data();
    const float* b = other.data();
    float* c = result.data();

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t b0 = vld1q_f32(b + 0);
    float32x4_t b1 = vld1q_f32(b + 4);
    float32x4_t b2 = vld1q_f32(b + 8);
    float32x4_t b3 = vld1q_f32(b + 12);

    for (std::size_t i = 0; i < 4; ++i) {
        float32x4_t a0 = vdupq_n_f32(a[i * 4 + 0]);
        float32x4_t a1 = vdupq_n_f32(a[i * 4 + 1]);
        float32x4_t a2 = vdupq_n_f32(a[i * 4 + 2]);
        float32x4_t a3 = vdupq_n_f32(a[i * 4 + 3]);

        float32x4_t r = vmulq_f32(a0, b0);
        r = vmlaq_f32(r, a1, b1);
        r = vmlaq_f32(r, a2, b2);
        r = vmlaq_f32(r, a3, b3);

        vst1q_f32(c + (i * 4), r);
    }
#elif defined(__x86_64__) || defined(_M_X64)
    for (std::size_t i = 0; i < 4; ++i) {
        __m256 r = _mm256_setzero_ps();
        for (std::size_t k = 0; k < 4; ++k) {
            __m256 a_val = _mm256_set1_ps(a[i * 4 + k]);
            __m256 b_val = _mm256_castps128_ps256(_mm_load_ps(b + (k * 4)));
            r = _mm256_fmadd_ps(a_val, b_val, r);
        }
        _mm_store_ps(c + (i * 4), _mm256_castps256_ps128(r));
    }
#else
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t k = 0; k < 4; ++k) {
            float temp = a[i * 4 + k];
            for (std::size_t j = 0; j < 4; ++j) {
                c[i * 4 + j] += temp * b[k * 4 + j];
            }
        }
    }
#endif

    return result;
}

} // namespace MathSuite