#pragma once

#include "ArenaAllocator.hpp"
#include <vector>
#include <stdexcept>

namespace MathSuite {

template <typename T>
class DynamicMatrix {
private:
    std::size_t m_rows;
    std::size_t m_cols;
    // Uses your custom arena allocator instead of standard heap allocations
    std::vector<T, ArenaAllocator<T>> m_data;

public:
    DynamicMatrix(std::size_t rows, std::size_t cols, Arena& arena)
        : m_rows(rows), m_cols(cols), m_data(ArenaAllocator<T>(arena)) {
        m_data.resize(rows * cols, static_cast<T>(0));
    }

    T& operator()(std::size_t r, std::size_t c) {
        return m_data[r * m_cols + c];
    }

    const T& operator()(std::size_t r, std::size_t c) const {
        return m_data[r * m_cols + c];
    }

    std::size_t rows() const noexcept { return m_rows; }
    std::size_t cols() const noexcept { return m_cols; }
    
    // Smooth O(N^3) cache-friendly i-k-j multiplication fallback for dynamic sizes
    DynamicMatrix operator*(const DynamicMatrix& other) const {
        if (m_cols != other.m_rows) {
            throw std::invalid_argument("Inner matrix dimensions must match for multiplication.");
        }
        
        // Share the same underlying arena allocator allocation block
        DynamicMatrix result(m_rows, other.m_cols, m_data.get_allocator().m_arena);
        
        for (std::size_t i = 0; i < m_rows; ++i) {
            for (std::size_t k = 0; k < m_cols; ++k) {
                T temp = (*this)(i, k);
                for (std::size_t j = 0; j < other.m_cols; ++j) {
                    result(i, j) += temp * other(k, j);
                }
            }
        }
        return result;
    }
};

} // namespace MathSuite