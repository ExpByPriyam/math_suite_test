#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>
#include <stdexcept>
#include <mutex>

namespace MathSuite {

class Arena {
private:
    std::byte* m_buffer = nullptr;
    std::size_t m_size = 0;
    std::size_t m_offset = 0;
    std::mutex m_mutex{}; // Ensure thread safety for parallel spatial partitioning loops

public:
    explicit Arena(std::size_t bytes) : m_size(bytes) {
        // Pre-allocate a large contiguous block of memory aligned to a 64-byte cache line boundary
        m_buffer = static_cast<std::byte*>(std::aligned_alloc(64, m_size));
        if (!m_buffer) {
            throw std::bad_alloc();
        }
    }

    ~Arena() {
        std::free(m_buffer);
    }

    // Disable copying to safeguard raw pointer ownership
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void* allocate(std::size_t bytes, std::size_t alignment = 64) {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Calculate the padding required to satisfy the requested alignment requirement
        std::uintptr_t current_address = reinterpret_cast<std::uintptr_t>(m_buffer + m_offset);
        std::size_t padding = (alignment - (current_address % alignment)) % alignment;

        if (m_offset + padding + bytes > m_size) {
            throw std::bad_alloc(); // Arena capacity exhausted
        }

        m_offset += padding;
        void* ptr = m_buffer + m_offset;
        m_offset += bytes;
        
        return ptr;
    }

    void reset() noexcept {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_offset = 0; // O(1) reclamation of all allocated memory blocks
    }

    std::size_t used_memory() const noexcept { return m_offset; }
    std::size_t capacity() const noexcept { return m_size; }
};

// C++ standard library compliant allocator wrapper
template <typename T>
class ArenaAllocator {
public:
    using value_type = T;

    Arena& m_arena;

    explicit ArenaAllocator(Arena& arena) noexcept : m_arena(arena) {}

    template <typename U>
    constexpr ArenaAllocator(const ArenaAllocator<U>& other) noexcept : m_arena(other.m_arena) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(m_arena.allocate(n * sizeof(T), alignof(T)));
    }

    void deallocate(T*, std::size_t) noexcept {
        // No-op: Memory lifecycle is handled entirely by the parent Arena container
    }

    template <typename U>
    bool operator==(const ArenaAllocator<U>& other) const noexcept {
        return &m_arena == &other.m_arena;
    }

    template <typename U>
    bool operator!=(const ArenaAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace MathSuite