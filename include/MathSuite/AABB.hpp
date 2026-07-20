#pragma once

#include "Matrix.hpp"
#include <algorithm>
#include <limits>

namespace MathSuite {

struct Ray {
    Vector3 origin;
    Vector3 direction; // Assumed to be normalized
    Vector3 inv_direction; // Precomputed 1.0f / direction to eliminate division in tight loops

    Ray(Vector3 o, Vector3 d) : origin(o), direction(d) {
        inv_direction = Vector3{
            1.0f / (std::abs(d(0, 0)) < 1e-6f ? 1e-6f : d(0, 0)),
            1.0f / (std::abs(d(1, 0)) < 1e-6f ? 1e-6f : d(1, 0)),
            1.0f / (std::abs(d(2, 0)) < 1e-6f ? 1e-6f : d(2, 0))
        };
    }
};

class AABB {
public:
    Vector3 min_bounds{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    Vector3 max_bounds{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

    AABB() = default;
    AABB(const Vector3& min, const Vector3& max) : min_bounds(min), max_bounds(max) {}

    // Expand the bounding box to encapsulate a physical point
    void grow(const Vector3& point) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            min_bounds(i, 0) = std::min(min_bounds(i, 0), point(i, 0));
            max_bounds(i, 0) = std::max(max_bounds(i, 0), point(i, 0));
        }
    }

    // Combine two AABB spaces together
    void merge(const AABB& other) noexcept {
        for (std::size_t i = 0; i < 3; ++i) {
            min_bounds(i, 0) = std::min(min_bounds(i, 0), other.min_bounds(i, 0));
            max_bounds(i, 0) = std::max(max_bounds(i, 0), other.max_bounds(i, 0));
        }
    }

    // Cache-friendly Slab Method Ray-AABB Intersection Check
    bool intersects(const Ray& ray, float& t_result) const noexcept {
        float t_min = -std::numeric_limits<float>::infinity();
        float t_max = std::numeric_limits<float>::infinity();

        for (std::size_t i = 0; i < 3; ++i) {
            float t1 = (min_bounds(i, 0) - ray.origin(i, 0)) * ray.inv_direction(i, 0);
            float t2 = (max_bounds(i, 0) - ray.origin(i, 0)) * ray.inv_direction(i, 0);

            float t_near_axis = std::min(t1, t2);
            float t_far_axis  = std::max(t1, t2);

            t_min = std::max(t_min, t_near_axis);
            t_max = std::min(t_max, t_far_axis);

            if (t_min > t_max) return false;
        }

        if (t_max < 0.0f) return false; // The box is behind the ray path
        
        // Return the closest intersection distance
        t_result = t_min < 0.0f ? t_max : t_min;
        return true;
    }
};

} // namespace MathSuite