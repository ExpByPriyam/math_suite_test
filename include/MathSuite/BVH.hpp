#pragma once

#include "AABB.hpp"
#include <vector>

namespace MathSuite {

struct BVHNode {
    AABB bounding_box;
    std::size_t left_first = 0; // If primitive_count > 0, this indices first item; else indices left child node
    std::uint32_t primitive_count = 0;
};

class BVH {
private:
    std::vector<BVHNode> m_nodes;
    std::vector<Vector3> m_centroids;
    std::vector<std::size_t> m_primitive_indices;
    const std::vector<Vector3>& m_primitives;

    void update_node_bounds(std::size_t node_idx) noexcept {
        BVHNode& node = m_nodes[node_idx];
        node.bounding_box = AABB();
        for (std::size_t i = 0; i < node.primitive_count; ++i) {
            std::size_t prim_idx = m_primitive_indices[node.left_first + i];
            node.bounding_box.grow(m_primitives[prim_idx]);
        }
    }

    void subdivide(std::size_t node_idx) {
        BVHNode& node = m_nodes[node_idx];
        if (node.primitive_count <= 2) return; // Leaf criteria met

        // Split along the longest axis for simple balanced distributions
        std::size_t axis = 0;
        float width = node.bounding_box.max_bounds(0,0) - node.bounding_box.min_bounds(0,0);
        for(std::size_t i = 1; i < 3; ++i) {
            float w = node.bounding_box.max_bounds(i,0) - node.bounding_box.min_bounds(i,0);
            if (w > width) { axis = i; width = w; }
        }

        float split_pos = node.bounding_box.min_bounds(axis,0) + (width * 0.5f);

        // Standard in-place partitioning sequence
        std::int64_t i = static_cast<std::int64_t>(node.left_first);
        std::int64_t j = i + node.primitive_count - 1;
        while (i <= j) {
            if (m_centroids[m_primitive_indices[static_cast<std::size_t>(i)]](axis, 0) < split_pos) {
                i++;
            } else {
                std::swap(m_primitive_indices[static_cast<std::size_t>(i)], m_primitive_indices[static_cast<std::size_t>(j)]);
                j--;
            }
        }

        std::size_t left_count = static_cast<std::size_t>(i) - node.left_first;
        if (left_count == 0 || left_count == node.primitive_count) return;

        // Allocate children contiguously within our flat pool array
        std::size_t left_child_idx = m_nodes.size();
        m_nodes.emplace_back();
        m_nodes.emplace_back();

        m_nodes[left_child_idx].left_first = node.left_first;
        m_nodes[left_child_idx].primitive_count = static_cast<std::uint32_t>(left_count);

        m_nodes[left_child_idx + 1].left_first = static_cast<std::size_t>(i);
        m_nodes[left_child_idx + 1].primitive_count = static_cast<std::uint32_t>(node.primitive_count - left_count);

        node.left_first = left_child_idx;
        node.primitive_count = 0; // Marks node explicitly as an inner node

        update_node_bounds(left_child_idx);
        update_node_bounds(left_child_idx + 1);

        subdivide(left_child_idx);
        subdivide(left_child_idx + 1);
    }

public:
    explicit BVH(const std::vector<Vector3>& primitives) : m_primitives(primitives) {
        std::size_t count = primitives.size();
        m_primitive_indices.resize(count);
        m_centroids.resize(count);

        for (std::size_t i = 0; i < count; ++i) {
            m_primitive_indices[i] = i;
            m_centroids[i] = primitives[i]; // For points, the point itself is the centroid
        }

        // Allocate root node pool estimation slot
        m_nodes.reserve(count * 2);
        m_nodes.emplace_back();

        m_nodes[0].left_first = 0;
        m_nodes[0].primitive_count = static_cast<std::uint32_t>(count);

        update_node_bounds(0);
        subdivide(0);
    }

    // Traverse the flat pool to see if our ray hits a primitive in O(log N)
    bool intersect(const Ray& ray, std::size_t& hit_primitive_idx) const noexcept {
        float t_box = 0.0f;
        if (!m_nodes[0].bounding_box.intersects(ray, t_box)) return false;

        // Fast stack allocation for index traversal depth tracking
        std::size_t stack[64];
        std::int32_t stack_ptr = 0;
        stack[stack_ptr++] = 0;

        float closest_t = std::numeric_limits<float>::infinity();
        bool found_hit = false;

        while (stack_ptr > 0) {
            std::size_t node_idx = stack[--stack_ptr];
            const BVHNode& node = m_nodes[node_idx];

            if (node.primitive_count > 0) {
                // Leaf Node: Evaluate intersection against the primitives directly
                for (std::size_t i = 0; i < node.primitive_count; ++i) {
                    std::size_t prim_idx = m_primitive_indices[node.left_first + i];
                    // Simple distance approximation for point testing
                    float dist = (m_primitives[prim_idx] - ray.origin).magnitude_squared();
                    if (dist < closest_t) {
                        closest_t = dist;
                        hit_primitive_idx = prim_idx;
                        found_hit = true;
                    }
                }
            } else {
                // Inner Node: Evaluate children boxes and push intersections to stack
                float t_left = 0.f, t_right = 0.f;
                bool hit_left = m_nodes[node.left_first].bounding_box.intersects(ray, t_left);
                bool hit_right = m_nodes[node.left_first + 1].bounding_box.intersects(ray, t_right);

                if (hit_left && hit_right) {
                    // Cache optimization: Traverse the closest node first
                    if (t_left > t_right) {
                        stack[stack_ptr++] = node.left_first;
                        stack[stack_ptr++] = node.left_first + 1;
                    } else {
                        stack[stack_ptr++] = node.left_first + 1;
                        stack[stack_ptr++] = node.left_first;
                    }
                } else if (hit_left) {
                    stack[stack_ptr++] = node.left_first;
                } else if (hit_right) {
                    stack[stack_ptr++] = node.left_first + 1;
                }
            }
        }
        return found_hit;
    }

    std::size_t total_nodes() const noexcept { return m_nodes.size(); }
};

} // namespace MathSuite