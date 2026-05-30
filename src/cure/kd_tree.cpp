#include "cure/cure/kd_tree.hpp"
#include "cure/cure/distance.hpp"
#include <algorithm>
#include <cmath>
#include <queue>

namespace cure {

// ============================================================
// KDNode Implementation
// ============================================================

KDNode::KDNode(const Point& p, Index idx, int ax)
    : point(p), index(idx), axis(ax) {}

// ============================================================
// NNResult Implementation
// ============================================================

NNResult::NNResult(double d, Index i) : distance(d), index(i) {}

bool NNResult::operator<(const NNResult& other) const {
    return distance < other.distance;
}

bool NNResult::operator>(const NNResult& other) const {
    return distance > other.distance;
}

// ============================================================
// KDTree Implementation
// ============================================================

KDTree::KDTree() = default;

KDTree::KDTree(const Matrix& data) {
    build(data);
}

void KDTree::build(const Matrix& data) {
    data_ = data;
    if (data_.empty()) {
        root_ = nullptr;
        n_points_ = 0;
        n_dims_ = 0;
        return;
    }
    
    n_points_ = data_.size();
    n_dims_ = data_[0].size();
    
    // Create index array
    IndexList indices(n_points_);
    for (Index i = 0; i < n_points_; ++i) {
        indices[i] = i;
    }
    
    root_ = buildRecursive(indices, 0);
}

std::unique_ptr<KDNode> KDTree::buildRecursive(IndexList& indices, int depth) {
    if (indices.empty()) {
        return nullptr;
    }
    
    // Choose axis based on depth
    int axis = depth % static_cast<int>(n_dims_);
    
    // Sort indices by value along current axis
    std::sort(indices.begin(), indices.end(), 
        [this, axis](Index a, Index b) {
            return data_[a][axis] < data_[b][axis];
        });
    
    // Choose median as pivot
    size_t mid = indices.size() / 2;
    
    // Create node
    auto node = std::make_unique<KDNode>(data_[indices[mid]], indices[mid], axis);
    
    // Split indices for subtrees
    IndexList left_indices(indices.begin(), indices.begin() + mid);
    IndexList right_indices(indices.begin() + mid + 1, indices.end());
    
    // Recursively build subtrees
    node->left = buildRecursive(left_indices, depth + 1);
    node->right = buildRecursive(right_indices, depth + 1);
    
    return node;
}

std::vector<NNResult> KDTree::query(const Point& query, int k, 
                                     double distance_upper_bound) const {
    if (!root_ || k <= 0) {
        return {};
    }
    
    // Vector to store k best candidates (will maintain sorted order)
    std::vector<NNResult> best;
    best.reserve(k);
    
    searchRecursive(root_.get(), query, k, distance_upper_bound, best);
    
    return best;
}

void KDTree::searchRecursive(const KDNode* node, const Point& query, int k,
                              double distance_upper_bound,
                              std::vector<NNResult>& best) const {
    if (!node) {
        return;
    }
    
    // Compute distance to current node
    double dist = euclidean_distance(query, node->point);
    
    // Update best if within bounds
    if (dist < distance_upper_bound) {
        if (static_cast<int>(best.size()) < k) {
            best.emplace_back(dist, node->index);
            // Keep sorted
            std::sort(best.begin(), best.end());
        } else if (dist < best.back().distance) {
            best.back() = NNResult(dist, node->index);
            std::sort(best.begin(), best.end());
        }
    }
    
    // Determine which subtree to search first
    int axis = node->axis;
    double diff = query[axis] - node->point[axis];
    
    const KDNode* first = (diff <= 0) ? node->left.get() : node->right.get();
    const KDNode* second = (diff <= 0) ? node->right.get() : node->left.get();
    
    // Search the closer subtree first
    searchRecursive(first, query, k, distance_upper_bound, best);
    
    // Check if we need to search the other subtree
    double worst_dist = (static_cast<int>(best.size()) == k) 
                        ? best.back().distance : distance_upper_bound;
    
    if (std::abs(diff) < worst_dist) {
        searchRecursive(second, query, k, distance_upper_bound, best);
    }
}

IndexList KDTree::queryBallPoint(const Point& query, double r) const {
    IndexList results;
    if (!root_) {
        return results;
    }
    
    ballSearchRecursive(root_.get(), query, r, results);
    return results;
}

void KDTree::ballSearchRecursive(const KDNode* node, const Point& query, 
                                  double r, IndexList& results) const {
    if (!node) {
        return;
    }
    
    double dist = euclidean_distance(query, node->point);
    if (dist <= r) {
        results.push_back(node->index);
    }
    
    int axis = node->axis;
    double diff = query[axis] - node->point[axis];
    
    if (diff <= 0) {
        ballSearchRecursive(node->left.get(), query, r, results);
        if (std::abs(diff) <= r) {
            ballSearchRecursive(node->right.get(), query, r, results);
        }
    } else {
        ballSearchRecursive(node->right.get(), query, r, results);
        if (std::abs(diff) <= r) {
            ballSearchRecursive(node->left.get(), query, r, results);
        }
    }
}

const Matrix& KDTree::data() const {
    return data_;
}

size_t KDTree::size() const {
    return n_points_;
}

bool KDTree::empty() const {
    return n_points_ == 0;
}

// ============================================================
// RepresentativeTree Implementation
// ============================================================

RepresentativeTree::RepresentativeTree() = default;

void RepresentativeTree::build(const Matrix& rep_points, 
                                const std::vector<int>& rep_to_cluster) {
    if (rep_points.empty()) {
        kdtree_ = KDTree();
        rep_to_cluster_.clear();
        return;
    }
    
    kdtree_.build(rep_points);
    rep_to_cluster_ = rep_to_cluster;
}

std::pair<int, double> RepresentativeTree::findClosestCluster(
    const std::vector<Point>& query_reps,
    int query_cluster_id,
    double threshold) const 
{
    if (kdtree_.empty() || query_reps.empty()) {
        return {-1, INF};
    }
    
    double min_dist = INF;
    int closest_id = -1;
    
    for (const auto& query_rep : query_reps) {
        // Query for multiple neighbors
        int k = std::min(static_cast<int>(kdtree_.size()), 10);
        auto results = kdtree_.query(query_rep, k, threshold);
        
        for (const auto& result : results) {
            if (result.distance >= threshold || result.index >= rep_to_cluster_.size()) {
                continue;
            }
            
            int neighbor_cluster_id = rep_to_cluster_[result.index];
            
            // Skip if same cluster
            if (neighbor_cluster_id == query_cluster_id) {
                continue;
            }
            
            if (result.distance < min_dist) {
                min_dist = result.distance;
                closest_id = neighbor_cluster_id;
            }
        }
    }
    
    return {closest_id, min_dist};
}

const KDTree& RepresentativeTree::kdtree() const {
    return kdtree_;
}

bool RepresentativeTree::empty() const {
    return kdtree_.empty();
}

} // namespace cure
