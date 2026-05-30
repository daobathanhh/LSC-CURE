#ifndef CURE_CPP_KD_TREE_HPP
#define CURE_CPP_KD_TREE_HPP

#include "../common/types.hpp"
#include <memory>
#include <vector>

namespace cure {

/**
 * @brief Node in the KD-tree
 */
struct KDNode {
    Point point;
    Index index;        // Original index in the data array
    int axis;           // Splitting axis
    std::unique_ptr<KDNode> left;
    std::unique_ptr<KDNode> right;
    
    KDNode(const Point& p, Index idx, int ax);
};

/**
 * @brief Result of a nearest neighbor query
 */
struct NNResult {
    double distance;
    Index index;
    
    NNResult(double d = INF, Index i = 0);
    
    bool operator<(const NNResult& other) const;
    bool operator>(const NNResult& other) const;
};

/**
 * @brief KD-Tree for efficient nearest neighbor search
 * 
 * Time complexity:
 * - Build: O(n log n)
 * - Query (average): O(log n)
 * - Query (worst): O(n) for highly unbalanced trees
 */
class KDTree {
public:
    KDTree();
    
    /**
     * @brief Build KD-tree from data points
     * @param data Array of points (n_points, n_dimensions)
     */
    explicit KDTree(const Matrix& data);
    
    /**
     * @brief Build the tree from data
     */
    void build(const Matrix& data);
    
    /**
     * @brief Find k nearest neighbors to query point
     * @param query Query point
     * @param k Number of neighbors to find
     * @param distance_upper_bound Maximum distance to consider
     * @return Vector of (distance, index) pairs sorted by distance
     */
    std::vector<NNResult> query(const Point& query, int k = 1, 
                                 double distance_upper_bound = INF) const;
    
    /**
     * @brief Find all points within radius r of query point
     * @param query Query point
     * @param r Search radius
     * @return List of indices of points within radius
     */
    IndexList queryBallPoint(const Point& query, double r) const;
    
    /**
     * @brief Get the stored data
     */
    const Matrix& data() const;
    
    /**
     * @brief Get number of points
     */
    size_t size() const;
    
    /**
     * @brief Check if tree is empty
     */
    bool empty() const;

private:
    std::unique_ptr<KDNode> root_;
    Matrix data_;
    size_t n_points_ = 0;
    size_t n_dims_ = 0;
    
    std::unique_ptr<KDNode> buildRecursive(IndexList& indices, int depth);
    
    void searchRecursive(const KDNode* node, const Point& query, int k,
                         double distance_upper_bound,
                         std::vector<NNResult>& best) const;
    
    void ballSearchRecursive(const KDNode* node, const Point& query, 
                              double r, IndexList& results) const;
};

/**
 * @brief Tree for storing cluster representative points
 * 
 * Maps representative points back to cluster IDs for efficient
 * nearest cluster lookup.
 */
class RepresentativeTree {
public:
    RepresentativeTree();
    
    /**
     * @brief Build tree from clusters' representative points
     * @param rep_points All representative points
     * @param rep_to_cluster Mapping from rep index to cluster ID
     */
    void build(const Matrix& rep_points, const std::vector<int>& rep_to_cluster);
    
    /**
     * @brief Find closest cluster to query representatives
     * @param query_reps Representative points of query cluster
     * @param query_cluster_id ID of query cluster (to exclude from results)
     * @param threshold Maximum distance to consider
     * @return Pair of (closest_cluster_id, distance), or (-1, INF) if not found
     */
    std::pair<int, double> findClosestCluster(
        const std::vector<Point>& query_reps,
        int query_cluster_id,
        double threshold = INF) const;
    
    /**
     * @brief Get the underlying KD-tree
     */
    const KDTree& kdtree() const;
    
    /**
     * @brief Check if tree is empty
     */
    bool empty() const;

private:
    KDTree kdtree_;
    std::vector<int> rep_to_cluster_;  // rep_index -> cluster_id
};

} // namespace cure

#endif // CURE_CPP_KD_TREE_HPP
