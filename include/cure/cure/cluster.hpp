#ifndef CURE_CPP_CLUSTER_HPP
#define CURE_CPP_CLUSTER_HPP

#include "../common/types.hpp"
#include <vector>
#include <tuple>

namespace cure {

/**
 * @brief Represents a cluster in the CURE algorithm
 * 
 * Each cluster maintains:
 * - ID: Unique cluster identifier
 * - Point indices: Original indices of points in this cluster
 * - Representatives: Scattered representative points (after shrinking)
 * - Mean/Centroid: Central point of the cluster
 * - Closest info: ID and distance to nearest cluster (for heap ordering)
 */
class Cluster {
public:
    int id;                     // Unique cluster identifier
    IndexList points_idx;       // Original point indices in this cluster
    std::vector<Point> reps;    // Representative points (after shrinking)
    Point mean;                 // Centroid of the cluster
    bool alive;                 // Whether cluster is still active
    double dist;                // Distance to closest cluster
    int closest;                // ID of closest cluster (-1 if none)
    
    /**
     * @brief Construct an empty cluster
     */
    Cluster();
    
    /**
     * @brief Construct a cluster with single point
     */
    Cluster(int cluster_id, Index point_idx, const Point& point);
    
    /**
     * @brief Construct a cluster with multiple points
     */
    Cluster(int cluster_id, IndexList indices, std::vector<Point> representatives, Point centroid);
    
    /**
     * @brief Comparison for heap (min-heap by distance)
     */
    bool operator<(const Cluster& other) const;
    
    /**
     * @brief Create heap entry tuple: (distance, id)
     */
    std::tuple<double, int> toHeapEntry() const;
    
    /**
     * @brief Get number of points in cluster
     */
    size_t size() const;
    
    /**
     * @brief Check if cluster is empty
     */
    bool empty() const;
};

/**
 * @brief Heap entry for priority queue
 * Stores (distance, cluster_id) for efficient minimum extraction
 */
struct HeapEntry {
    double distance;
    int cluster_id;
    
    HeapEntry(double d, int id);
    
    // For min-heap: smaller distance has higher priority
    bool operator>(const HeapEntry& other) const;
};

/**
 * @brief Compute distance between two clusters using Euclidean distance
 * 
 * Distance = minimum Euclidean distance between any pair of representatives
 * 
 * @param u First cluster
 * @param v Second cluster
 * @return Minimum distance between representatives
 */
double cluster_distance_euclidean(const Cluster& u, const Cluster& v);

/**
 * @brief Compute distance between two clusters using Pearson correlation
 * 
 * @param u First cluster
 * @param v Second cluster
 * @return Minimum Pearson distance between representatives
 */
double cluster_distance_pearson(const Cluster& u, const Cluster& v);

/**
 * @brief Generic cluster distance function
 */
double cluster_distance(const Cluster& u, const Cluster& v, 
                        DistanceMetric metric = DistanceMetric::Euclidean);

} // namespace cure

#endif // CURE_CPP_CLUSTER_HPP
