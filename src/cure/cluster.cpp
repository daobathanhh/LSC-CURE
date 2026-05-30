#include "cure/cure/cluster.hpp"
#include "cure/cure/distance.hpp"

namespace cure {

// ============================================================
// Cluster Implementation
// ============================================================

Cluster::Cluster() 
    : id(-1), alive(true), dist(INF), closest(-1) {}

Cluster::Cluster(int cluster_id, Index point_idx, const Point& point)
    : id(cluster_id)
    , points_idx{point_idx}
    , reps{point}
    , mean(point)
    , alive(true)
    , dist(INF)
    , closest(-1) {}

Cluster::Cluster(int cluster_id, IndexList indices, std::vector<Point> representatives, Point centroid)
    : id(cluster_id)
    , points_idx(std::move(indices))
    , reps(std::move(representatives))
    , mean(std::move(centroid))
    , alive(true)
    , dist(INF)
    , closest(-1) {}

bool Cluster::operator<(const Cluster& other) const {
    return dist > other.dist;  // Reversed for min-heap behavior
}

std::tuple<double, int> Cluster::toHeapEntry() const {
    return {dist, id};
}

size_t Cluster::size() const {
    return points_idx.size();
}

bool Cluster::empty() const {
    return points_idx.empty();
}

// ============================================================
// HeapEntry Implementation
// ============================================================

HeapEntry::HeapEntry(double d, int id) : distance(d), cluster_id(id) {}

bool HeapEntry::operator>(const HeapEntry& other) const {
    return distance > other.distance;
}

// ============================================================
// Cluster Distance Functions
// ============================================================

double cluster_distance_euclidean(const Cluster& u, const Cluster& v) {
    if (u.reps.empty() || v.reps.empty()) {
        return INF;
    }
    
    double min_dist = INF;
    for (const auto& p : u.reps) {
        for (const auto& q : v.reps) {
            double dist = euclidean_distance(p, q);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }
    return min_dist;
}

double cluster_distance_pearson(const Cluster& u, const Cluster& v) {
    if (u.reps.empty() || v.reps.empty()) {
        return INF;
    }
    
    double min_dist = INF;
    for (const auto& p : u.reps) {
        for (const auto& q : v.reps) {
            double dist = pearson_distance(p, q);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }
    return min_dist;
}

double cluster_distance(const Cluster& u, const Cluster& v, DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::Pearson:
            return cluster_distance_pearson(u, v);
        case DistanceMetric::Euclidean:
        default:
            return cluster_distance_euclidean(u, v);
    }
}

} // namespace cure
