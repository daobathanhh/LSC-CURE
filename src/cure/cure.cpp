#include "cure/cure/cure.hpp"
#include "cure/cure/distance.hpp"

#include <queue>
#include <set>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <cmath>
#include <utility>

namespace cure {

// ============================================================
// CURE Implementation
// ============================================================

CURE::CURE(const CureConfig& config)
    : k_(config.k)
    , c_(config.c)
    , alpha_(std::max(0.1, std::min(0.9, config.alpha)))
    , verbose_(config.verbose)
    , metric_(DistanceMetric::Euclidean)
    , outlier_sample_fraction_(config.outlier_sample_fraction >= 0.0 ? std::min(1.0, config.outlier_sample_fraction) : 0.0)
    , use_medoid_(config.use_medoid)
    , max_cluster_size_(std::max(0, config.max_cluster_size))
    , use_kd_tree_(config.use_kd_tree)
    , representatives_from_all_points_(config.representatives_from_all_points) {}

CURE::CURE(int k, int c, double alpha)
    : k_(k), c_(c), alpha_(std::max(0.1, std::min(0.9, alpha))), verbose_(false)
    , metric_(DistanceMetric::Euclidean)
    , outlier_sample_fraction_(0.0)
    , use_medoid_(false)
    , max_cluster_size_(0)
    , use_kd_tree_(true)
    , representatives_from_all_points_(false) {}

void CURE::setMetric(DistanceMetric metric) {
    metric_ = metric;
}

void CURE::setVerbose(bool verbose) {
    verbose_ = verbose;
}

void CURE::setOutlierSampleFraction(double fraction) {
    outlier_sample_fraction_ = (fraction <= 0.0) ? 0.0 : std::min(1.0, fraction);
}

CURE& CURE::fit(const Matrix& data) {
    data_ = data;
    n_ = data_.size();
    
    if (n_ == 0) {
        return *this;
    }
    
    d_ = data_[0].size();
    
    if (static_cast<int>(n_) < k_) {
        throw std::invalid_argument("n_samples must be >= k");
    }
    
    // Outlier-resistant sampling: CURE runs on SAMPLE only; whole dataset is assigned AFTER merge.
    // 1) Build in_sample = points closest to global centroid (or all if fraction=0).
    // 2) Merge phase uses only in_sample — no non-sample point participates in CURE.
    // 3) After k clusters remain, assign ALL n_ points to nearest representative (assignAllPointsByNearestRep).
    IndexList in_sample;
    if (outlier_sample_fraction_ > 0.0 && n_ > static_cast<size_t>(k_)) {
        IndexList all_idx(n_);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        Point centroid = compute_mean(data_, all_idx);
        size_t sample_size = std::min(n_,
            std::max(static_cast<size_t>(k_), static_cast<size_t>(n_ * outlier_sample_fraction_)));
        if (sample_size < n_) {
            in_sample = indices_nearest_to_centroid_heap(data_, centroid, metric_, sample_size);
            if (verbose_) {
                std::cout << "CURE: Outlier sampling — using " << in_sample.size()
                          << "/" << n_ << " points closest to centroid for merge\n";
            }
        }
    }
    if (in_sample.empty()) {
        in_sample.resize(n_);
        std::iota(in_sample.begin(), in_sample.end(), 0);
    }
    
    if (verbose_) {
        std::cout << "CURE: Clustering " << n_ << " points into " 
                  << k_ << " clusters\n";
        std::cout << "  c=" << c_ << " representatives, alpha=" << alpha_ << "\n";
    }
    
    // CURE merge phase: only in_sample points. Clusters and heap contain only sample indices.
    std::unordered_map<int, Cluster> clusters;
    for (Index idx : in_sample) {
        clusters[static_cast<int>(idx)] = Cluster(
            static_cast<int>(idx), idx, data_[idx]
        );
    }
    
    // Find initial closest for each cluster (brute force; only sample clusters)
    if (verbose_) {
        std::cout << "  Finding initial nearest neighbors...\n";
    }
    
    for (auto& kv : clusters) {
        Cluster& cluster = kv.second;
        int closest_id;
        double dist;
        std::tie(closest_id, dist) = findClosestBruteForce(cluster, clusters);
        cluster.closest = closest_id;
        cluster.dist = dist;
    }
    
    // KD-tree accelerates nearest-cluster search only under unconstrained Euclidean L2.
    // With a support cap, admissibility depends on cluster sizes, so brute force is used.
    const bool use_kd_merge = (
        use_kd_tree_ && metric_ == DistanceMetric::Euclidean && max_cluster_size_ <= 0
    );
    KDTree kdtree;
    std::vector<int> rep_map;
    if (use_kd_merge) {
        std::tie(kdtree, rep_map) = buildKDTree(clusters);
    }
    
    // Build min-heap ordered by distance to closest
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> Q;
    for (const auto& kv : clusters) {
        const Cluster& cluster = kv.second;
        if (cluster.closest != -1 && std::isfinite(cluster.dist)) {
            Q.emplace(cluster.dist, cluster.id);
        }
    }
    
    int next_id = static_cast<int>(n_);
    
    if (verbose_) {
        std::cout << "  Starting agglomerative merging...\n";
        if (!use_kd_merge) {
            std::cout << "  (brute-force neighbor search"
                      << (max_cluster_size_ > 0 ? " — size-constrained merges" : " — non-Euclidean metric")
                      << ")\n";
        }
    }
    
    int merge_count = 0;
    
    // Main loop: merge until k clusters remain
    while (static_cast<int>(clusters.size()) > k_) {
        if (Q.empty()) {
            break;
        }
        
        // Extract minimum
        auto top = Q.top();
        Q.pop();
        int u_id = top.cluster_id;
        
        // Skip if cluster no longer exists or is dead
        auto u_it = clusters.find(u_id);
        if (u_it == clusters.end() || !u_it->second.alive) {
            continue;
        }
        
        Cluster& u = u_it->second;
        
        // Get closest cluster v
        int v_id = u.closest;
        auto v_it = clusters.find(v_id);
        
        if (v_it == clusters.end() || !v_it->second.alive || !canMerge(u, v_it->second)) {
            // Recompute closest and re-insert
            int new_closest;
            double new_dist;
            if (use_kd_merge) {
                std::tie(new_closest, new_dist) = findClosestKDTree(u, kdtree, rep_map);
                if (new_closest == -1) {
                    std::tie(new_closest, new_dist) = findClosestBruteForce(u, clusters);
                }
            } else {
                std::tie(new_closest, new_dist) = findClosestBruteForce(u, clusters);
            }
            u.closest = new_closest;
            u.dist = new_dist;
            if (u.closest != -1 && std::isfinite(u.dist)) {
                Q.emplace(u.dist, u.id);
            }
            continue;
        }
        
        Cluster& v = v_it->second;
        
        // Merge clusters u and v
        Cluster w = mergeClusters(u, v, next_id);
        
        // Mark old clusters as dead and remove
        u.alive = false;
        v.alive = false;
        clusters.erase(u_id);
        clusters.erase(v_id);
        clusters[w.id] = std::move(w);
        Cluster& w_ref = clusters[next_id];
        
        if (use_kd_merge) {
            std::tie(kdtree, rep_map) = buildKDTree(clusters);
        }
        
        // Initialize w.closest
        w_ref.dist = INF;
        w_ref.closest = -1;
        
        // Update closest for all clusters
        for (auto& kv : clusters) {
            int x_id = kv.first;
            Cluster& x = kv.second;
            if (x_id == w_ref.id || !x.alive) {
                continue;
            }
            
            bool needs_relocate = false;
            
            // Check if w is closer to x's best
            double dist_w_x = constrainedClusterDistance(w_ref, x);
            if (dist_w_x < w_ref.dist) {
                w_ref.closest = x_id;
                w_ref.dist = dist_w_x;
            }
            
            // If x had u or v as closest
            if (x.closest == u_id || x.closest == v_id) {
                double dist_x_w = constrainedClusterDistance(x, w_ref);
                
                int z_id;
                double dist_x_z;
                if (use_kd_merge) {
                    std::tie(z_id, dist_x_z) = findClosestKDTree(x, kdtree, rep_map, dist_x_w);
                    if (z_id != -1 && dist_x_z < dist_x_w) {
                        x.closest = z_id;
                        x.dist = dist_x_z;
                    } else if (std::isfinite(dist_x_w)) {
                        x.closest = w_ref.id;
                        x.dist = dist_x_w;
                    } else {
                        x.closest = -1;
                        x.dist = INF;
                    }
                } else {
                    std::tie(z_id, dist_x_z) = findClosestBruteForce(x, clusters);
                    if (z_id != -1) {
                        x.closest = z_id;
                        x.dist = dist_x_z;
                    } else if (std::isfinite(dist_x_w)) {
                        x.closest = w_ref.id;
                        x.dist = dist_x_w;
                    } else {
                        x.closest = -1;
                        x.dist = INF;
                    }
                }
                
                needs_relocate = (x.closest != -1 && std::isfinite(x.dist));
            }
            // Check if w is better than current closest
            else {
                double dist_x_w = constrainedClusterDistance(x, w_ref);
                if (dist_x_w < x.dist) {
                    x.closest = w_ref.id;
                    x.dist = dist_x_w;
                    needs_relocate = true;
                }
            }
            
            if (needs_relocate && x.closest != -1 && std::isfinite(x.dist)) {
                Q.emplace(x.dist, x.id);
            }
        }
        
        // Insert w into heap
        if (w_ref.closest != -1 && std::isfinite(w_ref.dist)) {
            Q.emplace(w_ref.dist, w_ref.id);
        }
        
        next_id++;
        merge_count++;
        
        if (verbose_ && merge_count % 100 == 0) {
            std::cout << "    Merged " << merge_count << ", " 
                      << clusters.size() << " clusters remaining\n";
        }
    }
    
    if (verbose_) {
        std::cout << "  Clustering complete: " << clusters.size() << " clusters\n";
    }
    
    // Store final clusters
    clusters_.clear();
    for (auto& kv : clusters) {
        clusters_.push_back(std::move(kv.second));
    }
    
    // Assign labels: if we sampled, CURE was run on sample only — now assign WHOLE dataset.
    labels_.assign(n_, 0);
    if (in_sample.size() < n_) {
        assignAllPointsByNearestRep();  // every point (including non-sample) → nearest cluster rep
    } else {
        for (size_t label = 0; label < clusters_.size(); ++label) {
            for (Index idx : clusters_[label].points_idx) {
                labels_[idx] = static_cast<int>(label);
            }
        }
    }
    
    return *this;
}

std::vector<int> CURE::fit_predict(const Matrix& data) {
    fit(data);
    return labels_;
}

std::vector<int> CURE::predict(const Matrix& data) const {
    if (clusters_.empty()) {
        throw std::runtime_error("Model not fitted. Call fit() first.");
    }
    
    std::vector<int> labels(data.size(), 0);
    
    for (size_t i = 0; i < data.size(); ++i) {
        double min_dist = INF;
        int best_label = 0;
        
        for (size_t label = 0; label < clusters_.size(); ++label) {
            for (const auto& rep : clusters_[label].reps) {
                double dist = (metric_ == DistanceMetric::Euclidean)
                              ? euclidean_distance(data[i], rep)
                              : pearson_distance(data[i], rep);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_label = static_cast<int>(label);
                }
            }
        }
        
        labels[i] = best_label;
    }
    
    return labels;
}

void CURE::assignAllPointsByNearestRep() {
    // Assign whole dataset to clusters (CURE was run on sample only; reps come from sample).
    if (clusters_.empty()) return;
    labels_.resize(n_);
    for (size_t i = 0; i < n_; ++i) {
        double min_dist = INF;
        int best_label = 0;
        for (size_t label = 0; label < clusters_.size(); ++label) {
            for (const auto& rep : clusters_[label].reps) {
                double dist = (metric_ == DistanceMetric::Euclidean)
                    ? euclidean_distance(data_[i], rep)
                    : pearson_distance(data_[i], rep);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_label = static_cast<int>(label);
                }
            }
        }
        labels_[i] = best_label;
    }
}

const std::vector<int>& CURE::labels() const {
    return labels_;
}

const std::vector<Cluster>& CURE::clusters() const {
    return clusters_;
}

const Matrix& CURE::data() const {
    return data_;
}

std::pair<KDTree, std::vector<int>> CURE::buildKDTree(
    const std::unordered_map<int, Cluster>& clusters) const 
{
    Matrix rep_points;
    std::vector<int> rep_map;  // index -> cluster_id
    
    for (const auto& kv : clusters) {
        int id = kv.first;
        const Cluster& cluster = kv.second;
        if (!cluster.alive) continue;
        for (const auto& rep : cluster.reps) {
            rep_map.push_back(id);
            rep_points.push_back(rep);
        }
    }
    
    if (rep_points.empty()) {
        return std::make_pair(KDTree(), std::vector<int>());
    }
    
    return std::make_pair(KDTree(rep_points), rep_map);
}

std::pair<int, double> CURE::findClosestKDTree(
    const Cluster& query,
    const KDTree& kdtree,
    const std::vector<int>& rep_map,
    double threshold) const 
{
    if (kdtree.empty() || query.reps.empty()) {
        return {-1, INF};
    }
    
    double min_dist = INF;
    int closest_id = -1;
    
    for (const auto& query_rep : query.reps) {
        int k = std::min(static_cast<int>(kdtree.size()), c_ * 2 + 2);
        auto results = kdtree.query(query_rep, k, threshold);
        
        for (const auto& result : results) {
            if (result.distance >= threshold || result.index >= rep_map.size()) {
                continue;
            }
            
            int neighbor_id = rep_map[result.index];
            if (neighbor_id == query.id) {
                continue;
            }
            if (result.distance < min_dist) {
                min_dist = result.distance;
                closest_id = neighbor_id;
            }
        }
    }
    
    return {closest_id, min_dist};
}

std::pair<int, double> CURE::findClosestBruteForce(
    const Cluster& query,
    const std::unordered_map<int, Cluster>& clusters) const 
{
    double min_dist = INF;
    int closest_id = -1;
    
    for (const auto& kv : clusters) {
        int id = kv.first;
        const Cluster& cluster = kv.second;
        if (id == query.id || !cluster.alive) {
            continue;
        }
        
        double dist = constrainedClusterDistance(query, cluster);
        if (dist < min_dist) {
            min_dist = dist;
            closest_id = id;
        }
    }
    
    return {closest_id, min_dist};
}

bool CURE::canMerge(const Cluster& u, const Cluster& v) const {
    if (max_cluster_size_ <= 0) {
        return true;
    }
    return static_cast<int>(u.points_idx.size() + v.points_idx.size()) <= max_cluster_size_;
}

double CURE::constrainedClusterDistance(const Cluster& u, const Cluster& v) const {
    if (!canMerge(u, v)) {
        return INF;
    }
    return cluster_distance(u, v, metric_);
}

Point CURE::unshrinkPoint(const Point& shrunk, const Point& mean) const {
    if (std::abs(alpha_ - 1.0) < EPS) {
        return mean;
    }
    
    Point original(shrunk.size());
    for (size_t i = 0; i < shrunk.size(); ++i) {
        original[i] = (shrunk[i] - alpha_ * mean[i]) / (1.0 - alpha_);
    }
    return original;
}

Cluster CURE::mergeClusters(const Cluster& u, const Cluster& v, int new_id) const {
    // Combine points
    IndexList w_points_idx = u.points_idx;
    w_points_idx.insert(w_points_idx.end(), v.points_idx.begin(), v.points_idx.end());
    
    // Cluster center: mean (default) or medoid under metric_ (min total distance to members)
    Point w_center = use_medoid_
        ? compute_medoid(data_, w_points_idx, metric_)
        : compute_mean(data_, w_points_idx);
    
    std::vector<Point> candidates;
    if (representatives_from_all_points_) {
        for (Index idx : w_points_idx) {
            candidates.push_back(data_[idx]);
        }
    } else {
        // Get candidate representatives by unshrinking
        std::set<std::vector<double>> candidates_set;
        for (const auto& shrunk : u.reps) {
            Point original = unshrinkPoint(shrunk, u.mean);
            candidates_set.insert(original);
        }
        for (const auto& shrunk : v.reps) {
            Point original = unshrinkPoint(shrunk, v.mean);
            candidates_set.insert(original);
        }
        
        candidates.assign(candidates_set.begin(), candidates_set.end());
        
        // If not enough candidates, use all points
        if (static_cast<int>(candidates.size()) < c_) {
            candidates.clear();
            for (Index idx : w_points_idx) {
                candidates.push_back(data_[idx]);
            }
        }
    }
    
    // Select c scattered points
    std::vector<Point> selected;
    
    // First point: farthest from center (mean or medoid)
    double max_dist = -1;
    size_t best_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        double dist = (metric_ == DistanceMetric::Euclidean)
                      ? euclidean_distance(candidates[i], w_center)
                      : pearson_distance(candidates[i], w_center);
        if (representatives_from_all_points_ ? (dist >= max_dist) : (dist > max_dist)) {
            max_dist = dist;
            best_idx = i;
        }
    }
    
    if (!candidates.empty()) {
        selected.push_back(candidates[best_idx]);
    }
    
    // Remaining points: maximize minimum distance to selected
    for (int i = 1; i < c_ && !selected.empty(); ++i) {
        double max_min_dist = -1;
        size_t best_candidate = 0;
        bool found = false;
        
        for (size_t j = 0; j < candidates.size(); ++j) {
            // Skip if already selected
            bool is_selected = false;
            for (const auto& s : selected) {
                if (candidates[j] == s) {
                    is_selected = true;
                    break;
                }
            }
            if (is_selected) continue;
            
            // Find minimum distance to selected
            double min_dist = INF;
            for (const auto& s : selected) {
                double dist = (metric_ == DistanceMetric::Euclidean)
                              ? euclidean_distance(candidates[j], s)
                              : pearson_distance(candidates[j], s);
                min_dist = std::min(min_dist, dist);
            }
            
            if (representatives_from_all_points_ ? (min_dist >= max_min_dist) : (min_dist > max_min_dist)) {
                max_min_dist = min_dist;
                best_candidate = j;
                found = true;
            }
        }
        
        if (found) {
            selected.push_back(candidates[best_candidate]);
        } else {
            break;
        }
    }
    
    // Shrink toward center (mean or medoid)
    std::vector<Point> shrunk_reps;
    for (const auto& p : selected) {
        Point shrunk(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            shrunk[i] = p[i] + alpha_ * (w_center[i] - p[i]);
        }
        shrunk_reps.push_back(shrunk);
    }
    
    return Cluster(new_id, w_points_idx, shrunk_reps, w_center);
}

// ============================================================
// LSCCURE Implementation
// ============================================================

LSCCURE::LSCCURE(const LSCCureConfig& config)
    : k_(config.k)
    , c_(config.c)
    , alpha_(std::max(0.1, std::min(0.9, config.alpha)))
    , sample_size_(config.sample_size)
    , n_partitions_(config.n_partitions)
    , local_factor_(std::max(1, config.local_factor))
    , outlier_threshold_(config.outlier_threshold)
    , random_seed_(config.random_seed)
    , sample_size_auto_(config.sample_size_auto)
    , sample_size_exponent_(config.sample_size_exponent)
    , sample_size_multiplier_(config.sample_size_multiplier)
    , sample_size_min_(std::max(1, config.sample_size_min))
    , sample_size_max_(std::max(1, config.sample_size_max))
    , use_centroid_sampling_(config.use_centroid_sampling)
    , use_heap_for_centroid_sample_(config.use_heap_for_centroid_sample)
    , use_kd_tree_(config.use_kd_tree)
    , verbose_(config.verbose)
    , metric_(DistanceMetric::Euclidean)
    , use_medoid_(config.use_medoid)
    , representatives_from_all_points_(config.representatives_from_all_points) {}

void LSCCURE::setMetric(DistanceMetric metric) {
    metric_ = metric;
}

void LSCCURE::setVerbose(bool verbose) {
    verbose_ = verbose;
}

LSCCURE& LSCCURE::fit(const Matrix& data) {
    data_ = data;
    n_ = data_.size();
    
    if (n_ == 0) {
        return *this;
    }
    
    d_ = data_[0].size();
    
    // Setup random generator
    std::mt19937 rng;
    if (random_seed_ >= 0) {
        rng.seed(static_cast<unsigned>(random_seed_));
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
    
    if (verbose_) {
        std::cout << "LSC-CURE: " << n_ << " points -> " << k_ << " clusters\n";
    }
    
    // Step 1: Sampling (random or nearest-to-centroid)
    size_t sample_n = computeSampleSize();
    sample_n = std::min(sample_n, n_);
    
    IndexList sample_indices;
    sample_indices.reserve(sample_n);
    
    if (use_centroid_sampling_) {
        IndexList all_idx(n_);
        std::iota(all_idx.begin(), all_idx.end(), 0);
        Point centroid = compute_mean(data_, all_idx);
        if (use_heap_for_centroid_sample_) {
            sample_indices = indices_nearest_to_centroid_heap(data_, centroid, metric_, sample_n);
        } else {
            std::vector<std::pair<double, Index>> dist_idx;
            dist_idx.reserve(n_);
            for (Index i = 0; i < n_; ++i) {
                double d = (metric_ == DistanceMetric::Euclidean)
                    ? euclidean_distance(data_[i], centroid)
                    : pearson_distance(data_[i], centroid);
                dist_idx.emplace_back(d, i);
            }
            std::sort(dist_idx.begin(), dist_idx.end());
            sample_indices.reserve(sample_n);
            for (size_t j = 0; j < sample_n; ++j)
                sample_indices.push_back(dist_idx[j].second);
        }
        if (verbose_) {
            std::cout << "  Step 1: Sampled " << sample_n << " points (nearest to centroid, "
                      << (use_heap_for_centroid_sample_ ? "heap" : "full sort") << ")\n";
        }
    } else {
        std::vector<Index> all_indices(n_);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        sample_indices.assign(all_indices.begin(), all_indices.begin() + sample_n);
        if (verbose_) {
            std::cout << "  Step 1: Sampled " << sample_n << " points (random)\n";
        }
    }
    
    // Step 2: Partition sample
    size_t partition_size = sample_n / n_partitions_;
    std::vector<std::pair<Matrix, IndexList>> partitions;
    
    for (int i = 0; i < n_partitions_; ++i) {
        size_t start = i * partition_size;
        size_t end = (i == n_partitions_ - 1) ? sample_n : start + partition_size;
        
        Matrix part_data;
        IndexList part_indices;
        for (size_t j = start; j < end; ++j) {
            part_data.push_back(data_[sample_indices[j]]);
            part_indices.push_back(sample_indices[j]);
        }
        partitions.emplace_back(std::move(part_data), std::move(part_indices));
    }
    
    if (verbose_) {
        std::cout << "  Step 2: Created " << n_partitions_ << " partitions\n";
    }
    
    // Step 3: Size-constrained local clustering on each partition
    std::vector<Cluster> all_clusters;
    for (int i = 0; i < n_partitions_; ++i) {
        const Matrix& part_data = partitions[i].first;
        const IndexList& part_indices = partitions[i].second;
        if (part_data.empty()) continue;
        
        const int target = std::min(
            static_cast<int>(part_data.size()),
            std::max(k_, local_factor_ * k_));
        const int max_cluster_size = std::max(
            1,
            static_cast<int>(part_data.size()) / std::max(1, k_));
        const int min_keep_size = std::max(1, c_);
        std::vector<Cluster> partial = partialCluster(part_data, part_indices, target, max_cluster_size);
        
        // Remove small clusters (outliers)
        for (auto& c : partial) {
            if (static_cast<int>(c.points_idx.size()) >= min_keep_size) {
                all_clusters.push_back(std::move(c));
            }
        }
        
        if (verbose_) {
            std::cout << "    Partition " << i + 1 << ": " << part_data.size()
                      << " pts -> " << partial.size() << " clusters"
                      << " (target=" << target
                      << ", U=" << max_cluster_size
                      << ", keep>=" << min_keep_size << ")\n";
        }
    }
    
    if (verbose_) {
        std::cout << "  Step 3: Total partial clusters: " << all_clusters.size() << "\n";
    }
    
    // Step 4: Second pass - cluster the partial clusters
    if (static_cast<int>(all_clusters.size()) > k_) {
        // Build synthetic dataset from representatives
        Matrix rep_data;
        std::vector<int> rep_to_cluster;
        
        for (size_t i = 0; i < all_clusters.size(); ++i) {
            for (const auto& rep : all_clusters[i].reps) {
                rep_data.push_back(rep);
                rep_to_cluster.push_back(static_cast<int>(i));
            }
        }
        
        // Run CURE on representatives
        CureConfig config;
        config.k = k_;
        config.c = c_;
        config.alpha = alpha_;
        config.use_medoid = use_medoid_;
        config.use_kd_tree = use_kd_tree_;
        config.representatives_from_all_points = representatives_from_all_points_;
        
        CURE cure(config);
        cure.setMetric(metric_);
        cure.fit(rep_data);
        
        // Map back to original clusters
        std::vector<Cluster> final_clusters;
        for (size_t label = 0; label < cure.clusters().size(); ++label) {
            const auto& cure_cluster = cure.clusters()[label];
            
            IndexList merged_points;
            for (Index rep_idx : cure_cluster.points_idx) {
                const auto& original_cluster = all_clusters[rep_to_cluster[rep_idx]];
                merged_points.insert(merged_points.end(),
                    original_cluster.points_idx.begin(),
                    original_cluster.points_idx.end());
            }
            
            // Remove duplicates
            std::sort(merged_points.begin(), merged_points.end());
            merged_points.erase(
                std::unique(merged_points.begin(), merged_points.end()),
                merged_points.end()
            );
            
            final_clusters.emplace_back(
                static_cast<int>(label),
                merged_points,
                cure_cluster.reps,
                cure_cluster.mean
            );
        }
        
        all_clusters = std::move(final_clusters);
    }
    
    if (verbose_) {
        std::cout << "  Step 4: Merged to " << all_clusters.size() << " clusters\n";
    }
    
    // Step 5: Eliminate outlier clusters (small ones), but always keep at least k_ clusters
    clusters_.clear();
    std::vector<Cluster> kept, discarded;
    for (auto& c : all_clusters) {
        if (static_cast<int>(c.points_idx.size()) > outlier_threshold_) {
            kept.push_back(std::move(c));
        } else {
            discarded.push_back(std::move(c));
        }
    }
    // If we dropped too many and have fewer than k_ clusters, keep largest of the discarded
    if (static_cast<int>(kept.size()) < k_ && !discarded.empty()) {
        std::sort(discarded.begin(), discarded.end(),
            [](const Cluster& a, const Cluster& b) {
                return a.points_idx.size() > b.points_idx.size();
            });
        size_t need = static_cast<size_t>(k_) - kept.size();
        for (size_t i = 0; i < need && i < discarded.size(); ++i) {
            kept.push_back(std::move(discarded[i]));
        }
    }
    clusters_ = std::move(kept);
    
    if (verbose_) {
        std::cout << "  Step 5: After outlier removal: " << clusters_.size() << " clusters\n";
    }
    
    // Step 6: Label all data points
    assignLabels();
    
    if (verbose_) {
        std::cout << "  Step 6: Labeled all " << n_ << " points\n";
    }
    
    return *this;
}

std::vector<int> LSCCURE::fit_predict(const Matrix& data) {
    fit(data);
    return labels_;
}

const std::vector<int>& LSCCURE::labels() const {
    return labels_;
}

const std::vector<Cluster>& LSCCURE::clusters() const {
    return clusters_;
}

size_t LSCCURE::computeSampleSize() const {
    if (sample_size_auto_) {
        return lsc_sample_size(
            n_,
            sample_size_multiplier_,
            sample_size_exponent_,
            sample_size_min_,
            sample_size_max_);
    }
    if (sample_size_ <= 1.0) {
        size_t s = std::max(static_cast<size_t>(k_ * 10),
                            std::max(static_cast<size_t>(sample_size_min_),
                                     static_cast<size_t>(n_ * sample_size_)));
        s = std::min(s, static_cast<size_t>(sample_size_max_));
        s = std::min(s, n_);
        return s;
    }
    return std::min(std::max(static_cast<size_t>(sample_size_), static_cast<size_t>(sample_size_min_)), n_);
}

std::vector<Cluster> LSCCURE::partialCluster(
    const Matrix& points,
    const IndexList& indices,
    int target_clusters,
    int max_cluster_size) const 
{
    if (points.size() <= static_cast<size_t>(target_clusters)) {
        std::vector<Cluster> result;
        for (size_t i = 0; i < points.size(); ++i) {
            result.emplace_back(static_cast<int>(i), indices[i], points[i]);
        }
        return result;
    }
    
    // Use base CURE
    CureConfig config;
    config.k = target_clusters;
    config.c = c_;
    config.alpha = alpha_;
    config.use_medoid = use_medoid_;
    config.max_cluster_size = max_cluster_size;
    config.use_kd_tree = use_kd_tree_;
    config.representatives_from_all_points = representatives_from_all_points_;
    
    CURE cure(config);
    cure.setMetric(metric_);
    cure.fit(points);
    
    // Map labels back to original indices
    std::vector<Cluster> result;
    for (size_t label = 0; label < cure.clusters().size(); ++label) {
        const auto& cluster = cure.clusters()[label];
        
        IndexList original_indices;
        for (Index i : cluster.points_idx) {
            original_indices.push_back(indices[i]);
        }
        
        result.emplace_back(
            static_cast<int>(label),
            original_indices,
            cluster.reps,
            cluster.mean
        );
    }
    
    return result;
}

void LSCCURE::assignLabels() {
    labels_.assign(n_, 0);

    if (clusters_.empty()) {
        return;
    }

    // Collect all representatives
    Matrix all_reps;
    std::vector<int> rep_labels;

    for (size_t label = 0; label < clusters_.size(); ++label) {
        for (const auto& rep : clusters_[label].reps) {
            all_reps.push_back(rep);
            rep_labels.push_back(static_cast<int>(label));
        }
    }

    if (all_reps.empty()) {
        return;
    }

    // Brute force: assign each point to nearest representative
    for (size_t i = 0; i < n_; ++i) {
        double min_dist = INF;
        int best_label = 0;
        for (size_t r = 0; r < all_reps.size(); ++r) {
            double d = (metric_ == DistanceMetric::Pearson)
                ? pearson_distance(data_[i], all_reps[r])
                : euclidean_distance(data_[i], all_reps[r]);
            if (d < min_dist) {
                min_dist = d;
                best_label = rep_labels[r];
            }
        }
        labels_[i] = best_label;
    }
}

// ============================================================
// Convenience Function
// ============================================================

std::vector<int> cure_clustering(
    const Matrix& data,
    int k,
    int c,
    double alpha,
    bool use_lsc,
    bool verbose)
{
    if (use_lsc || data.size() > 5000) {
        LSCCureConfig config;
        config.k = k;
        config.c = c;
        config.alpha = alpha;
        config.verbose = verbose;
        
        LSCCURE cure(config);
        return cure.fit_predict(data);
    } else {
        CureConfig config;
        config.k = k;
        config.c = c;
        config.alpha = alpha;
        config.verbose = verbose;
        
        CURE cure(config);
        return cure.fit_predict(data);
    }
}

} // namespace cure
