#ifndef CURE_CPP_TYPES_HPP
#define CURE_CPP_TYPES_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace cure {

// Type aliases for clarity
using Point = std::vector<double>;
using Matrix = std::vector<Point>;
using Index = std::size_t;
using IndexList = std::vector<Index>;

// Constants
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-10;

/**
 * LSC-CURE step-1 sample size:
 * min(max(multiplier * n^exponent, min_pts), max_pts, n).
 */
inline size_t lsc_sample_size(
    size_t n,
    double multiplier = 50.0,
    double exponent = 1.0 / 3.0,
    int min_pts = 5000,
    int max_pts = 15000)
{
    if (n == 0) {
        return 0;
    }
    const double raw = multiplier * std::pow(static_cast<double>(n), exponent);
    size_t s = static_cast<size_t>(std::round(raw));
    s = std::max(s, static_cast<size_t>(std::max(1, min_pts)));
    s = std::min(s, static_cast<size_t>(std::max(1, max_pts)));
    s = std::min(s, n);
    return s;
}

/**
 * @brief Distance metric types.
 */
enum class DistanceMetric {
    Euclidean,
    Pearson
};

/**
 * @brief Configuration for CURE algorithm
 */
struct CureConfig {
    int k = 5;              // Number of clusters
    int c = 5;              // Number of representative points per cluster
    double alpha = 0.3;     // Shrink factor (0.1 to 0.9 only)
    bool verbose = false;   // Print progress
    /** Distance metric for all CURE and merge steps (Euclidean or Pearson). */
    DistanceMetric metric = DistanceMetric::Euclidean;
    /** Outlier-resistant sampling: fraction of points closest to global centroid
     *  used for merge phase (0 = disabled). E.g. 0.5 = 1/2, 0.2 = 1/5. Then all
     *  points are assigned to nearest cluster. */
    double outlier_sample_fraction = 0.0;
    /** If true: cluster center = medoid (min total distance to members); shrink toward medoid. Uses metric_. */
    bool use_medoid = false;
    /** If > 0, disallow agglomerative merges whose combined support exceeds this size. */
    int max_cluster_size = 0;
    /** If true, Euclidean CURE may use KD-tree acceleration; false forces exact brute-force merge search. */
    bool use_kd_tree = true;
    /** If true, representative selection scans all member points, matching pyclustering CURE semantics. */
    bool representatives_from_all_points = false;

    CureConfig() = default;
    CureConfig(int k_, int c_, double alpha_) : k(k_), c(c_), alpha(alpha_) {}
};

/**
 * @brief Configuration for LSC-CURE.
 */
struct LSCCureConfig : public CureConfig {
    /** Fraction (0-1) or absolute count; ignored if sample_size_auto is true. */
    double sample_size = 0.1;
    int n_partitions = 5;
    /** Local over-clustering factor f: each partition targets at most f * k clusters. */
    int local_factor = 5;
    int outlier_threshold = 0;
    /** RNG seed for Step-1 random subsampling only (when use_centroid_sampling is false). Ignored for centroid sampling. */
    int random_seed = -1;
    /** If true: sample_n = lsc_sample_size(n, multiplier, exponent, min, max). */
    bool sample_size_auto = true;
    double sample_size_exponent = 1.0 / 3.0;  // n^(1/3)
    double sample_size_multiplier = 50.0;     // C in C * n^(1/3)
    int sample_size_min = 5000;
    int sample_size_max = 15000;              // cap for tractable Step-4 CURE on reps
    /** If true: sample = points nearest to global centroid (outlier-resistant); else random. */
    bool use_centroid_sampling = false;
    /** If true: nearest-to-centroid selection uses heap O(n log k). */
    bool use_heap_for_centroid_sample = true;
    /** If true, Euclidean CURE merges may use KD-tree acceleration. Pearson remains brute force. */
    bool use_kd_tree = true;

    LSCCureConfig() = default;
    LSCCureConfig(int k_, int c_, double alpha_)
        : CureConfig(k_, c_, alpha_) {}
};

} // namespace cure

#endif // CURE_CPP_TYPES_HPP
