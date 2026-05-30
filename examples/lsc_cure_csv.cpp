/**
 * @file lsc_cure_csv.cpp
 * @brief Run LSC-CURE on a CSV file and write labels to JSON.
 *
 * Example:
 *   lsc_cure_csv --input data.csv --k 5 --c 10 --alpha 0.50 --metric euclidean --output labels.json --header --cols 1,2,3 --partitions=5 --f=5 --seed=42
 *   lsc_cure_csv --input data.csv --k 5 --c 10 --alpha 0.50 --metric pearson --output labels.json --header --cols 1,2,3 --partitions=5 --f=5 --seed=42 --use-medoid
 */

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cure/cure/cure.hpp"

using namespace cure;

static std::set<int> parseColIndices(const std::string& s) {
    std::set<int> indices;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        indices.insert(std::stoi(token));
    }
    return indices;
}

static Matrix readCSV(const std::string& filename, bool skip_header, const std::set<int>& col_indices) {
    Matrix data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open " << filename << "\n";
        return data;
    }

    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (skip_header && first) {
            first = false;
            std::stringstream header_stream(line);
            std::string first_cell;
            if (std::getline(header_stream, first_cell, ',')) {
                bool numeric = true;
                for (char ch : first_cell) {
                    if (!std::isdigit(static_cast<unsigned char>(ch)) && ch != '-' && ch != '.' && ch != '+') {
                        numeric = false;
                        break;
                    }
                }
                if (!numeric) continue;
            }
        }

        std::vector<double> all_cells;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            all_cells.push_back(std::stod(value));
        }

        Point row;
        if (col_indices.empty()) {
            row = all_cells;
        } else {
            for (int idx : col_indices) {
                if (idx >= 0 && idx < static_cast<int>(all_cells.size())) {
                    row.push_back(all_cells[idx]);
                }
            }
        }
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

static bool writeLabelsJson(const std::string& path, const std::vector<int>& labels) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "{\"labels\": [";
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i > 0) out << ", ";
        out << labels[i];
    }
    out << "]}\n";
    return true;
}

static void printUsage() {
    std::cerr << "Usage: lsc_cure_csv --input data.csv --k K --c C --alpha A --metric M --output labels.json [options]\n";
    std::cerr << "  Options:\n";
    std::cerr << "    --input PATH                input CSV file\n";
    std::cerr << "    --output PATH               output labels JSON file\n";
    std::cerr << "    --k N                       final number of clusters\n";
    std::cerr << "    --c N                       representative points per cluster\n";
    std::cerr << "    --alpha X                   representative shrink factor\n";
    std::cerr << "    --metric euclidean|pearson  distance metric\n";
    std::cerr << "    --f N or --f=N              local over-clustering factor, default 5\n";
    std::cerr << "    --header                    skip CSV header row\n";
    std::cerr << "    --cols i,j,...              0-based input columns to use\n";
    std::cerr << "    --partitions=N              number of sampled partitions, default 5\n";
    std::cerr << "    --seed=N                    random seed for sampling\n";
    std::cerr << "    --sample-min=N              auto-sample lower bound, default 5000\n";
    std::cerr << "    --sample-max=N              auto-sample upper bound, default 15000\n";
    std::cerr << "    --sample-multiplier=X       auto-sample multiplier, default 50\n";
    std::cerr << "    --sample-exponent=X         auto-sample exponent, default 1/3\n";
    std::cerr << "    --sample-size=N             fixed absolute sample size\n";
    std::cerr << "    --use-medoid                use medoid centers; recommended with pearson\n";
    std::cerr << "    --no-kdtree                 disable KD-tree acceleration for Euclidean CURE merges\n";
    std::cerr << "    --centroid-sampling         sample nearest-to-centroid points instead of random\n";
    std::cerr << "    --pure-representatives      select representatives from all member points\n";
    std::cerr << "    --verbose                   print LSC-CURE progress\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string csv_path;
    std::string out_path;
    std::string metric_str;
    bool have_k = false;
    bool have_c = false;
    bool have_alpha = false;

    bool skip_header = false;
    bool verbose = false;
    bool use_medoid = false;
    bool centroid_sampling = false;
    bool pure_representatives = false;
    std::set<int> col_indices;

    LSCCureConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg.rfind("--input=", 0) == 0) {
            csv_path = arg.substr(8);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg.rfind("--output=", 0) == 0) {
            out_path = arg.substr(9);
        } else if (arg == "--k" && i + 1 < argc) {
            config.k = std::stoi(argv[++i]);
            have_k = true;
        } else if (arg.rfind("--k=", 0) == 0) {
            config.k = std::stoi(arg.substr(4));
            have_k = true;
        } else if (arg == "--c" && i + 1 < argc) {
            config.c = std::stoi(argv[++i]);
            have_c = true;
        } else if (arg.rfind("--c=", 0) == 0) {
            config.c = std::stoi(arg.substr(4));
            have_c = true;
        } else if (arg == "--alpha" && i + 1 < argc) {
            config.alpha = std::stod(argv[++i]);
            have_alpha = true;
        } else if (arg.rfind("--alpha=", 0) == 0) {
            config.alpha = std::stod(arg.substr(8));
            have_alpha = true;
        } else if (arg == "--metric" && i + 1 < argc) {
            metric_str = argv[++i];
        } else if (arg.rfind("--metric=", 0) == 0) {
            metric_str = arg.substr(9);
        } else if (arg == "--header") {
            skip_header = true;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--use-medoid") {
            use_medoid = true;
        } else if (arg == "--no-kdtree") {
            config.use_kd_tree = false;
        } else if (arg == "--centroid-sampling") {
            centroid_sampling = true;
        } else if (arg == "--pure-representatives") {
            pure_representatives = true;
        } else if (arg == "--cols" && i + 1 < argc) {
            col_indices = parseColIndices(argv[++i]);
        } else if (arg.rfind("--cols=", 0) == 0) {
            col_indices = parseColIndices(arg.substr(7));
        } else if (arg == "--partitions" && i + 1 < argc) {
            config.n_partitions = std::stoi(argv[++i]);
        } else if (arg.rfind("--partitions=", 0) == 0) {
            config.n_partitions = std::stoi(arg.substr(13));
        } else if (arg == "--f" && i + 1 < argc) {
            config.local_factor = std::stoi(argv[++i]);
        } else if (arg.rfind("--f=", 0) == 0) {
            config.local_factor = std::stoi(arg.substr(4));
        } else if (arg.rfind("--seed=", 0) == 0) {
            config.random_seed = std::stoi(arg.substr(7));
        } else if (arg == "--seed" && i + 1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        } else if (arg.rfind("--sample-min=", 0) == 0) {
            config.sample_size_min = std::stoi(arg.substr(13));
        } else if (arg == "--sample-min" && i + 1 < argc) {
            config.sample_size_min = std::stoi(argv[++i]);
        } else if (arg.rfind("--sample-max=", 0) == 0) {
            config.sample_size_max = std::stoi(arg.substr(13));
        } else if (arg == "--sample-max" && i + 1 < argc) {
            config.sample_size_max = std::stoi(argv[++i]);
        } else if (arg.rfind("--sample-multiplier=", 0) == 0) {
            config.sample_size_multiplier = std::stod(arg.substr(20));
        } else if (arg == "--sample-multiplier" && i + 1 < argc) {
            config.sample_size_multiplier = std::stod(argv[++i]);
        } else if (arg.rfind("--sample-exponent=", 0) == 0) {
            config.sample_size_exponent = std::stod(arg.substr(18));
        } else if (arg == "--sample-exponent" && i + 1 < argc) {
            config.sample_size_exponent = std::stod(argv[++i]);
        } else if (arg.rfind("--sample-size=", 0) == 0) {
            config.sample_size = std::stod(arg.substr(14));
            config.sample_size_auto = false;
        } else if (arg == "--sample-size" && i + 1 < argc) {
            config.sample_size = std::stod(argv[++i]);
            config.sample_size_auto = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (csv_path.empty() || out_path.empty() || metric_str.empty() || !have_k || !have_c || !have_alpha) {
        std::cerr << "Missing required argument.\n";
        printUsage();
        return 1;
    }

    DistanceMetric metric = DistanceMetric::Euclidean;
    if (metric_str == "pearson") {
        metric = DistanceMetric::Pearson;
    } else if (metric_str != "euclidean") {
        std::cerr << "Unknown metric: " << metric_str << " (use euclidean or pearson)\n";
        return 1;
    }

    Matrix data = readCSV(csv_path, skip_header, col_indices);
    if (data.empty()) {
        std::cerr << "No data read from " << csv_path << "\n";
        return 1;
    }

    config.verbose = verbose;
    config.use_medoid = use_medoid;
    config.use_centroid_sampling = centroid_sampling;
    config.representatives_from_all_points = pure_representatives;

    LSCCURE model(config);
    model.setMetric(metric);
    std::vector<int> labels = model.fit_predict(data);

    if (!writeLabelsJson(out_path, labels)) {
        std::cerr << "Cannot write " << out_path << "\n";
        return 1;
    }

    std::cerr << "Data: " << data.size() << " points, " << data[0].size() << " dims\n";
    std::cerr << "Done. Labels written to " << out_path << "\n";
    return 0;
}
