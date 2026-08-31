# Local Size-Constrained CURE

This repository provides the C++ implementation of LSC-CURE algorithm, which is used for the paper **"A CURE Clustering Approach for RFM-based Customer Segmentation"**.

For methodological details, please refer to the paper.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

Euclidean Distance:

```bash
build/examples/lsc_cure_csv \
  --input data/demo_rfm.csv --output labels.json \
  --k 3 --c 1 --alpha 0.50 --metric euclidean --f 3 \
  --header --cols 1,2,3 --partitions 2 --seed 42
```

Pearson Correlation Distance:

```bash
build/examples/lsc_cure_csv \
  --input data/demo_rfm.csv --output labels_pearson.json \
  --k 3 --c 1 --alpha 0.50 --metric pearson --f 3 \
  --header --cols 1,2,3 --partitions 2 --seed 42 --use-medoid
```

Output format:

```json
{"labels": [0, 0, 1, ...]}
```

## Input Format

The executable reads numeric CSV files. Use `--header` to skip a header row and `--cols` to select zero-based feature columns. For an RFM table with an ID column followed by `Recency,Frequency,Monetary`, use:

```bash
--header --cols 1,2,3
```

## Main Parameters

- `--k`: final number of clusters.
- `--c`: number of representative points.
- `--alpha`: shrink factor.
- `--metric`: `euclidean` or `pearson`.
- `--f`: local factor, default `5`.
- `--partitions`: number of partitions, default `5`.
- `--seed`: random seed.
- `--use-medoid`: medoid centers, used for Pearson Correlation Distance experiments.
- `--no-kdtree`: disable KD-tree acceleration for Euclidean Distance.

## Data

The real datasets used in the paper are not included in this repository. Please obtain them from their original sources and prepare the RFM CSV files as described in the paper.

The included `data/demo_rfm.csv` is only a small format-checking example.
