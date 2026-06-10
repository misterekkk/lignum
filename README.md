# Lignum: Ultra-Fast Decision Tree Inference

[![R package check](https://github.com/misterekkk/lignum/actions/workflows/r-check.yml/badge.svg)](https://github.com/misterekkk/lignum/actions/workflows/r-check.yml)
[![Python package test](https://github.com/misterekkk/lignum/actions/workflows/py-check.yml/badge.svg)](https://github.com/misterekkk/lignum/actions/workflows/py-check.yml)
[![Deploy Documentation](https://github.com/misterekkk/lignum/actions/workflows/docs.yml/badge.svg)](https://github.com/misterekkk/lignum/actions/workflows/docs.yml)

`lignum` is a high-performance inference library for decision-tree models (LightGBM, XGBoost) written in native C++. By utilizing a custom binary format loaded via memory mapping (`mmap`) and a highly optimized multi-threaded prediction engine (OpenMP), it completely bypasses the parsing and prediction bottlenecks of standard frameworks.

## Documentation & Guides

The unified, comprehensive documentation for both ecosystems is available on GitHub Pages:
**[Read the Full Documentation](https://misterekkk.github.io/lignum/)**

---

## Benchmarks

The following benchmarks show direct head-to-head comparisons on a synthetic dataset. 

### Benchmark Setup
* **Dataset:** 1,000,000 samples, 20 features (15 informative) with a **5% random missing value (NaN) rate**.
* **Model:** 1,000 trees with a maximum depth of 6.
* **Hardware:** Multi-threaded execution scaling across all available CPU cores. Ran on MacBook Pro M4 16GB RAM.

### Head-to-Head Comparisons

#### Lignum vs LightGBM
| Framework | Execution Time (s) | Speedup | Match Accuracy |
| :--- | :---: | :---: | :---: |
| LightGBM | 4.5768 s | 1.0x (Baseline) | — |
| **Lignum** | **0.7082 s** | **6.46x faster** | **100% Perfect Match** |

#### Lignum vs XGBoost
| Framework | Execution Time (s) | Speedup | Match Accuracy |
| :--- | :---: | :---: | :---: |
| XGBoost | 1.5921 s | 1.0x (Baseline) | — |
| **Lignum** | **0.8168 s** | **1.95x faster** | **100% Perfect Match** |

#### Lignum vs Treelite (Compiled)
| Framework | Execution Time (s) | Speedup | Match Accuracy |
| :--- | :---: | :---: | :---: |
| Treelite | 1.4242 s | 1.0x (Baseline) | — |
| **Lignum** | **0.7426 s** | **1.91x faster** | **100% Perfect Match** |

---

## Repository Structure

This repository is managed as a multi-language monorepo:

* **[`R-package/`](./R-package)** – R interface featuring `cpp11` bindings and native `pkgdown` documentation.
* **[`py-package/`](./py-package)** – Python interface featuring `nanobind` extensions optimized for NumPy arrays.
* **[`src_core/`](./src_core)** & **[`include/`](./include)** – Core C++ inference engine shared by both packages.

---

## Quick Local Setup

To quickly compile and install the packages locally from the root directory:

### For R:
```bash
# Install development prerequisites in R, then:
make r-build

```

*For detailed setup, examples, and production mode deployment, see the [R Package README](./R-package/README.md).*

### For Python:

```bash
pip install ./py-package

```

*For NumPy-specific optimization notes and developer guides, see the [Python Package README](./py-package/README.md).*

### Run Tests Locally

You can use the global `Makefile` to trigger tests across both environments simultaneously:

```bash
make test

```
## Acknowledgments

`lignum` achieves extreme parsing speeds by utilizing the [simdjson](https://github.com/simdjson/simdjson) library for parsing JSON model files. We are grateful to the authors and contributors of simdjson for their outstanding work.

---

## License

This project is licensed under the **[MIT License](./LICENSE)**. Both the R and Python packages, along with the core C++ engine, are open-source and free to use under the terms of the MIT license. See the global **[LICENSE](./LICENSE)** file for the full legal text.