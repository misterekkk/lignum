<p align="center"><img src="docs/assets/logo.svg" width="112" alt="lignum logo"></p>

# Lignum: Ultra-Fast Decision Tree Inference

```bash
pip install pylignum   # imported as `import lignum`
```

[![Tests](https://github.com/misterekkk/lignum/actions/workflows/tests.yml/badge.svg)](https://github.com/misterekkk/lignum/actions/workflows/tests.yml)
[![Deploy Documentation](https://github.com/misterekkk/lignum/actions/workflows/docs.yml/badge.svg)](https://github.com/misterekkk/lignum/actions/workflows/docs.yml)

`lignum` is a high-performance inference library for decision-tree models (LightGBM, XGBoost) written in native C++. By utilizing a custom binary format loaded via memory mapping (`mmap`) and a highly optimized multi-threaded prediction engine (OpenMP), it completely bypasses the parsing and prediction bottlenecks of standard frameworks.

## Documentation & Guides

The documentation is available on GitHub Pages:
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

* **[`include/lignum/`](./include/lignum)** – public C++ API (`lignum.hpp`, `model.hpp`).
* **[`src/`](./src)** – core engine: JSON parsers, IR builder, prediction kernel, binary I/O.
* **[`python/`](./python)** – Python package with `nanobind` bindings.
* **[`tests/`](./tests)** – GoogleTest suite (`tests/cpp`) and parity tests against LightGBM/XGBoost (`tests/python`).
* **[`benchmarks/`](./benchmarks)** – benchmark scripts.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the design.

---

## Quick Local Setup

```bash
make venv       # Python 3.12 virtualenv via uv, with test dependencies
make test       # C++ (GoogleTest) + Python parity tests
```

Or install the Python package directly:

```bash
pip install .
```

On macOS, OpenMP requires `brew install libomp`.

---

## Acknowledgments

`lignum` achieves extreme parsing speeds by utilizing the [simdjson](https://github.com/simdjson/simdjson) library for parsing JSON model files. We are grateful to the authors and contributors of simdjson for their outstanding work.

---

## License

This project is licensed under the **[MIT License](./LICENSE)**. The Python package and the core C++ engine are open-source and free to use under the terms of the MIT license. See the global **[LICENSE](./LICENSE)** file for the full legal text.