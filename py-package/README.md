# lignum

Python bindings for the `lignum` ultra-fast decision tree inference engine. This package leverages `nanobind` and `scikit-build-core` to expose a high-performance C++ prediction pipeline directly to NumPy.

## Key Features

* **Instant Loading:** Bypasses standard JSON parsing overhead by utilizing memory-mapped files (`mmap`) for custom `.bin` models.
* **NumPy Integration:** Built from the ground up to support C-contiguous `float64` NumPy arrays, maximizing CPU cache efficiency.
* **Multi-threaded Inference:** Native OpenMP parallelization scales predictions seamlessly across available CPU cores.

## Installation

### Prerequisites
* C++17 compliant compiler (GCC, Clang, or MSVC)
* CMake (>= 3.18)
* Python (>= 3.8)

### Installing from Source
From the root directory of the repository, run:

```bash
pip install ./py-package

```

For local development and running benchmarks, install in editable mode with development dependencies:

```bash
pip install -e ./py-package

```

## Quick Start

```python
import numpy as np
import lignum

# 1. Load a model (supports 'lightgbm', 'xgboost', 'binary', or 'auto')
model = lignum.load("model_lightgbm.json", format="lightgbm")

# 2. Prepare C-contiguous float64 input data for optimal performance
X = np.random.randn(1000, 50).astype(np.float64)
X = np.ascontiguousarray(X)

# 3. Generate predictions using all CPU cores
preds = model.predict(X, n_jobs=-1)

# 4. Save to highly optimized Lignum binary format for production
model.save("lignum_model.bin")

```

## Running Tests

To verify the installation and core functionality, run `pytest` from the `py-package` directory:

```bash
pytest tests/ -v

```
