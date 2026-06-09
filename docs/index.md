# lignum

Ultra-fast inference library for decision-tree based models (LightGBM, XGBoost) in Python. 

The `lignum` package bypasses the bottlenecks of standard parsers by providing a native C++ engine designed for maximum CPU cache efficiency and instant file loading via memory mapping (`mmap`).

## Key Features

* **Zero-Overhead Architecture:** The custom binary format (`.bin`) is loaded via `mmap`, allowing massive models to be initialized in milliseconds while sharing RAM across multiple processes.
* **Direct JSON Inference:** You can load and predict directly from XGBoost/LightGBM JSON dumps without any mandatory conversion steps.
* **Cache Efficiency:** The package is optimized for C-contiguous (row-major) NumPy arrays, ensuring maximum performance for the CPU cache during inference.
* **Multi-threading:** Native OpenMP support (`n_jobs`) allows predictions to scale seamlessly across all available CPU cores.

## Installation

You can install the package directly from the source using `pip`. 

```bash
# Clone the repository
git clone [https://github.com/misterekkk/lignum.git](https://github.com/misterekkk/lignum.git)
cd lignum

# Install the Python package directly from the directory
pip install ./py-package

```

## Quick Start

### Scenario A: Direct Prediction from JSON

You can run predictions straight from a model exported by LightGBM or XGBoost. This is ideal for testing and standard data science workflows.

```python
import numpy as np
import lignum

# 1. Load a trained model from a JSON file
model = lignum.load("model_lightgbm.json", format="lightgbm")

# 2. Prepare test data
# For optimal performance, ensure the array is C-contiguous float64
X = np.random.randn(1000, 50).astype(np.float64)
X = np.ascontiguousarray(X)

# 3. Predict using all available CPU cores
preds = model.predict(X, raw_score=False, n_jobs=-1)

```

### Scenario B: Production mode via Binary Format

For production environments (e.g., serverless functions, REST APIs) where cold-start time and RAM usage matter, save the model to Lignum's binary format.

```python
import numpy as np
import lignum

# 1. Load your JSON model and convert it to a binary file
model = lignum.load("model_lightgbm.json", format="lightgbm")
model.save("lignum_model.bin")

# 2. Load via mmap (instantaneous, shares RAM across processes)
fast_model = lignum.load("lignum_model.bin", format="binary")

# 3. Predict
# (Assuming X is defined as above)
preds = fast_model.predict(X, n_jobs=-1)

```