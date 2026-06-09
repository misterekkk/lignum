# lignum

Ultra-fast inference library for decision-tree based models (LightGBM, XGBoost) in R. 

The `lignum` package bypasses the bottlenecks of standard parsers by providing a native C++ engine designed for maximum CPU cache efficiency and instant file loading via memory mapping (`mmap`).

## Key Features

* **Zero-Overhead Architecture:** The custom binary format (`.bin`) is loaded via `mmap`, allowing massive models to be initialized in milliseconds while sharing RAM across multiple processes.
* **Direct JSON Inference:** You can load and predict directly from XGBoost/LightGBM JSON dumps without any mandatory conversion steps.
* **Cache Efficiency:** The package automatically performs an internal transpose of default R matrices (which are Fortran-contiguous) to ensure optimal C-contiguous (row-major) layout for the CPU cache.
* **Multi-threading:** Native OpenMP support (`n_jobs`) allows predictions to scale seamlessly across all available CPU cores.

## Installation

Since the package is not on CRAN, you can install it directly from GitHub or your local clone.

```r
# Option 1: Install from a local clone using devtools
devtools::install("R-package")

# Option 2: Install directly from GitHub
# install.packages("remotes")
remotes::install_github("misterekkk/lignum", subdir = "R-package")

```

## Quick Start

### Scenario A: Direct Prediction from JSON

You can run predictions straight from a model exported by LightGBM/XGBoost. This is ideal for testing and standard data science workflows.

```r
library(lignum)

# 1. Load a trained model from a JSON file
model <- load_model("model_lightgbm.json", format = "lightgbm")

# 2. Prepare test data
X <- matrix(rnorm(1000 * 50), nrow = 1000, ncol = 50)

# 3. Predict using all available CPU cores
preds <- predict(model, X, raw_score = FALSE, n_jobs = -1L)

```

### Scenario B: Production mode via Binary Format

For production environments (e.g., serverless functions, REST APIs) where cold-start time and RAM usage matter, save the model to Lignum's binary format.

```r
library(lignum)

# 1. Load your JSON model and convert it to a binary file
model <- load_model("model_lightgbm.json", format = "lightgbm")
save_model(model, "lignum_model.bin")

# 2. Load via mmap (instantaneous, shares RAM across processes)
fast_model <- load_model("lignum_model.bin", format = "binary")

# 3. Predict
preds <- predict(fast_model, X, n_jobs = -1L)

```