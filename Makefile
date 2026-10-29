.PHONY: venv bench-deps py-build py-test cpp-build cpp-test test bench bench-quick bench-suite docs clean

PYTHON ?= .venv/bin/python
CPP_BUILD_DIR ?= build/cpp

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) numpy pandas scikit-learn lightgbm xgboost catboost pytest nanobind scikit-build-core pyarrow ninja

# Competitor engines for benchmarks/bench.py (lleaves 1.3 needs llvmlite < 0.45).
bench-deps:
	uv pip install --python $(PYTHON) treelite tl2cgen lleaves "llvmlite<0.45" onnxruntime onnxmltools onnxconverter-common

py-build:
	uv pip install --python $(PYTHON) --no-build-isolation -e .

py-test: py-build
	$(PYTHON) -m pytest tests/python -v --tb=short

cpp-build:
	cmake -S . -B $(CPP_BUILD_DIR) -DLIGNUM_BUILD_TESTS=ON
	cmake --build $(CPP_BUILD_DIR) --parallel

cpp-test: cpp-build
	ctest --test-dir $(CPP_BUILD_DIR) --output-on-failure

test: cpp-test py-test

# Compares lignum with native LightGBM/XGBoost, TL2cgen, GTIL, lleaves and ONNX Runtime (see bench-deps).
bench:
	$(PYTHON) benchmarks/bench.py

bench-quick:
	$(PYTHON) benchmarks/bench.py --quick

# The published benchmark (benchmarks/README.md).
bench-suite:
	$(PYTHON) benchmarks/suite.py

# Starlight site in docs/dist (needs Node.js); the Python reference comes from the installed package.
docs: py-build
	cd docs && $(abspath $(PYTHON)) scripts/api_reference.py && $(abspath $(PYTHON)) scripts/release_notes.py && npm ci && npm run build

clean:
	rm -rf build/ .pytest_cache/ benchmarks/.cache docs/dist/
