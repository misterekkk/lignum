.PHONY: venv py-build py-test cpp-build cpp-test test bench docs clean

PYTHON ?= .venv/bin/python
CPP_BUILD_DIR ?= build/cpp

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) numpy pandas scikit-learn lightgbm xgboost pytest nanobind scikit-build-core pyarrow ninja

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

# Benchmarks use a machine-tuned build; never ship it.
bench:
	CMAKE_ARGS="-DLIGNUM_NATIVE=ON" uv pip install --python $(PYTHON) --no-build-isolation -e .
	$(PYTHON) benchmarks/bench_lightgbm.py
	$(PYTHON) benchmarks/bench_xgboost.py

docs:
	rm -rf site/
	$(PYTHON) -m pip install mkdocs-material "mkdocstrings[python]"
	$(PYTHON) -m mkdocs build

clean:
	rm -rf build/ .pytest_cache/ benchmarks/bench_model_* site/
