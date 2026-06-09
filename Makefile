.PHONY: r-build py-build r-test py-test py-bench test all clean docs

r-build:
	mkdir -p R-package/inst/include R-package/src
	
	rm -f R-package/inst/include/*.hpp R-package/inst/include/simdjson.h
	rm -f R-package/src/builder.cpp R-package/src/io.cpp R-package/src/load.cpp R-package/src/model.cpp R-package/src/parser_lightgbm.cpp R-package/src/parser_xgboost.cpp R-package/src/simdjson.cpp
	
	cp include/*.hpp R-package/inst/include/
	cp extern/simdjson/simdjson.h R-package/inst/include/
	cp src_core/*.cpp R-package/src/
	cp extern/simdjson/simdjson.cpp R-package/src/
	
	Rscript -e "cpp11::cpp_register('R-package')"
	Rscript -e "devtools::document('R-package')"
	R CMD INSTALL R-package

r-test: r-build
	Rscript -e "tinytest::test_all('R-package')"

py-build:
	python -m pip install -e py-package/ --no-cache-dir

py-test: py-build
	python -m pytest py-package/tests/ -v --tb=short

py-bench: py-build
	python benchmarks/bench_lightgbm.py
	python benchmarks/bench_xgboost.py
	python benchmarks/bench_treelite.py
	rm -f benchmarks/bench_model_*

docs:
	rm -rf site/
	Rscript -e "pkgdown::build_site('R-package', override = list(destination = '../site'))"
	python -m pip install mkdocs-material "mkdocstrings[python]"
	python -m mkdocs build

test: r-test py-test
all: r-build py-build

clean:
	rm -f R-package/src/builder.cpp R-package/src/io.cpp R-package/src/load.cpp R-package/src/model.cpp R-package/src/parser_lightgbm.cpp R-package/src/parser_xgboost.cpp R-package/src/simdjson.cpp
	rm -rf R-package/inst/include/
	rm -f R-package/src/*.o R-package/src/*.so R-package/src/*.dll
	rm -rf build/ R-package.Rcheck/ lignum.Rcheck/ py-package/.pytest_cache/ benchmarks/bench_model_* site/