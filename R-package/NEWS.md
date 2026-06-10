# lignum 0.1.0

* **Core Engine:** Added native C++ inference engine for decision-tree models.
* **Format Support:** Integrated `simdjson` for fast parsing of LightGBM and XGBoost JSON models, alongside a custom binary format loaded via `mmap`.
* **Performance:** Implemented multi-threaded prediction support powered by OpenMP.
* **R Package:** Created R interface featuring `cpp11` bindings and comprehensive `pkgdown` documentation.
* **Python Package:** Created Python interface featuring `nanobind` extensions optimized for NumPy arrays.
