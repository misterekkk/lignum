#include <lignum/lignum.hpp>
#include "compiled.hpp"
#include "catboost.hpp"
#include "validate.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace nb = nanobind;
using namespace lignum;

namespace {

    template <typename T>
    T* capsule_pointer(const nb::object& capsule, const char* name) {
        void* p = PyCapsule_GetPointer(capsule.ptr(), name);
        if (!p) throw nb::python_error();
        return static_cast<T*>(p);
    }

    // Predictions as a numpy array: 1-D for one value per row, otherwise (n_rows, width).
    nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu> to_numpy(std::vector<double>&& values, size_t n_rows, size_t width) {
        auto* data = new std::vector<double>(std::move(values));
        nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
        size_t shape[2] = {n_rows, width};
        return nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>(data->data(), width == 1 ? 1 : 2, shape, owner);
    }

    // CatBoost hash of one categorical value: strings as UTF-8, integers (Python or numpy) as their
    // decimal text, like CatBoost. Anything else (floats, NaN, None) is rejected, as CatBoost does.
    uint32_t hash_category(PyObject* value) {
        if (PyUnicode_Check(value)) {
            Py_ssize_t size = 0;
            const char* text = PyUnicode_AsUTF8AndSize(value, &size);
            if (!text) throw nb::python_error();
            return catboost_hash(std::string_view(text, static_cast<size_t>(size)));
        }
        if (PyLong_Check(value) || (!PyFloat_Check(value) && PyIndex_Check(value))) {
            nb::object number = nb::steal(PyNumber_Index(value));
            if (!number.is_valid()) throw nb::python_error();
            nb::object text = nb::steal(PyObject_Str(number.ptr()));
            if (!text.is_valid()) throw nb::python_error();
            Py_ssize_t size = 0;
            const char* utf8 = PyUnicode_AsUTF8AndSize(text.ptr(), &size);
            if (!utf8) throw nb::python_error();
            return catboost_hash(std::string_view(utf8, static_cast<size_t>(size)));
        }
        nb::object repr = nb::steal(PyObject_Repr(value));
        const char* text = repr.is_valid() ? PyUnicode_AsUTF8AndSize(repr.ptr(), nullptr) : nullptr;
        if (!text) PyErr_Clear();
        throw std::invalid_argument("Categorical values must be strings or integers, got " + std::string(text ? text : "?") +
                                    " (convert NaN or None to a string, as CatBoost requires).");
    }

    // One cell of a CatBoost input: the hash in a categorical column, otherwise the number (None: NaN).
    double catboost_cell(PyObject* item, bool categorical) {
        if (categorical) return static_cast<double>(hash_category(item));
        if (item == Py_None) return std::numeric_limits<double>::quiet_NaN();
        const double v = PyFloat_AsDouble(item);
        if (v == -1.0 && PyErr_Occurred()) throw nb::python_error();
        return v;
    }

    // Copy rows [r0, r1) of a 2-D numeric pandas block (n_bcols x n_rows) into row-major float64 `out`
    // of width `out_cols`, skipping trailing columns >= out_cols.
    template <typename T>
    void copy_block_2d_range(const void* data, int64_t s0, int64_t s1, const int64_t* locs, size_t n_bcols,
                             size_t r0, size_t r1, size_t out_cols, double* LIGNUM_RESTRICT out) {
        const auto* src = static_cast<const T*>(data);
        for (size_t c = 0; c < n_bcols; c++) {
            const int64_t raw_col = locs[c];
            if (raw_col < 0 || static_cast<size_t>(raw_col) >= out_cols) continue;
            const size_t col = static_cast<size_t>(raw_col);
            const T* col_ptr = src + static_cast<ptrdiff_t>(c) * s0;
            for (size_t r = r0; r < r1; r++) {
                out[r * out_cols + col] = static_cast<double>(col_ptr[static_cast<ptrdiff_t>(r) * s1]);
            }
        }
    }

    // Map rows [r0, r1) of a 1-D Categorical codes array through a float64 hash table (CatBoost).
    // Returns false if any code is negative or out of range (missing value).
    template <typename T>
    bool map_cat_codes_range(const void* data, int64_t stride, size_t r0, size_t r1,
                             const double* LIGNUM_RESTRICT table, size_t table_len,
                             size_t col, size_t out_cols, double* LIGNUM_RESTRICT out) {
        const auto* codes = static_cast<const T*>(data);
        for (size_t r = r0; r < r1; r++) {
            const int64_t code = static_cast<int64_t>(codes[static_cast<ptrdiff_t>(r) * stride]);
            if (code < 0 || static_cast<size_t>(code) >= table_len) return false;
            out[r * out_cols + col] = table[static_cast<size_t>(code)];
        }
        return true;
    }

    // Map rows [r0, r1) of a 1-D Categorical codes array for LightGBM / XGBoost (-1 -> NaN; optional recode table).
    template <typename T>
    void map_coded_range(const void* data, int64_t stride, size_t r0, size_t r1,
                         const double* LIGNUM_RESTRICT table, size_t table_len,
                         size_t col, size_t out_cols, double* LIGNUM_RESTRICT out) {
        const auto* codes = static_cast<const T*>(data);
        const double nan_val = std::numeric_limits<double>::quiet_NaN();
        if (table) {
            for (size_t r = r0; r < r1; r++) {
                const int64_t code = static_cast<int64_t>(codes[static_cast<ptrdiff_t>(r) * stride]);
                out[r * out_cols + col] = (code < 0 || static_cast<size_t>(code) >= table_len) ? nan_val : table[static_cast<size_t>(code)];
            }
        } else {
            for (size_t r = r0; r < r1; r++) {
                const int64_t code = static_cast<int64_t>(codes[static_cast<ptrdiff_t>(r) * stride]);
                out[r * out_cols + col] = code < 0 ? nan_val : static_cast<double>(code);
            }
        }
    }

    struct NumBlockDesc {
        const void* data;
        int64_t s0, s1;
        const int64_t* locs;
        size_t n_bcols;
        uint8_t code;
        uint8_t bits;
    };

    struct CatColDesc {
        const void* data;
        int64_t stride;
        const double* table;
        size_t table_len;
        size_t col;
        uint8_t bits;
    };

    bool numeric_dtype_supported(nb::dlpack::dtype dt) {
        if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) return dt.bits == 64 || dt.bits == 32;
        if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int))   return dt.bits == 64 || dt.bits == 32 || dt.bits == 16 || dt.bits == 8;
        if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt))  return dt.bits == 64 || dt.bits == 32 || dt.bits == 16 || dt.bits == 8;
        if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool))  return dt.bits == 8;
        return false;
    }

    void copy_num_desc_range(const NumBlockDesc& nb_desc, size_t r0, size_t r1, size_t out_cols, double* out) {
        if (nb_desc.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
            if (nb_desc.bits == 64)      copy_block_2d_range<double>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else                         copy_block_2d_range<float>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
        } else if (nb_desc.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int)) {
            if (nb_desc.bits == 64)      copy_block_2d_range<int64_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else if (nb_desc.bits == 32) copy_block_2d_range<int32_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else if (nb_desc.bits == 16) copy_block_2d_range<int16_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else                         copy_block_2d_range<int8_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
        } else if (nb_desc.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt)) {
            if (nb_desc.bits == 64)      copy_block_2d_range<uint64_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else if (nb_desc.bits == 32) copy_block_2d_range<uint32_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else if (nb_desc.bits == 16) copy_block_2d_range<uint16_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
            else                         copy_block_2d_range<uint8_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
        } else {
            copy_block_2d_range<uint8_t>(nb_desc.data, nb_desc.s0, nb_desc.s1, nb_desc.locs, nb_desc.n_bcols, r0, r1, out_cols, out);
        }
    }

    nb::object wrap_raw_matrix(std::unique_ptr<double[]> out_buf, size_t n_rows, size_t out_cols) {
        double* raw = out_buf.release();
        nb::capsule owner(raw, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[2] = {n_rows, out_cols};
        return nb::cast(nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>(raw, 2, shape, owner));
    }

    Strategy parse_strategy(const std::string& name) {
        if (name == "auto") return Strategy::Auto;
        if (name == "serial") return Strategy::Serial;
        if (name == "rows") return Strategy::Rows;
        throw std::invalid_argument("strategy must be one of 'auto', 'serial', 'rows'; got '" + name + "'.");
    }

} // namespace

NB_MODULE(_core, m) {
    nb::class_<Model>(m, "Model")
        // const: read-only arrays too (pandas 3.1 gives read-only views of single-dtype frames).
        .def("predict", [](const Model& self, nb::ndarray<const double, nb::c_contig, nb::device::cpu> X, bool raw_score, int n_jobs, const std::string& strategy_name) {
            const Strategy strategy = parse_strategy(strategy_name);
            if (X.ndim() != 2) {
                throw std::invalid_argument("Input array must be 2D.");
            }

            size_t n_samples = X.shape(0);
            size_t n_features = X.shape(1);
            size_t width = self.output_width(raw_score);

            double* out_data = new double[n_samples * width];

            nb::capsule owner(out_data, [](void *p) noexcept { delete[] static_cast<double*>(p); });

            // 1-D for single-output models (and argmax), 2-D (n_samples, n_outputs) otherwise.
            size_t shape[2] = {n_samples, width};
            nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu> out_preds(out_data, width == 1 ? 1 : 2, shape, owner);

            {
                nb::gil_scoped_release release;

                if (!raw_score) {
                    self.predict(X.data(), n_samples, n_features, out_data, n_jobs, strategy);
                } else {
                    self.predict_raw(X.data(), n_samples, n_features, out_data, n_jobs, strategy);
                }
            }
            
            return out_preds;
        },
        nb::arg("X"),
        nb::arg("raw_score") = false,
        nb::arg("n_jobs") = -1,
        nb::arg("strategy") = "auto",
        "Returns model prediction for input data.")
        
        .def("predict_arrow", [](const Model& self, nb::object schema_capsule, nb::object array_capsule, bool raw_score, int n_jobs,
                                 const std::string& strategy_name, bool validate_features) {
            const Strategy strategy = parse_strategy(strategy_name);
            const auto* schema = capsule_pointer<ArrowSchema>(schema_capsule, "arrow_schema");
            const auto* batch = capsule_pointer<ArrowArray>(array_capsule, "arrow_array");
            const size_t width = self.output_width(raw_score);
            std::vector<double> out(static_cast<size_t>(std::max<int64_t>(batch->length, 0)) * width);
            {
                nb::gil_scoped_release release;
                predict_arrow(self, *schema, *batch, out.data(), raw_score, n_jobs, strategy, validate_features);
            }
            return to_numpy(std::move(out), out.size() / width, width);
        }, nb::arg("schema"), nb::arg("array"), nb::arg("raw_score") = false, nb::arg("n_jobs") = -1, nb::arg("strategy") = "auto",
           nb::arg("validate_features") = false,
           "Predicts an Arrow record batch given as PyCapsules (arrow_schema, arrow_array).")
        .def("predict_arrow_stream", [](const Model& self, nb::object stream_capsule, bool raw_score, int n_jobs, const std::string& strategy_name,
                                        bool validate_features) {
            const Strategy strategy = parse_strategy(strategy_name);
            auto* stream = capsule_pointer<ArrowArrayStream>(stream_capsule, "arrow_array_stream");
            std::vector<double> out;
            size_t n_rows = 0;
            {
                nb::gil_scoped_release release;
                out = predict_arrow_stream(self, *stream, raw_score, n_jobs, strategy, &n_rows, validate_features);
            }
            return to_numpy(std::move(out), n_rows, self.output_width(raw_score));
        }, nb::arg("stream"), nb::arg("raw_score") = false, nb::arg("n_jobs") = -1, nb::arg("strategy") = "auto",
           nb::arg("validate_features") = false,
           "Predicts every batch of an Arrow stream given as a PyCapsule (arrow_array_stream).")
        .def_prop_ro("categorical_features", [](const Model& self) {
            std::vector<uint32_t> columns;
            if (self.oblivious) columns = self.oblivious->cat_columns;
            std::sort(columns.begin(), columns.end());
            return columns;
        }, "Input columns that hold categories (CatBoost models), in increasing order.")
        .def_prop_ro("is_catboost", [](const Model& self) { return self.oblivious != nullptr; })
        .def_prop_ro("has_categorical_splits", [](const Model& self) {
            if (self.oblivious || !self.has_special_trees) return false;
            for (size_t t = 0; t < self.n_offsets; t++) {
                if (self.tree_kinds[t] & tree_kind::Categorical) return true;
            }
            return false;
        }, "Whether some tree splits on categories (LightGBM / XGBoost models).")
        .def_prop_ro("training_categories", [](const Model& self) -> nb::object {
            if (!self.categories) return nb::none();
            const TrainingCategories& c = *self.categories;
            nb::list lists;
            for (const auto& list : c.lists) {
                nb::list values;
                for (const CategoryValue& v : list) {
                    switch (v.kind) {
                        case CategoryValue::Kind::Text:    values.append(nb::str(v.text.data(), v.text.size())); break;
                        case CategoryValue::Kind::Integer: values.append(nb::int_(v.integer)); break;
                        case CategoryValue::Kind::Number:  values.append(nb::float_(v.number)); break;
                        case CategoryValue::Kind::Boolean: values.append(nb::bool_(v.integer != 0)); break;
                    }
                }
                lists.append(values);
            }
            nb::dict d;
            d["by_position"] = c.by_position;
            d["unseen_is_missing"] = c.unseen_is_missing;
            d["features"] = c.features;
            d["lists"] = lists;
            return d;
        }, "Categories of training (pandas category columns), or None: by_position (LightGBM: list k for the k-th "
           "categorical column), unseen_is_missing (LightGBM) or an error (XGBoost), features (XGBoost), lists.")
        .def_prop_ro("feature_names", [](const Model& self) { return self.feature_names; },
                     "Names of the training columns, in order; empty when the model has none.")
        .def_prop_ro("source", [](const Model& self) -> std::string {
            switch (self.source) {
                case model_source::LightGBM: return "lightgbm";
                case model_source::XGBoost: return "xgboost";
                case model_source::CatBoost: return "catboost";
                case model_source::Miter: return "miter";
                default: return "";
            }
        }, "Library the model was trained with.")
        .def("check_feature_names", [](const Model& self, const std::vector<std::string>& columns) { check_feature_names(self, columns); },
             nb::arg("columns"), "Raises ValueError when the columns do not start with the model's feature names.")
        .def("compile", [](Model& self, const std::string& compiler, const std::string& flags, int jobs, int block_trees, const std::string& library) {
            CompileOptions options;
            options.compiler = compiler;
            options.flags = flags;
            options.jobs = jobs;
            options.block_trees = block_trees;
            options.library = library;
            CompileStats stats;
            {
                nb::gil_scoped_release release;
                set_compiled(self, compile_forest(self, options, &stats));
            }
            nb::dict result;
            result["codegen_ms"] = stats.codegen_ms;
            result["compile_ms"] = stats.compile_ms;
            result["load_ms"] = stats.load_ms;
            result["source_bytes"] = stats.source_bytes;
            result["cached"] = stats.cached;
            result["library"] = stats.library;
            result["compiler"] = stats.compiler;
            return result;
        }, nb::arg("compiler") = "auto", nb::arg("flags") = "", nb::arg("jobs") = 0, nb::arg("block_trees") = 16, nb::arg("library") = "",
           "Experimental: compiles the trees to native code with the system compiler.")
        .def("load_compiled", [](Model& self, const std::string& library) {
            std::shared_ptr<const CompiledForest> compiled;
            {
                nb::gil_scoped_release release;
                compiled = load_compiled(self, library);
            }
            set_compiled(self, std::move(compiled));
        }, nb::arg("library"), "Attaches a library built earlier by compile() for this model.")
        .def("decompile", [](Model& self) { set_compiled(self, nullptr); }, "Drops the compiled trees.")
        .def_prop_ro("uses_compiled", [](const Model& self) { return get_compiled(self) != nullptr; })
        .def_prop_ro("compiled_library", [](const Model& self) -> std::optional<std::string> {
            const auto compiled = get_compiled(self);
            if (!compiled) return std::nullopt;
            return compiled->library;
        }, "Path of the compiled library in use, or None.")
        .def_prop_rw("compiled_min_rows", [](const Model& self) { return self.compiled_min_rows; },
                     [](Model& self, size_t rows) { self.compiled_min_rows = rows; },
                     "Batches with fewer rows use the interpreter even when compiled.")
        .def("save", &Model::save, nb::arg("filepath"), nb::call_guard<nb::gil_scoped_release>(), "Saves lignum format model to binary.")
        .def_prop_ro("n_outputs", [](const Model& self) { return self.n_outputs; }, "Number of outputs (classes for multiclass models).")
        .def_prop_ro("uses_binned", [](const Model& self) { return self.binned != nullptr; },
                     "Whether batches use the binned kernel (False when disabled or unsupported by the model).")
        .def_prop_ro("n_features", [](const Model& self) { return self.n_features; }, "Minimum number of input columns.")
        .def_prop_ro("n_trees", &Model::n_trees, "Number of trees, every class counted.")
        .def_prop_ro("n_leaves", &Model::total_leaves, "Number of leaves of all trees.")
        .def_prop_ro("objective", [](const Model& self) { return self.objective; }, "Objective as the source library names it.")
        .def_prop_ro("written_by", [](const Model& self) { return self.written_by; }, "lignum version that wrote the .bin file.")
        .def_prop_ro("transform", [](const Model& self) -> std::string {
            switch (self.transform) {
                case transform::Sigmoid: return "sigmoid";
                case transform::Exp: return "exp";
                case transform::Log1pExp: return "log1p_exp";
                case transform::Softmax: return "softmax";
                case transform::Argmax: return "argmax";
                case transform::SignedSquare: return "signed_square";
                case transform::Hinge: return "hinge";
                default: return "identity";
            }
        }, "Transform from raw scores to predictions.");
    
    m.def("cache_dir", &default_cache_dir, "Folder of the per-user cache of compiled libraries.");
    m.def("clear_cache", [] { return clear_compiled_cache(""); }, "Removes the cached compiled libraries; returns how many.");
    m.def("catboost_hash", [](nb::handle value) { return hash_category(value.ptr()); }, nb::arg("category"),
          "CatBoost's hash of a category (str, or int hashed as its decimal text).");
    m.def("_catboost_hashes", [](nb::handle values) {
        // A list (the limited API has no fast sequence access); items are borrowed references.
        nb::object list = nb::steal(PySequence_List(values.ptr()));
        if (!list.is_valid()) throw nb::python_error();
        const Py_ssize_t n = PyList_Size(list.ptr());
        std::vector<double> hashes(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; i++) hashes[static_cast<size_t>(i)] = static_cast<double>(hash_category(PyList_GetItem(list.ptr(), i)));
        return to_numpy(std::move(hashes), static_cast<size_t>(n), 1);
    }, nb::arg("values"), "CatBoost hashes of a sequence of categories, as float64.");

    // Rows (a sequence of sequences, or one row) to a float64 matrix with category hashes in the
    // categorical columns: one pass in C++ for lists, tuples and small object arrays.
    m.def("_catboost_rows", [](nb::handle rows, const std::vector<uint32_t>& categorical) {
        nb::object outer = nb::steal(PySequence_List(rows.ptr()));
        if (!outer.is_valid()) throw nb::python_error();
        Py_ssize_t n_rows = PyList_Size(outer.ptr());
        // One row: a sequence of scalars.
        const bool one_row = n_rows > 0 && !PySequence_Check(PyList_GetItem(outer.ptr(), 0)) ? true
                           : n_rows > 0 && PyUnicode_Check(PyList_GetItem(outer.ptr(), 0));
        if (one_row) {
            nb::object wrapped = nb::steal(PyList_New(1));
            if (!wrapped.is_valid()) throw nb::python_error();
            PyList_SetItem(wrapped.ptr(), 0, outer.release().ptr());
            outer = wrapped;
            n_rows = 1;
        }
        std::vector<char> is_cat;
        std::vector<double> values;
        size_t n_columns = 0;
        for (Py_ssize_t i = 0; i < n_rows; i++) {
            nb::object row = nb::steal(PySequence_List(PyList_GetItem(outer.ptr(), i)));
            if (!row.is_valid()) throw nb::python_error();
            const size_t width = static_cast<size_t>(PyList_Size(row.ptr()));
            if (i == 0) {
                n_columns = width;
                is_cat.assign(n_columns, 0);
                for (uint32_t c : categorical) if (c < n_columns) is_cat[c] = 1;
                values.reserve(static_cast<size_t>(n_rows) * n_columns);
            } else if (width != n_columns) {
                throw std::invalid_argument("All rows must have the same number of values.");
            }
            for (size_t j = 0; j < n_columns; j++) {
                values.push_back(catboost_cell(PyList_GetItem(row.ptr(), static_cast<Py_ssize_t>(j)), is_cat[j]));
            }
        }
        nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu> out = [&] {
            auto* data = new std::vector<double>(std::move(values));
            nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
            size_t shape[2] = {static_cast<size_t>(n_rows), n_columns};
            return nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>(data->data(), 2, shape, owner);
        }();
        return out;
    }, nb::arg("rows"), nb::arg("categorical"), "Rows with categories as a float64 matrix of CatBoost hashes.");

    // Fast path for CatBoost pandas DataFrames: collects all numeric and Categorical block
    // descriptors, allocates uninitialized float64 output, and fills all columns <= out_cols in a
    // single L1D-resident 64-row tiled pass. Trailing columns >= out_cols are ignored.
    // Returns None if any needed block has an unsupported dtype.
    m.def("_catboost_blocks", [](nb::handle blocks_obj, size_t n_rows, size_t n_cols, size_t n_features,
                                 const std::vector<uint8_t>& is_cat, nb::dict category_cache,
                                 nb::callable get_table, nb::handle columns_obj) -> nb::object {
        nb::object blks = nb::steal(PySequence_List(blocks_obj.ptr()));
        if (!blks.is_valid()) throw nb::python_error();
        const Py_ssize_t n_blks = PyList_Size(blks.ptr());
        const size_t out_cols = std::min(n_cols, n_features);

        std::vector<NumBlockDesc> num_blks;
        std::vector<CatColDesc> cat_cols;
        std::vector<nb::object> keepalive;
        num_blks.reserve(static_cast<size_t>(n_blks));
        cat_cols.reserve(static_cast<size_t>(n_blks));
        keepalive.reserve(static_cast<size_t>(n_blks) * 3);

        for (Py_ssize_t b = 0; b < n_blks; b++) {
            nb::handle blk(PyList_GetItem(blks.ptr(), b));
            nb::object locs_obj = blk.attr("mgr_locs").attr("as_array");
            nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> locs;
            if (!nb::try_cast(locs_obj, locs)) return nb::none();
            const size_t n_bcols = locs.shape(0);
            const int64_t* loc_ptr = locs.data();
            bool any_needed = false;
            for (size_t c = 0; c < n_bcols; c++) {
                if (loc_ptr[c] < 0) throw std::invalid_argument("Invalid negative block column index.");
                const size_t col = static_cast<size_t>(loc_ptr[c]);
                if (col >= n_cols) return nb::none();
                if (col < out_cols) any_needed = true;
            }
            if (!any_needed) continue;  // W10: ignore trailing unused columns regardless of dtype

            nb::object vals = blk.attr("values");

            if (n_bcols == 1 && nb::hasattr(vals, "_codes") && nb::hasattr(vals, "categories")) {
                const size_t col = static_cast<size_t>(loc_ptr[0]);
                if (col >= is_cat.size() || !is_cat[col]) return nb::none();
                nb::object cats = vals.attr("categories");
                nb::object key = nb::steal(PyLong_FromVoidPtr(cats.ptr()));
                if (!key.is_valid()) throw nb::python_error();
                nb::object table_obj;
                if (PyObject* entry = PyDict_GetItem(category_cache.ptr(), key.ptr());
                    entry && PyTuple_Check(entry) && PyTuple_Size(entry) == 2 && PyTuple_GetItem(entry, 0) == cats.ptr()) {
                    table_obj = nb::borrow(PyTuple_GetItem(entry, 1));
                } else {
                    table_obj = get_table(cats);
                }
                nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> table;
                if (!nb::try_cast(table_obj, table)) return nb::none();
                nb::object codes_obj = vals.attr("_codes");
                nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu> codes;
                if (!nb::try_cast(codes_obj, codes) || codes.shape(0) != n_rows) return nb::none();
                const auto dt = codes.dtype();
                if (dt.code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
                    (dt.bits != 8 && dt.bits != 16 && dt.bits != 32 && dt.bits != 64)) return nb::none();
                cat_cols.push_back({codes.data(), codes.stride(0), table.data(), table.shape(0), col, dt.bits});
                keepalive.push_back(std::move(codes_obj));
                keepalive.push_back(std::move(table_obj));
                continue;
            }

            nb::ndarray<nb::ro, nb::ndim<2>, nb::device::cpu> arr;
            if (!nb::try_cast(vals, arr) || arr.shape(0) != n_bcols || arr.shape(1) != n_rows) return nb::none();
            for (size_t c = 0; c < n_bcols; c++) {
                const size_t col = static_cast<size_t>(loc_ptr[c]);
                if (col < out_cols && col < is_cat.size() && is_cat[col]) return nb::none();
            }
            const auto dt = arr.dtype();
            if (!numeric_dtype_supported(dt)) return nb::none();
            num_blks.push_back({arr.data(), arr.stride(0), arr.stride(1), loc_ptr, n_bcols, dt.code, dt.bits});
            keepalive.push_back(std::move(locs_obj));
            keepalive.push_back(std::move(vals));
        }

        std::unique_ptr<double[]> out_buf(new double[n_rows * out_cols]);
        double* out = out_buf.get();
        constexpr size_t kTile = 64;
        for (size_t r0 = 0; r0 < n_rows; r0 += kTile) {
            const size_t r1 = std::min(r0 + kTile, n_rows);
            for (const auto& nd : num_blks) copy_num_desc_range(nd, r0, r1, out_cols, out);
            for (const auto& cc : cat_cols) {
                bool ok = false;
                if (cc.bits == 8)       ok = map_cat_codes_range<int8_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else if (cc.bits == 16) ok = map_cat_codes_range<int16_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else if (cc.bits == 32) ok = map_cat_codes_range<int32_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else                    ok = map_cat_codes_range<int64_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                if (!ok) {
                    nb::object col_name = columns_obj[nb::int_(cc.col)];
                    nb::str r = nb::repr(col_name);
                    throw std::invalid_argument(std::string("Categorical column ") + r.c_str() +
                                                " has missing values; CatBoost needs a category (e.g. the string 'nan') in every row.");
                }
            }
        }
        return wrap_raw_matrix(std::move(out_buf), n_rows, out_cols);
    }, nb::arg("blocks"), nb::arg("n_rows"), nb::arg("n_cols"), nb::arg("n_features"), nb::arg("is_cat"),
       nb::arg("category_cache"), nb::arg("get_table"), nb::arg("columns"),
       "Fast extraction of a pandas DataFrame into a float64 matrix for CatBoost.");

    // Fast path for purely numeric pandas DataFrames (LightGBM, XGBoost, numeric CatBoost):
    // avoids DataFrame.__array__() consolidation overhead (~15-20 us on small batches),
    // ignores trailing columns >= n_features, and fills needed blocks in one 64-row tiled pass.
    m.def("_numeric_blocks", [](nb::handle blocks_obj, size_t n_rows, size_t n_cols, size_t n_features) -> nb::object {
        nb::object blks = nb::steal(PySequence_List(blocks_obj.ptr()));
        if (!blks.is_valid()) throw nb::python_error();
        const Py_ssize_t n_blks = PyList_Size(blks.ptr());
        const size_t out_cols = std::min(n_cols, n_features);

        // Single already row-major float64 block matching out_cols on large batches: let numpy take a zero-copy view.
        if (n_cols == out_cols && out_cols > 0 && n_blks == 1 && n_rows >= 2048) {
            nb::handle blk(PyList_GetItem(blks.ptr(), 0));
            nb::object vals = blk.attr("values");
            nb::ndarray<nb::ro, nb::ndim<2>, nb::device::cpu> arr;
            if (nb::try_cast(vals, arr) &&
                arr.dtype().code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) &&
                arr.dtype().bits == 64 && arr.stride(0) == 1 && arr.stride(1) == static_cast<int64_t>(n_cols)) {
                return nb::none();
            }
        }

        std::vector<NumBlockDesc> num_blks;
        std::vector<nb::object> keepalive;
        num_blks.reserve(static_cast<size_t>(n_blks));
        keepalive.reserve(static_cast<size_t>(n_blks) * 2);

        for (Py_ssize_t b = 0; b < n_blks; b++) {
            nb::handle blk(PyList_GetItem(blks.ptr(), b));
            nb::object locs_obj = blk.attr("mgr_locs").attr("as_array");
            nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> locs;
            if (!nb::try_cast(locs_obj, locs)) return nb::none();
            const size_t n_bcols = locs.shape(0);
            const int64_t* loc_ptr = locs.data();
            bool any_needed = false;
            for (size_t c = 0; c < n_bcols; c++) {
                if (loc_ptr[c] < 0) throw std::invalid_argument("Invalid negative block column index.");
                const size_t col = static_cast<size_t>(loc_ptr[c]);
                if (col >= n_cols) return nb::none();
                if (col < out_cols) any_needed = true;
            }
            if (!any_needed) continue;  // W10: ignore trailing unused columns regardless of dtype

            nb::object vals = blk.attr("values");
            nb::ndarray<nb::ro, nb::ndim<2>, nb::device::cpu> arr;
            if (!nb::try_cast(vals, arr) || arr.shape(0) != n_bcols || arr.shape(1) != n_rows) return nb::none();
            const auto dt = arr.dtype();
            if (!numeric_dtype_supported(dt)) return nb::none();
            num_blks.push_back({arr.data(), arr.stride(0), arr.stride(1), loc_ptr, n_bcols, dt.code, dt.bits});
            keepalive.push_back(std::move(locs_obj));
            keepalive.push_back(std::move(vals));
        }

        std::unique_ptr<double[]> out_buf(new double[n_rows * out_cols]);
        double* out = out_buf.get();
        constexpr size_t kTile = 64;
        for (size_t r0 = 0; r0 < n_rows; r0 += kTile) {
            const size_t r1 = std::min(r0 + kTile, n_rows);
            for (const auto& nd : num_blks) copy_num_desc_range(nd, r0, r1, out_cols, out);
        }
        return wrap_raw_matrix(std::move(out_buf), n_rows, out_cols);
    }, nb::arg("blocks"), nb::arg("n_rows"), nb::arg("n_cols"), nb::arg("n_features"),
       "Fast extraction of a purely numeric pandas DataFrame into a float64 matrix.");

    // Fast path for LightGBM / XGBoost pandas DataFrames with Categorical columns:
    // counts all categorical columns (K5) while ignoring trailing non-categorical or >= out_cols data (W10),
    // allocates uninitialized output of width out_cols, and fills all columns in one 64-row tiled pass.
    m.def("_coded_blocks", [](nb::handle blocks_obj, size_t n_rows, size_t n_cols, size_t n_features,
                              bool has_tc, bool by_position, size_t n_lists, nb::dict where_dict,
                              nb::dict recode_cache, nb::callable get_recode, nb::handle columns_obj) -> nb::object {
        nb::object blks = nb::steal(PySequence_List(blocks_obj.ptr()));
        if (!blks.is_valid()) throw nb::python_error();
        const Py_ssize_t n_blks = PyList_Size(blks.ptr());
        const size_t out_cols = std::min(n_cols, n_features);

        std::vector<std::pair<size_t, Py_ssize_t>> cat_order;
        cat_order.reserve(static_cast<size_t>(n_blks));
        for (Py_ssize_t b = 0; b < n_blks; b++) {
            nb::handle blk(PyList_GetItem(blks.ptr(), b));
            nb::object locs_obj = blk.attr("mgr_locs").attr("as_array");
            nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> locs;
            if (!nb::try_cast(locs_obj, locs)) return nb::none();
            const size_t n_bcols = locs.shape(0);
            const int64_t* loc_ptr = locs.data();
            for (size_t c = 0; c < n_bcols; c++) {
                if (loc_ptr[c] < 0) throw std::invalid_argument("Invalid negative block column index.");
                if (static_cast<size_t>(loc_ptr[c]) >= n_cols) return nb::none();
            }
            nb::object vals = blk.attr("values");
            if (nb::hasattr(vals, "_codes") && nb::hasattr(vals, "categories")) {
                if (n_bcols != 1) return nb::none();
                cat_order.emplace_back(static_cast<size_t>(loc_ptr[0]), b);  // K5: count all category cols 0..n_cols-1
            }
        }
        std::sort(cat_order.begin(), cat_order.end());
        if (has_tc && !cat_order.empty() && by_position && cat_order.size() != n_lists) return nb::none();

        std::vector<int64_t> col_list(out_cols, -1);
        if (has_tc && !cat_order.empty()) {
            if (by_position) {
                for (size_t k = 0; k < cat_order.size(); k++) {
                    const size_t col = cat_order[k].first;
                    if (col < out_cols) col_list[col] = static_cast<int64_t>(k);
                }
            } else {
                for (const auto& [col, _] : cat_order) {
                    if (col >= out_cols) continue;
                    nb::object key = nb::int_(col);
                    PyObject* v = PyDict_GetItem(where_dict.ptr(), key.ptr());
                    if (!v) return nb::none();
                    col_list[col] = PyLong_AsLongLong(v);
                }
            }
        }

        std::vector<NumBlockDesc> num_blks;
        std::vector<CatColDesc> cat_cols;
        std::vector<nb::object> keepalive;
        num_blks.reserve(static_cast<size_t>(n_blks));
        cat_cols.reserve(static_cast<size_t>(n_blks));
        keepalive.reserve(static_cast<size_t>(n_blks) * 3);

        for (Py_ssize_t b = 0; b < n_blks; b++) {
            nb::handle blk(PyList_GetItem(blks.ptr(), b));
            nb::object locs_obj = blk.attr("mgr_locs").attr("as_array");
            nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> locs;
            if (!nb::try_cast(locs_obj, locs)) return nb::none();
            const size_t n_bcols = locs.shape(0);
            const int64_t* loc_ptr = locs.data();
            bool any_needed = false;
            for (size_t c = 0; c < n_bcols; c++) {
                if (static_cast<size_t>(loc_ptr[c]) < out_cols) { any_needed = true; break; }
            }
            if (!any_needed) continue;  // W10: ignore trailing unused columns regardless of dtype

            nb::object vals = blk.attr("values");

            if (n_bcols == 1 && nb::hasattr(vals, "_codes") && nb::hasattr(vals, "categories")) {
                const size_t col = static_cast<size_t>(loc_ptr[0]);
                nb::object codes_obj = vals.attr("_codes");
                nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu> codes;
                if (!nb::try_cast(codes_obj, codes) || codes.shape(0) != n_rows) return nb::none();
                const auto dt = codes.dtype();
                if (dt.code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
                    (dt.bits != 8 && dt.bits != 16 && dt.bits != 32 && dt.bits != 64)) return nb::none();
                const double* table_ptr = nullptr;
                size_t table_len = 0;
                if (col_list[col] >= 0) {
                    nb::object cats = vals.attr("categories");
                    nb::tuple key = nb::make_tuple(nb::steal(PyLong_FromVoidPtr(cats.ptr())), nb::int_(col_list[col]));
                    nb::object table_holder;
                    if (PyObject* entry = PyDict_GetItem(recode_cache.ptr(), key.ptr());
                        entry && PyTuple_Check(entry) && PyTuple_Size(entry) == 2 && PyTuple_GetItem(entry, 0) == cats.ptr()) {
                        table_holder = nb::borrow(PyTuple_GetItem(entry, 1));
                    } else {
                        table_holder = get_recode(cats, nb::int_(col_list[col]), columns_obj[nb::int_(col)]);
                    }
                    if (!table_holder.is_none()) {
                        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> table_arr;
                        if (!nb::try_cast(table_holder, table_arr)) return nb::none();
                        table_ptr = table_arr.data();
                        table_len = table_arr.shape(0);
                        keepalive.push_back(std::move(table_holder));
                    }
                }
                cat_cols.push_back({codes.data(), codes.stride(0), table_ptr, table_len, col, dt.bits});
                keepalive.push_back(std::move(codes_obj));
                continue;
            }

            nb::ndarray<nb::ro, nb::ndim<2>, nb::device::cpu> arr;
            if (!nb::try_cast(vals, arr) || arr.shape(0) != n_bcols || arr.shape(1) != n_rows) return nb::none();
            const auto dt = arr.dtype();
            if (!numeric_dtype_supported(dt)) return nb::none();
            num_blks.push_back({arr.data(), arr.stride(0), arr.stride(1), loc_ptr, n_bcols, dt.code, dt.bits});
            keepalive.push_back(std::move(locs_obj));
            keepalive.push_back(std::move(vals));
        }

        std::unique_ptr<double[]> out_buf(new double[n_rows * out_cols]);
        double* out = out_buf.get();
        constexpr size_t kTile = 64;
        for (size_t r0 = 0; r0 < n_rows; r0 += kTile) {
            const size_t r1 = std::min(r0 + kTile, n_rows);
            for (const auto& nd : num_blks) copy_num_desc_range(nd, r0, r1, out_cols, out);
            for (const auto& cc : cat_cols) {
                if (cc.bits == 8)       map_coded_range<int8_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else if (cc.bits == 16) map_coded_range<int16_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else if (cc.bits == 32) map_coded_range<int32_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
                else                    map_coded_range<int64_t>(cc.data, cc.stride, r0, r1, cc.table, cc.table_len, cc.col, out_cols, out);
            }
        }
        return wrap_raw_matrix(std::move(out_buf), n_rows, out_cols);
    }, nb::arg("blocks"), nb::arg("n_rows"), nb::arg("n_cols"), nb::arg("n_features"),
       nb::arg("has_tc"), nb::arg("by_position"), nb::arg("n_lists"), nb::arg("where_dict"),
       nb::arg("recode_cache"), nb::arg("get_recode"), nb::arg("columns"),
       "Fast extraction of a pandas DataFrame with Categorical columns for LightGBM / XGBoost.");

    m.def("load", &lignum::load, nb::arg("filepath"), nb::arg("format") = "auto", nb::arg("binned") = true,
          nb::call_guard<nb::gil_scoped_release>(), "Loads the JSON or binary file format.");
}