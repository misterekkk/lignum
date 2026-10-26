#include <lignum/lignum.hpp>
#include "compiled.hpp"
#include "catboost.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>
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

    Strategy parse_strategy(const std::string& name) {
        if (name == "auto") return Strategy::Auto;
        if (name == "serial") return Strategy::Serial;
        if (name == "rows") return Strategy::Rows;
        throw std::invalid_argument("strategy must be one of 'auto', 'serial', 'rows'; got '" + name + "'.");
    }

} // namespace

NB_MODULE(_core, m) {
    nb::class_<Model>(m, "Model")
        .def("predict", [](const Model& self, nb::ndarray<double, nb::c_contig, nb::device::cpu> X, bool raw_score, int n_jobs, const std::string& strategy_name) {
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
                                 const std::string& strategy_name) {
            const Strategy strategy = parse_strategy(strategy_name);
            const auto* schema = capsule_pointer<ArrowSchema>(schema_capsule, "arrow_schema");
            const auto* batch = capsule_pointer<ArrowArray>(array_capsule, "arrow_array");
            const size_t width = self.output_width(raw_score);
            std::vector<double> out(static_cast<size_t>(std::max<int64_t>(batch->length, 0)) * width);
            {
                nb::gil_scoped_release release;
                predict_arrow(self, *schema, *batch, out.data(), raw_score, n_jobs, strategy);
            }
            return to_numpy(std::move(out), out.size() / width, width);
        }, nb::arg("schema"), nb::arg("array"), nb::arg("raw_score") = false, nb::arg("n_jobs") = -1, nb::arg("strategy") = "auto",
           "Predicts an Arrow record batch given as PyCapsules (arrow_schema, arrow_array).")
        .def("predict_arrow_stream", [](const Model& self, nb::object stream_capsule, bool raw_score, int n_jobs, const std::string& strategy_name) {
            const Strategy strategy = parse_strategy(strategy_name);
            auto* stream = capsule_pointer<ArrowArrayStream>(stream_capsule, "arrow_array_stream");
            std::vector<double> out;
            size_t n_rows = 0;
            {
                nb::gil_scoped_release release;
                out = predict_arrow_stream(self, *stream, raw_score, n_jobs, strategy, &n_rows);
            }
            return to_numpy(std::move(out), n_rows, self.output_width(raw_score));
        }, nb::arg("stream"), nb::arg("raw_score") = false, nb::arg("n_jobs") = -1, nb::arg("strategy") = "auto",
           "Predicts every batch of an Arrow stream given as a PyCapsule (arrow_array_stream).")
        .def_prop_ro("categorical_features", [](const Model& self) {
            std::vector<uint32_t> columns;
            if (self.oblivious) columns = self.oblivious->cat_columns;
            std::sort(columns.begin(), columns.end());
            return columns;
        }, "Input columns that hold categories (CatBoost models), in increasing order.")
        .def_prop_ro("is_catboost", [](const Model& self) { return self.oblivious != nullptr; })
        .def("compile", [](Model& self, const std::string& compiler, const std::string& flags, const std::string& work_dir, int jobs, int block_trees, const std::string& output) {
            if (self.oblivious) {
                throw std::runtime_error("CatBoost models have no compiled mode: their symmetric trees are branch-free already.");
            }
            CompileOptions options;
            options.compiler = compiler;
            options.flags = flags;
            options.work_dir = work_dir;
            options.jobs = jobs;
            options.block_trees = block_trees;
            options.output = output;
            CompileStats stats;
            {
                nb::gil_scoped_release release;
                self.compiled = compile_forest(self, options, &stats);
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
        }, nb::arg("compiler") = "auto", nb::arg("flags") = "", nb::arg("work_dir") = "", nb::arg("jobs") = 0, nb::arg("block_trees") = 16, nb::arg("output") = "",
           "Experimental: compiles the trees to native code with the system compiler.")
        .def("load_compiled", [](Model& self, const std::string& library) {
            std::shared_ptr<const CompiledForest> compiled;
            {
                nb::gil_scoped_release release;
                compiled = load_compiled(self, library);
            }
            self.compiled = compiled;
        }, nb::arg("library"), "Attaches a library built earlier by compile() for this model.")
        .def("decompile", [](Model& self) { self.compiled = nullptr; }, "Drops the compiled trees.")
        .def_prop_ro("uses_compiled", [](const Model& self) { return self.compiled != nullptr; })
        .def_prop_rw("compiled_min_rows", [](const Model& self) { return self.compiled_min_rows; },
                     [](Model& self, size_t rows) { self.compiled_min_rows = rows; },
                     "Batches with fewer rows use the interpreter even when compiled.")
        .def("save", &Model::save, nb::arg("filepath"), "Saves lignum format model to binary.")
        .def_prop_ro("n_outputs", [](const Model& self) { return self.n_outputs; }, "Number of outputs (classes for multiclass models).")
        .def_prop_ro("uses_binned", [](const Model& self) { return self.binned != nullptr; },
                     "Whether batches use the binned kernel (False when disabled or unsupported by the model).")
        .def_prop_ro("n_features", [](const Model& self) { return self.n_features; }, "Minimum number of input columns.");
    
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

    m.def("load", &lignum::load, nb::arg("filepath"), nb::arg("format") = "auto", nb::arg("binned") = true,
          "Loads the JSON or binary file format.");
}