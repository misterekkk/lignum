#include <lignum/lignum.hpp>
#include "compiled.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

namespace nb = nanobind;
using namespace lignum;

namespace {

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
        
        .def("compile", [](Model& self, const std::string& compiler, const std::string& flags, const std::string& work_dir, int jobs, int block_trees, const std::string& output) {
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
            return result;
        }, nb::arg("compiler") = "cc", nb::arg("flags") = "-O2", nb::arg("work_dir") = "", nb::arg("jobs") = 0, nb::arg("block_trees") = 16, nb::arg("output") = "",
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
        .def("save", &Model::save, nb::arg("filepath"), "Saves lignum format model to binary.")
        .def_prop_ro("n_outputs", [](const Model& self) { return self.n_outputs; }, "Number of outputs (classes for multiclass models).")
        .def_prop_ro("uses_binned", [](const Model& self) { return self.binned != nullptr; },
                     "Whether batches use the binned kernel (False when disabled or unsupported by the model).")
        .def_prop_ro("n_features", [](const Model& self) { return self.n_features; }, "Minimum number of input columns.");
    
    m.def("load", &lignum::load, nb::arg("filepath"), nb::arg("format") = "auto", nb::arg("binned") = true,
          "Loads the JSON or binary file format.");
}