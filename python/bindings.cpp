#include <lignum/lignum.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

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
        
        .def("save", &Model::save, nb::arg("filepath"), "Saves lignum format model to binary.")
        .def_prop_ro("n_outputs", [](const Model& self) { return self.n_outputs; }, "Number of outputs (classes for multiclass models).")
        .def_prop_ro("uses_binned", [](const Model& self) { return self.binned != nullptr; },
                     "Whether batches use the binned kernel (False when disabled or unsupported by the model).")
        .def_prop_ro("n_features", [](const Model& self) { return self.n_features; }, "Minimum number of input columns.");
    
    m.def("load", &lignum::load, nb::arg("filepath"), nb::arg("format") = "auto", nb::arg("binned") = true,
          "Loads the JSON or binary file format.");
}