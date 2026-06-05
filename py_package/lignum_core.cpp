#include "lignum.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace lignum;

NB_MODULE(lignum_core, m) {
    nb::class_<Model>(m, "Model")
        .def("predict", [](const Model& self, nb::ndarray<double, nb::c_contig, nb::device::cpu> X, int n_jobs) {
            if (X.ndim() != 2) {
                throw std::invalid_argument("Input array must be 2D.");
            }

            size_t n_samples = X.shape(0);
            size_t n_features = X.shape(1);

            double* out_data = new double[n_samples];

            nb::capsule owner(out_data, [](void *p) noexcept { delete[] static_cast<double*>(p); });

            nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu> out_preds(out_data, { n_samples }, owner);

            self.predict(X.data(), n_samples, n_features, out_data, n_jobs);

            return out_preds;
        },
        nb::arg("X"),
        nb::arg("n_jobs") = -1,
        "Returns model prediction for input data.")
        
        .def("save", &Model::save, nb::arg("filepath"), "Saves lignum format model to binary.");
    
    m.def("load", &lignum::load, nb::arg("filepath"), nb::arg("format") = "auto", "Loads the JSON or binary file format.");
}