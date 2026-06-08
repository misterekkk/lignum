#include "cpp11.hpp"
#include "lignum.hpp"
#include "lignum_types.hpp"

using namespace cpp11;
using namespace lignum;

[[cpp11::register]]
model_ptr_t load_model_cpp(std::string filepath, std::string format) {
    Model* model = new Model(lignum::load(filepath, format));
    return model_ptr_t(model);
}

[[cpp11::register]]
void save_model_cpp(model_ptr_t model, std::string filepath) {
    model->save(filepath);
}

[[cpp11::register]]
doubles predict_cpp(model_ptr_t model, doubles X_flat, int n_samples, int n_features, bool raw_score, int n_jobs) {
    writable::doubles out_preds(n_samples);
    
    const double* x_ptr = REAL(X_flat);
    double* out_ptr = REAL(out_preds);

    if (!raw_score) {
        model->predict(x_ptr, n_samples, n_features, out_ptr, n_jobs);
    } else {
        model->predict_raw(x_ptr, n_samples, n_features, out_ptr, n_jobs);
    }
    
    return out_preds;
}