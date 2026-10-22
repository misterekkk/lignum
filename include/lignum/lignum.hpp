#pragma once

#include <lignum/arrow.h>
#include <lignum/model.hpp>

#include <string>
#include <vector>

namespace lignum {

    // binned = true prepares the binned kernel (integer comparisons on per-row bin codes,
    // precomputed categorical decisions), used for every prediction when the model supports it;
    // false keeps only the direct KNode traversal. Both give identical results.
    Model load(const std::string& filepath, const std::string& format = "auto", bool binned = true);

    // Arrow input (C Data Interface, no libarrow needed). `batch` is a record batch (struct array)
    // described by `schema`; its first model.n_features columns are the features, by position.
    // Columns may be float64, float32, signed or unsigned integers, boolean or null;
    // dictionary-encoded columns give their indices (category codes, as pandas categoricals give
    // LightGBM and XGBoost). Nulls are NaN. Values become doubles exactly as numpy's
    // astype(float64) makes them, so predictions equal those for the same float64 matrix.
    // out holds batch.length x model.output_width(raw_score) values. Rows are converted in chunks,
    // never the whole batch at once.
    void predict_arrow(const Model& model, const ArrowSchema& schema, const ArrowArray& batch, double* out,
                       bool raw_score = false, int n_jobs = -1, Strategy strategy = Strategy::Auto);

    // Every batch of an Arrow stream, in order; outputs are concatenated (n_rows rows). The
    // stream itself is not released.
    std::vector<double> predict_arrow_stream(const Model& model, ArrowArrayStream& stream, bool raw_score = false,
                                             int n_jobs = -1, Strategy strategy = Strategy::Auto, size_t* n_rows = nullptr);

} // namespace lignum