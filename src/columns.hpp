#pragma once

#include <lignum/model.hpp>

#include <cstddef>

namespace lignum {

    // Columnar input for the binned kernel (e.g. Arrow): it bins column by column straight from
    // the source instead of reading a row-major matrix. read() is called concurrently from
    // several threads.
    struct ColumnSource {
        virtual ~ColumnSource() = default;
        // Input column `column` for rows [begin, begin + count) as doubles (NaN = missing),
        // converted exactly as numpy's astype(float64) would.
        virtual void read(size_t column, size_t begin, size_t count, double* dst) const = 0;
    };

    // Like Model::predict / predict_raw on the equivalent float64 matrix (n_columns columns).
    // Requires the binned kernel (model.binned).
    void predict_columns(const Model& model, const ColumnSource& source, size_t n_rows, size_t n_columns, double* out,
                         bool raw_score, int n_jobs, Strategy strategy);

} // namespace lignum
