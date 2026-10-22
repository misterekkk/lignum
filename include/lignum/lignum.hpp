#pragma once

#include <lignum/model.hpp>

#include <string>

namespace lignum {

    // binned = true prepares the binned kernel (integer comparisons on per-row bin codes,
    // precomputed categorical decisions) used for batches of 8+ rows; false keeps only the
    // direct KNode traversal. Both give identical results.
    Model load(const std::string& filepath, const std::string& format = "auto", bool binned = true);

} // namespace lignum