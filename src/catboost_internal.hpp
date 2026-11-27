#pragma once

#include "catboost.hpp"

namespace lignum::catboost_detail {

    // Runtime AVX2 support check shared across CatBoost kernels and exp helpers.
    bool cpu_has_avx2();

    // Populates derived lookup/prefix/condition layout arrays after loading or building a forest.
    void derive_layout(ObliviousForest& f);

} // namespace lignum::catboost_detail
