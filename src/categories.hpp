#pragma once

// Recoding of categorical input to the categories of training (TrainingCategories in model.hpp).
//
// LightGBM and XGBoost 3 store the categories of every pandas `category` column of training; their
// predict() maps an input column's own categories onto them, so a category gets the code it had in
// training whatever order (or subset) the input's categories come in. lignum does the same:
//   1. at load, one hash table per list maps a category to its position (index_categories);
//   2. per input column, a table from the column's categories (pandas categories, Arrow
//      dictionary) to those positions: m lookups for m categories (lookup);
//   3. per row, code -> table[code]: O(1), inside the loop that converts the column anyway.
// That is O(M) once and O(m + n) per column and call, which no method can beat: every category of
// the input and every row must be read once. A table that maps every category to itself (the
// input has the categories of training) is skipped.

#include <lignum/model.hpp>

#include <cstdint>

namespace lignum {

    // Builds c.index; call once the lists are final.
    void index_categories(TrainingCategories& c);

    // Position of `value` in list `list` of c (its code in training), or -1. Equality as pandas
    // compares categories: 1 equals 1.0, text never equals a number, booleans only booleans.
    int32_t lookup(const TrainingCategories& c, size_t list, const CategoryValue& value);

} // namespace lignum
