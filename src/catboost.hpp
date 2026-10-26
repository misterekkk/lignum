#pragma once

// CatBoost models: oblivious (symmetric) trees over float features, one-hot categorical features
// and CTRs (target statistics of categories and their combinations). See ARCHITECTURE.md.

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace lignum {

    // CityHash64 exactly as CatBoost hashes categories with it (util/digest/city.cpp).
    uint64_t city_hash64(const char* s, size_t len);

    // Prediction is two phases per block of rows:
    //  1. every *source* gets a 16-bit bucket per row: float features (number of borders below the
    //     value), one-hot categorical features (1 + index of the matching value, 0 if none) and
    //     CTRs (number of CTR borders below the CTR value);
    //  2. every tree builds its leaf index from bits `bucket >= threshold` (or `== threshold` for
    //     one-hot), one bit per level, and leaves are added in model order.
    struct ObliviousForest : Data {
        struct Float {
            uint32_t column = 0;
            bool nan_as_true = false;  // NaN is above every border (CatBoost AsTrue), otherwise below
            std::vector<float> borders;  // strictly increasing
            std::vector<float> search;   // derived at load: borders, then +inf up to a power of two
        };
        struct OneHot {
            uint32_t cat = 0;  // index into cat_columns
            std::vector<uint32_t> values;  // category hashes
        };
        // The categories and binary conditions whose combination a CTR counts.
        struct Projection {
            std::vector<uint32_t> cats;  // indices into cat_columns, hashed in this order
            std::vector<std::pair<uint32_t, uint16_t>> floats;    // (float source, threshold): bucket >= threshold
            std::vector<std::pair<uint32_t, uint16_t>> one_hots;  // (one-hot source, value): bucket == value
            std::vector<uint32_t> tables;  // tables of CTRs over this combination
        };
        // Learned counts of one projection, looked up by the combination hash (open addressing).
        struct Table {
            uint32_t projection = 0;
            std::vector<uint64_t> keys;   // slot -> combination hash
            std::vector<uint32_t> slots;  // slot -> entry + 1, 0 = empty
            uint64_t mask = 0;
        };
        // A CTR: its bucket for every table entry, computed at load time with CatBoost's float
        // arithmetic, and the bucket of combinations missing from the table.
        struct Ctr {
            uint32_t table = 0;
            std::vector<uint16_t> entry_buckets;
            uint16_t missing_bucket = 0;
        };

        uint32_t n_columns = 0;
        std::vector<Float> floats;           // sources [0, F)
        std::vector<uint32_t> cat_columns;   // categorical features used by the model -> input column
        std::vector<OneHot> one_hots;        // sources [F, F + H)
        std::vector<Projection> projections;
        std::vector<Table> tables;
        std::vector<Ctr> ctrs;               // sources [F + H, F + H + C)

        // Trees: splits [split_begin[t], split_begin[t + 1]) from the lowest bit of the leaf index.
        std::vector<uint32_t> split_begin;
        std::vector<uint16_t> split_source;
        std::vector<uint16_t> split_threshold;
        std::vector<uint8_t> split_equal;    // 1: bucket == threshold (one-hot), 0: bucket >= threshold
        std::vector<uint64_t> leaf_begin;    // first leaf value of each tree
        std::vector<double> leaves;          // per tree: 2^depth leaves x dim outputs

        uint32_t dim = 1;
        double scale = 1.0;
        std::vector<double> bias;  // dim entries

        // Derived at load (derive_layout, not saved): the projections' category lists as a prefix
        // tree in depth-first order, so a prefix shared by several projections is hashed once per
        // row. Node n adds category prefix_cat[n] to the hash of its parent, the latest node one
        // level up; the projections with exactly that category list follow it.
        std::vector<uint32_t> prefix_cat;
        std::vector<uint8_t> prefix_level;              // 0: first category
        std::vector<uint32_t> prefix_projection_begin;  // nodes + 1 entries, into prefix_projections
        std::vector<uint32_t> prefix_projections;
        uint32_t prefix_levels = 0;                     // longest category list

        // Derived at load (derive_layout, not saved), for the single-row path: every split as one
        // byte of a per-row condition array. Source s owns bytes [condition_begin[s],
        // condition_begin[s + 1]): byte j is `bucket >= j + 1`, or `bucket == j + 1` for one-hot
        // sources, up to the largest threshold a split uses on s.
        std::vector<uint32_t> condition_begin;  // sources + 1 entries
        std::vector<uint32_t> split_condition;  // per split
        std::vector<uint16_t> split_range;      // per split: `bucket >= t` is (uint16_t)(bucket - t) <= range
        bool row_conditions = false;            // single rows through condition bytes (few sources per split)
        std::vector<uint32_t> run_end;          // runs of consecutive trees of one depth: last tree + 1
        std::vector<uint8_t> run_depth;         // and their depth

        size_t n_sources() const { return floats.size() + one_hots.size() + ctrs.size(); }
        size_t n_trees() const { return leaf_begin.size(); }
    };

    namespace parsers {
        // CatBoost save_model(format="json") and the native .cbm.
        Model load_catboost_json(const std::string& filepath);
        Model load_catboost_cbm(const std::string& filepath);
    }

    // The forest in lignum's binary format (after the file header), and back; loading validates
    // every index and throws std::runtime_error on a damaged file.
    void save_oblivious(std::ostream& out, const ObliviousForest& forest);
    std::shared_ptr<ObliviousForest> load_oblivious(const char* data, size_t size, size_t offset);

    // exp as CatBoost's predict computes it (Poisson / Tweedie, softmax): on x86 fmath::expd_v, a
    // table-based approximation, elsewhere std::exp. In place.
    void catboost_exp(double* x, size_t n);
    // CatBoost's softmax of each row of dim values, in place.
    void catboost_softmax(double* out, size_t n_rows, size_t dim);

    // Raw scores (scale * sum + bias) of n_samples rows into out (n_samples x dim). Categorical
    // columns hold category hashes (catboost_hash) as doubles.
    void predict_oblivious(const ObliviousForest& forest, const double* X, size_t n_samples, size_t n_features,
                           double* out, int n_threads, Strategy strategy);

} // namespace lignum
