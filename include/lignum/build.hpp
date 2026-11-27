#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lignum {

    // Building a model in memory (no file), e.g. from a training library that links lignum.
    // Semantics are lignum's normalised rules (ARCHITECTURE.md "Split semantics"):
    //   Numeric:     go right iff isnan(x) ? default_right : x >= threshold
    //   Categorical: LightGBM semantics on integer codes: int(x) in `categories` goes left,
    //                everything else (and NaN) right
    //   Oblique:     see ObliqueSplit; needs a quantizer for every column it reads
    //   Ctr:         `feature` is CTR table index in Forest::ctr_tables; right iff val >= float(value)
    namespace ir {

        struct CtrTable {
            std::vector<uint32_t> columns;  // 1..3 input columns carrying 32-bit cat hashes as double
            float default_value = 0.0f;
            uint32_t mask = 0;
            std::vector<uint64_t> keys;     // power-of-2 size (or empty)
            std::vector<uint32_t> slots;    // 0 = empty, 1..values.size() = 1-based index into values
            std::vector<float> values;
        };

        struct Node {
            enum class Kind : uint8_t { Leaf = 0, Numeric = 1, Categorical = 2, Oblique = 3, Ctr = 4 };
            Kind kind = Kind::Leaf;
            bool default_right = false;
            uint32_t left = 0;   // child node indices within the tree (splits only)
            uint32_t right = 0;
            uint32_t feature = 0;              // Numeric / Categorical: input column; Ctr: CTR table index
            double value = 0.0;                // Numeric / Ctr: threshold; Leaf: leaf value
            std::vector<uint32_t> categories;  // Categorical: the set going left
            ObliqueSplit oblique;              // Oblique
        };

        struct Tree {
            std::vector<Node> nodes;  // node 0 is the root; children must come after their parent
        };

        struct Forest {
            std::vector<Tree> trees;   // model order
            double base_score = 0.0;   // single-output margin space; overridden when base_scores is non-empty
            std::vector<double> base_scores;    // multi-output (K classes): K entries in margin space
            std::vector<uint16_t> tree_outputs; // output (class < K) of each tree when base_scores.size() > 1
            uint8_t transform = transform::Identity;  // Identity, Sigmoid, Softmax, Exp, Log1pExp, Argmax
            uint32_t n_features = 0;   // input columns (at least every column a split reads)
            std::vector<FeatureQuantizer> quantizers;          // n_features entries when there are oblique splits
            std::vector<uint8_t> quantizer_constant;           // n_features entries (1: constant column)
            std::vector<std::vector<double>> quantizer_knots;  // optional per-feature quantile knots (2..257 doubles; empty = affine)
            std::vector<CtrTable> ctr_tables;                  // optional native online ordered CTR tables
            std::vector<std::string> feature_names;            // optional
            std::string objective;                             // the training library's name for it
            uint8_t source = model_source::Miter;
        };

    } // namespace ir

    // Validates the forest (indices, children after parents, quantizers, CTR tables) and
    // builds a model; binned = true also prepares the binned kernel when the model supports it.
    // Throws std::invalid_argument on an invalid forest.
    Model build_model(const ir::Forest& forest, bool binned = true);

} // namespace lignum
