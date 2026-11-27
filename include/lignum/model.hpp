#pragma once

#if defined(_MSC_VER)
    #define LIGNUM_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define LIGNUM_RESTRICT __restrict__
#else
    #define LIGNUM_RESTRICT
#endif

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>

namespace lignum {

    // Output transforms; the values are part of the binary format.
    namespace transform {
        constexpr uint8_t Sigmoid = 0;
        constexpr uint8_t Exp = 1;
        constexpr uint8_t Identity = 2;
        constexpr uint8_t Log1pExp = 3;
        constexpr uint8_t Softmax = 4;  // across outputs
        constexpr uint8_t Argmax = 5;   // index of the largest margin (XGBoost multi:softmax); one output column
        constexpr uint8_t SignedSquare = 6;  // sign(x) * x * x (LightGBM reg_sqrt: trained on sqrt of the target)
        constexpr uint8_t Hinge = 7;         // 1 if x > 0, else 0 (XGBoost binary:hinge)
        constexpr uint8_t Last = Hinge;      // files with a larger value are rejected
    } // namespace transform

    // Library a model was trained with; the values are part of the binary format.
    namespace model_source {
        constexpr uint8_t Unknown = 0;
        constexpr uint8_t LightGBM = 1;
        constexpr uint8_t XGBoost = 2;
        constexpr uint8_t CatBoost = 3;
        constexpr uint8_t Miter = 4;
    } // namespace model_source

    // How predict() spreads work across threads. All strategies return identical results.
    enum class Strategy : uint8_t {
        Auto,    // pick from the batch size and thread count
        Serial,  // everything on the calling thread; best for one or a few rows
        Rows,    // threads take blocks of rows, sized so every thread gets work
    };

    // Bits of KNode::dirs.
    namespace dir_bits {
        constexpr uint8_t DefaultRight = 1 << 0;   // where missing values go
        constexpr uint8_t ZeroIsMissing = 1 << 1;  // LightGBM missing_type=Zero: |x| <= 1e-35 is missing too
        constexpr uint8_t XGBoostCategory = 1 << 2;  // categorical slot with XGBoost semantics (see slot_kind)
    } // namespace dir_bits

    // Per-tree kinds: the kernel picks a traversal per tree, so special cases only cost
    // in the trees that need them. Values are part of the binary format.
    namespace tree_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t ZeroMissing = 1 << 0;  // has splits with dir_bits::ZeroIsMissing
        constexpr uint8_t Categorical = 1 << 1;  // has categorical splits (see slot_kind)
        constexpr uint8_t Oblique = 1 << 2;      // has oblique splits (miter; see ObliqueSplit)
        constexpr uint8_t Ctr = 1 << 3;          // has native CTR splits (miter; see CtrTableHeader)
        constexpr uint8_t All = ZeroMissing | Categorical | Oblique | Ctr;
    } // namespace tree_kind

    // Kind of each KNode slot. A categorical slot stores its category set in the 64 bits of
    // `thresholds[slot]`. NaN follows dirs & DefaultRight. Otherwise:
    //  - LightGBM: x <= -1 (or >= 2^31) goes right; int(x) in the set goes left.
    //  - XGBoost (dirs & XGBoostCategory): c = int(float(x)); c < 0 or c >= 2^24 goes left;
    //    c in the set goes right.
    namespace slot_kind {
        constexpr uint8_t Numeric = 0;
        constexpr uint8_t CatMask = 1;    // bits: categories 0..63 (bit c set = c in the set)
        constexpr uint8_t CatRange = 2;   // low 32 bits: first category, high 32 bits: last - first
        constexpr uint8_t CatBitset = 3;  // low 32 bits: offset into Model::cat_bitsets, high 32 bits: words
        constexpr uint8_t Oblique = 4;    // 64 bits: index into Model::oblique_splits; `features` is unused
        constexpr uint8_t Ctr = 5;        // `features` is table index in Model::ctr_tables; right iff val >= float(threshold)
    } // namespace slot_kind

    // Per-input-column quantizer of miter models (Q16 affine or quantile):
    //   NaN -> 65535; constant -> 0;
    //   quantile (when quantizer_knot_offsets gives non-empty knots): piecewise linear over intervals;
    //   affine: v = (x - lo) * scale clamped to [0, 65534], rounded via floor(v + 0.5)
    // (bit-identical to miter's Q16Quantizer::encode_u16; needs -ffp-contract=off).
    struct alignas(16) FeatureQuantizer {
        double lo = 0.0;
        double scale = 0.0;
    };
    static_assert(sizeof(FeatureQuantizer) == 16, "FeatureQuantizer is part of the binary format");

    // An oblique split of a miter model (k = 2 or 3 inputs; for k = 2, features[2] repeats
    // features[0] with weight 0): with c_q = code(x_{features[q]}) - 32768,
    //   P = w0 * c0 + w1 * c1 + w2 * c2   (int32; |w| <= 2^14 so no overflow),
    // go right iff (any of the inputs is NaN) ? default direction : P >= threshold.
    // The default direction is KNode::dirs & DefaultRight of the slot.
    struct ObliqueSplit {
        uint32_t features[3] = {0, 0, 0};  // input columns (each has a quantizer)
        int16_t weights[3] = {0, 0, 0};
        uint8_t k = 0;
        uint8_t reserved = 0;
        int32_t threshold = 0;
    };
    static_assert(sizeof(ObliqueSplit) == 24, "ObliqueSplit is part of the binary format");

    // Native online ordered CTR open-addressed hash table (miter):
    // input columns[0..n_cols-1] hold 32-bit cat hashes stored as doubles; keys are combined via
    // combine_cat_hash and looked up with linear probing on Model::ctr_keys/ctr_slots/ctr_values.
    struct alignas(8) CtrTableHeader {
        uint32_t columns[3] = {0, 0, 0};
        uint8_t n_cols = 0;               // 1..3
        uint8_t padding[3] = {0, 0, 0};
        float default_value = 0.0f;
        uint32_t mask = 0;                // n_slots - 1 when n_slots > 0
        uint32_t n_slots = 0;             // 0 (empty table) or power of 2 >= 2
        uint32_t n_values = 0;
        uint64_t keys_offset = 0;         // offset into Model::ctr_keys (n_slots entries)
        uint64_t slots_offset = 0;        // offset into Model::ctr_slots (n_slots entries)
        uint64_t values_offset = 0;       // offset into Model::ctr_values (n_values entries)
    };
    static_assert(sizeof(CtrTableHeader) == 56, "CtrTableHeader is part of the binary format");

    // Two levels of a binary tree in one cache line. Slot 0 is the root split, slots 1 and 2
    // its left and right children; children[0..3] are the four grandchildren (negative = leaf).
    // A numeric split goes right iff  isnan(x) ? (dirs & DefaultRight) : (x >= threshold).
    struct alignas(64) KNode {
        double thresholds[4];  // numeric threshold, or the category set of a categorical slot
        int32_t children[4];
        uint16_t features[4];
        uint8_t dirs[4];
        uint8_t kinds[4];      // slot_kind of each slot
    };
    static_assert(sizeof(KNode) == 64, "KNode must fill exactly one cache line");

    // A category of training, as the library stored it: text, an integer, a number or a boolean.
    struct CategoryValue {
        enum class Kind : uint8_t { Text = 0, Integer = 1, Number = 2, Boolean = 3 };
        Kind kind = Kind::Text;
        std::string text;
        int64_t integer = 0;  // Integer, and Boolean (0 / 1)
        double number = 0.0;  // Number
    };

    // The categories of a model trained on pandas `category` columns (LightGBM's pandas_categorical,
    // XGBoost 3's cats). The library maps the categories of an input DataFrame (or of an Arrow
    // dictionary column) onto these lists before predicting, so that a category gets the code it had
    // in training whatever the input's own order; lignum does the same (recode_table).
    struct TrainingCategories {
        bool by_position = false;       // LightGBM: list k belongs to the k-th categorical column of the input
        bool unseen_is_missing = true;  // LightGBM: a category not in the list is missing; XGBoost: an error
        std::vector<uint32_t> features;  // XGBoost (not by_position): the input column of each list
        std::vector<std::vector<CategoryValue>> lists;  // training order: a category's position is its code
        std::shared_ptr<const void> index;  // value -> position lookups (src/categories.hpp)
    };

    struct BinnedForest;    // binned representation (src/binned.hpp)
    struct CompiledForest;  // experimental compiled trees (src/compiled.hpp)
    struct ObliviousForest; // CatBoost symmetric trees (src/catboost.hpp)

    class Data {
        public:
            virtual ~Data() = default;
    };

    class VectorData : public Data {
        public:
            std::vector<KNode> k_nodes;
            std::vector<int32_t> tree_offsets;
            std::vector<uint8_t> tree_kinds;
            std::vector<uint16_t> tree_outputs;
            std::vector<double> leaf_values;
            std::vector<double> base_scores;
            std::vector<uint32_t> cat_bitsets;
            std::vector<ObliqueSplit> oblique_splits;
            std::vector<FeatureQuantizer> quantizers;
            std::vector<uint8_t> quantizer_constant;
            std::vector<double> quantizer_knots;
            std::vector<uint32_t> quantizer_knot_offsets;
            std::vector<CtrTableHeader> ctr_tables;
            std::vector<uint64_t> ctr_keys;
            std::vector<uint32_t> ctr_slots;
            std::vector<float> ctr_values;
    };

    class Model {
        public:
            const KNode* k_nodes = nullptr;
            size_t n_nodes = 0;

            const int32_t* tree_offsets = nullptr;
            const uint8_t* tree_kinds = nullptr;     // n_offsets entries, see tree_kind
            const uint16_t* tree_outputs = nullptr;  // n_offsets entries when n_outputs > 1: output (class) of each tree
            size_t n_offsets = 0;

            const double* leaf_values = nullptr;
            size_t n_leaves = 0;

            const uint32_t* cat_bitsets = nullptr;  // category sets too large for one slot
            size_t n_cat_words = 0;

            // miter models: oblique splits, per-column Q16 quantizers (affine or quantile knots),
            // and native ordered CTR hash tables.
            const ObliqueSplit* oblique_splits = nullptr;
            size_t n_oblique = 0;
            const FeatureQuantizer* quantizers = nullptr;
            const uint8_t* quantizer_constant = nullptr;  // 1: constant column, code 0
            size_t n_quantizers = 0;
            const double* quantizer_knots = nullptr;      // concatenated quantile knot arrays
            size_t n_quantizer_knots = 0;
            const uint32_t* quantizer_knot_offsets = nullptr;  // n_quantizers + 1 entries when n_quantizer_knots > 0
            const CtrTableHeader* ctr_tables = nullptr;
            size_t n_ctr_tables = 0;
            const uint64_t* ctr_keys = nullptr;
            size_t n_ctr_keys = 0;
            const uint32_t* ctr_slots = nullptr;
            size_t n_ctr_slots = 0;
            const float* ctr_values = nullptr;
            size_t n_ctr_values = 0;

            const double* base_scores = nullptr;  // n_outputs entries, margin space
            uint32_t n_outputs = 1;                // classes for multiclass models, otherwise 1
            double sigmoid = 1.0;
            uint8_t transform = transform::Identity;
            bool float32 = false;           // accumulate and transform in float32 (XGBoost semantics)
            bool has_special_trees = false;  // some tree is not tree_kind::Numeric
            uint32_t n_features = 0;        // minimum number of input columns
            // Names of the training columns, in order (at least n_features of them); empty when the
            // model has none (trained on a matrix: XGBoost and CatBoost store none, LightGBM's
            // Column_0, Column_1, ... are dropped). Used to check named inputs; predict() itself
            // reads columns by position.
            std::vector<std::string> feature_names;
            uint8_t source = model_source::Unknown;
            std::string objective;   // as the source library names it; empty when unknown
            std::string written_by;  // lignum version that wrote the .bin file; empty for other formats
            std::shared_ptr<const TrainingCategories> categories;  // null when the model has none

            std::shared_ptr<const Data> data;
            std::shared_ptr<const BinnedForest> binned;  // binned kernel for batches; null when disabled or unsupported
            std::shared_ptr<const CompiledForest> compiled;  // experimental: trees compiled to native code (needs binned)
            size_t compiled_min_rows = 0;  // batches with fewer rows use the interpreter even when compiled
            std::shared_ptr<const ObliviousForest> oblivious;  // CatBoost model: predicted by its own kernel, no KNodes

            Model() = default;

            // Values per row written by predict (raw_score=false) or predict_raw (raw_score=true).
            // out_preds must hold n_samples * output_width(raw_score) doubles, row-major.
            size_t output_width(bool raw_score) const {
                return (!raw_score && transform == transform::Argmax) ? 1 : n_outputs;
            }

            // Throws std::invalid_argument if n_features is smaller than the model requires.
            void predict(const double* LIGNUM_RESTRICT X, size_t n_samples, size_t n_features, double* LIGNUM_RESTRICT out_preds,
                         int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void predict_raw(const double* LIGNUM_RESTRICT X, size_t n_samples, size_t n_features, double* LIGNUM_RESTRICT out_preds,
                             int n_jobs = -1, Strategy strategy = Strategy::Auto) const;
            void save(const std::string& filepath) const;

            size_t n_trees() const;      // trees in model order, every class counted (CatBoost: symmetric trees)
            size_t total_leaves() const; // leaves of all trees
    };

} // namespace lignum
