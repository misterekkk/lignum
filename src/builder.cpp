#include "builder.hpp"
#include "categories.hpp"
#include "parsers/parsers.hpp"
#include "validate.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace lignum {

    Model Builder::load_model(const std::string& filepath, ModelSource source) {
        ModelInfo info;

        std::vector<std::unique_ptr<TempBinNode>> temp_forest;
        switch (source) {
            case ModelSource::LightGBMJson:  temp_forest = parsers::load_lightgbm_json(filepath, info); break;
            case ModelSource::LightGBMText:  temp_forest = parsers::load_lightgbm_text(filepath, info); break;
            case ModelSource::XGBoostJson:   temp_forest = parsers::load_xgboost_json(filepath, info); break;
            case ModelSource::XGBoostUbjson: temp_forest = parsers::load_xgboost_ubj(filepath, info); break;
        }
        const bool lightgbm = source == ModelSource::LightGBMJson || source == ModelSource::LightGBMText;
        return build(temp_forest, info, lightgbm ? model_source::LightGBM : model_source::XGBoost);
    }

    Model Builder::build(std::vector<std::unique_ptr<TempBinNode>>& temp_forest, ModelInfo& info,
                         uint8_t source, uint32_t min_features) {
        auto data = std::make_shared<VectorData>();
        BuildStats stats;
        data->cat_bitsets.assign(1, 0);  // word 0 is a zero target for branch-free lookups
        for (const auto& root : temp_forest) {
            stats.tree_kind = tree_kind::Numeric;
            data->tree_offsets.push_back(build_knode(root.get(), *data, stats));
            data->tree_kinds.push_back(stats.tree_kind);
            stats.any_special |= stats.tree_kind != tree_kind::Numeric;
        }

        data->base_scores = info.base_scores;
        data->tree_outputs = info.tree_outputs;
        if (data->base_scores.empty() || (data->base_scores.size() > 1 && data->tree_outputs.size() != data->tree_offsets.size())) {
            throw std::runtime_error("Inconsistent multi-output model.");
        }

        Model model;
        model.n_outputs = static_cast<uint32_t>(data->base_scores.size());
        model.base_scores = data->base_scores.data();
        model.tree_outputs = model.n_outputs > 1 ? data->tree_outputs.data() : nullptr;
        model.sigmoid = info.sigmoid;
        model.transform = info.transform;
        model.float32 = info.precision == Precision::Float32;
        model.has_special_trees = stats.any_special;
        model.n_features = std::max<uint32_t>(min_features, static_cast<uint32_t>(stats.max_feature + 1));
        model.n_nodes = data->k_nodes.size();
        model.k_nodes = data->k_nodes.data();
        model.n_offsets = data->tree_offsets.size();
        model.tree_offsets = data->tree_offsets.data();
        model.tree_kinds = data->tree_kinds.data();
        model.n_leaves = data->leaf_values.size();
        model.leaf_values = data->leaf_values.data();
        model.n_cat_words = data->cat_bitsets.size();
        model.cat_bitsets = data->cat_bitsets.data();
        model.data = data;
        model.source = source;
        if (!info.ctr_tables.empty()) {
            data->ctr_tables = std::move(info.ctr_tables);
            data->ctr_keys = std::move(info.ctr_keys);
            data->ctr_slots = std::move(info.ctr_slots);
            data->ctr_values = std::move(info.ctr_values);
            for (const CtrTableHeader& t : data->ctr_tables) {
                for (uint8_t c = 0; c < t.n_cols; c++) {
                    model.n_features = std::max<uint32_t>(model.n_features, t.columns[c] + 1);
                }
            }
            model.ctr_tables = data->ctr_tables.data();
            model.n_ctr_tables = data->ctr_tables.size();
            model.ctr_keys = data->ctr_keys.empty() ? nullptr : data->ctr_keys.data();
            model.n_ctr_keys = data->ctr_keys.size();
            model.ctr_slots = data->ctr_slots.empty() ? nullptr : data->ctr_slots.data();
            model.n_ctr_slots = data->ctr_slots.size();
            model.ctr_values = data->ctr_values.empty() ? nullptr : data->ctr_values.data();
            model.n_ctr_values = data->ctr_values.size();
        }
        if (!data->oblique_splits.empty()) {
            if (info.quantizers.size() < model.n_features || info.quantizer_constant.size() != info.quantizers.size()) {
                throw std::runtime_error("Oblique splits need a quantizer for every input column.");
            }
            data->quantizers = std::move(info.quantizers);
            data->quantizer_constant = std::move(info.quantizer_constant);
            model.n_features = std::max<uint32_t>(model.n_features, static_cast<uint32_t>(data->quantizers.size()));
            model.oblique_splits = data->oblique_splits.data();
            model.n_oblique = data->oblique_splits.size();
            model.quantizers = data->quantizers.data();
            model.quantizer_constant = data->quantizer_constant.data();
            model.n_quantizers = data->quantizers.size();
            if (!info.quantizer_knots.empty()) {
                if (info.quantizer_knot_offsets.size() != data->quantizers.size() + 1) {
                    throw std::runtime_error("Quantile knot offsets must have n_quantizers + 1 entries.");
                }
                data->quantizer_knots = std::move(info.quantizer_knots);
                data->quantizer_knot_offsets = std::move(info.quantizer_knot_offsets);
                model.quantizer_knots = data->quantizer_knots.data();
                model.n_quantizer_knots = data->quantizer_knots.size();
                model.quantizer_knot_offsets = data->quantizer_knot_offsets.data();
            }
        }
        model.feature_names = model_feature_names(std::move(info.feature_names), model.n_features, model.source);
        model.objective = std::move(info.objective);
        if (info.categories) {
            // XGBoost keeps a list for every categorical column of training; the input needs only
            // the first n_features columns, so lists beyond them never apply.
            TrainingCategories& c = *info.categories;
            if (!c.by_position) {
                for (size_t k = c.lists.size(); k-- > 0;) {
                    if (c.features[k] >= model.n_features) {
                        c.lists.erase(c.lists.begin() + static_cast<ptrdiff_t>(k));
                        c.features.erase(c.features.begin() + static_cast<ptrdiff_t>(k));
                    }
                }
            }
            if (!c.lists.empty()) {
                index_categories(c);
                model.categories = std::move(info.categories);
            }
        }

        return model;
    }

    int32_t Builder::process_leaf(TempBinNode* node, VectorData& data) {
        data.leaf_values.push_back(node->threshold_or_value);
        return -static_cast<int32_t>(data.leaf_values.size());
    }

    void Builder::fill_slot(KNode& kn, int slot, const TempBinNode* split, VectorData& data, BuildStats& stats) {
        if (!split->oblique && (split->feature_id < 0 || split->feature_id > 65535)) {
            throw std::out_of_range("Features are stored as uint16_t.");
        }
        if (split->oblique) {
            fill_oblique(kn, slot, split, data, stats);
            return;
        }
        if (split->ctr) {
            fill_ctr(kn, slot, split, stats);
            return;
        }
        kn.features[slot] = static_cast<uint16_t>(split->feature_id);
        kn.dirs[slot] = static_cast<uint8_t>((split->default_dir ? dir_bits::DefaultRight : 0) |
                                             (split->zero_is_missing ? dir_bits::ZeroIsMissing : 0) |
                                             (split->cat_set_goes_right ? dir_bits::XGBoostCategory : 0));
        if (split->categorical) {
            fill_categorical(kn, slot, split, data);
            stats.tree_kind |= tree_kind::Categorical;
        } else {
            kn.thresholds[slot] = split->threshold_or_value;
            kn.kinds[slot] = slot_kind::Numeric;
        }
        stats.max_feature = std::max(stats.max_feature, split->feature_id);
        if (split->zero_is_missing) stats.tree_kind |= tree_kind::ZeroMissing;
    }

    // An oblique slot keeps the index of its ObliqueSplit in the 64 bits of thresholds[slot];
    // `features[slot]` is 0 (never read for this slot kind).
    void Builder::fill_oblique(KNode& kn, int slot, const TempBinNode* split, VectorData& data, BuildStats& stats) {
        const ObliqueSplit& o = split->oblique_split;
        if (o.k < 2 || o.k > 3) throw std::invalid_argument("Oblique splits have 2 or 3 inputs.");
        kn.features[slot] = 0;
        kn.dirs[slot] = static_cast<uint8_t>(split->default_dir ? dir_bits::DefaultRight : 0);
        kn.kinds[slot] = slot_kind::Oblique;
        kn.thresholds[slot] = std::bit_cast<double>(static_cast<uint64_t>(data.oblique_splits.size()));
        data.oblique_splits.push_back(o);
        for (uint32_t f : o.features) {
            if (f > 0x7FFFFFFFu) throw std::out_of_range("Oblique input column out of range.");
            stats.max_feature = std::max(stats.max_feature, static_cast<int32_t>(f));
        }
        stats.tree_kind |= tree_kind::Oblique;
    }

    // A CTR slot keeps the CTR table index in `features[slot]` and the float threshold widened to double.
    void Builder::fill_ctr(KNode& kn, int slot, const TempBinNode* split, BuildStats& stats) {
        kn.features[slot] = static_cast<uint16_t>(split->feature_id);
        kn.dirs[slot] = 0;
        kn.kinds[slot] = slot_kind::Ctr;
        kn.thresholds[slot] = static_cast<double>(static_cast<float>(split->threshold_or_value));
        stats.tree_kind |= tree_kind::Ctr;
    }

    // Picks the cheapest exact encoding of a category set: a 64-bit mask when every category
    // is below 64, a [first, last] range when the set is contiguous, otherwise a bitset.
    void Builder::fill_categorical(KNode& kn, int slot, const TempBinNode* split, VectorData& data) {
        std::vector<uint32_t> cats = split->categories;
        std::sort(cats.begin(), cats.end());
        cats.erase(std::unique(cats.begin(), cats.end()), cats.end());

        uint64_t payload = 0;
        if (cats.empty() || cats.back() < 64) {
            kn.kinds[slot] = slot_kind::CatMask;
            for (uint32_t c : cats) payload |= uint64_t{1} << c;
        } else if (cats.back() - cats.front() + 1 == cats.size()) {
            kn.kinds[slot] = slot_kind::CatRange;
            payload = uint64_t{cats.front()} | (uint64_t{cats.back() - cats.front()} << 32);
        } else {
            kn.kinds[slot] = slot_kind::CatBitset;
            const uint64_t n_words = uint64_t{cats.back()} / 32 + 1;
            const uint64_t offset = data.cat_bitsets.size();
            // The bitsets grow with the largest category of each split, not with the file: a small
            // file naming large categories would otherwise allocate gigabytes.
            // About 256 MB: one split with any category LightGBM accepts (below 2^31) fits.
            constexpr uint64_t kMaxWords = (uint64_t{1} << 26) + (uint64_t{1} << 16);
            if (offset + n_words > kMaxWords) {
                throw std::length_error("The categorical splits need more than 256 MB of category sets; use consecutive "
                                        "category codes from 0 (as LightGBM recommends).");
            }
            data.cat_bitsets.resize(offset + n_words, 0);
            for (uint32_t c : cats) data.cat_bitsets[offset + c / 32] |= uint32_t{1} << (c % 32);
            payload = offset | (n_words << 32);
        }
        kn.thresholds[slot] = std::bit_cast<double>(payload);
    }

    int32_t Builder::build_knode(TempBinNode* node, VectorData& data, BuildStats& stats) {
        if (node->is_leaf) return process_leaf(node, data);

        int32_t current_idx = data.k_nodes.size();
        data.k_nodes.push_back(KNode{});
        KNode kn = {};

        fill_slot(kn, 0, node, data, stats);

        TempBinNode* left = node->left.get();
        if (left->is_leaf) {
            int32_t leaf_idx = process_leaf(left, data);
            kn.children[0] = leaf_idx;
            kn.children[1] = leaf_idx;
        } else {
            fill_slot(kn, 1, left, data, stats);
            kn.children[0] = build_knode(left->left.get(), data, stats);
            kn.children[1] = build_knode(left->right.get(), data, stats);
        }

        TempBinNode* right = node->right.get();
        if (right->is_leaf) {
            int32_t leaf_idx = process_leaf(right, data);
            kn.children[2] = leaf_idx;
            kn.children[3] = leaf_idx;
        } else {
            fill_slot(kn, 2, right, data, stats);
            kn.children[2] = build_knode(right->left.get(), data, stats);
            kn.children[3] = build_knode(right->right.get(), data, stats);
        }

        data.k_nodes[current_idx] = kn;

        return current_idx;
    }

} // namespace lignum
