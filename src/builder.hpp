#pragma once

#include <lignum/model.hpp>
#include "ir.hpp"

#include <string>

namespace lignum {

    class Builder {
        public:
            static Model load_json(const std::string& filepath, const std::string& model_format);

        private:
            struct BuildStats {
                int32_t max_feature = -1;
                bool any_special = false;  // some tree is not tree_kind::Numeric
                uint8_t tree_kind = 0;     // tree_kind bits of the tree being built
            };

            static int32_t process_leaf(TempBinNode* node, VectorData& data);
            static void fill_slot(KNode& kn, int slot, const TempBinNode* split, VectorData& data, BuildStats& stats);
            static void fill_categorical(KNode& kn, int slot, const TempBinNode* split, VectorData& data);
            static int32_t build_knode(TempBinNode* node, VectorData& data, BuildStats& stats);
    };

} // namespace lignum
