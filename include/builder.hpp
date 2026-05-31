#pragma once

#include "parsers.hpp"
#include "model.hpp"
#include "ir.hpp"

#include <string>

namespace lignum {

    class Builder {
        public:
            static Model load_json(const std::string& filepath, const std::string& model_format);
        
        private:
            static int process_leaf(TempBinNode* node, Model& model);
            static int build_knode(TempBinNode* node, Model& model);
    };

} // namespace lignum