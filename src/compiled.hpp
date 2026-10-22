#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lignum {

    // Experimental compiled mode: the trees of the binned forest become C code (nested ifs on
    // constant thresholds, leaves in per-block tables), built with the system compiler and loaded
    // with dlopen. The generated library bins the numeric features (same codes as the kernel)
    // and walks the trees in blocks: a block adds its trees, in model order, to the running sums
    // of a chunk of rows before the next block runs, which keeps the block's code and branch
    // history hot. Categorical decision bits, threads and transforms stay in lignum, so results
    // are bit-identical.
    struct CompiledForest {
        using BinFn = void (*)(const double* X, size_t stride, size_t n_rows, uint16_t* codes);
        using BlockFn = void (*)(const uint16_t* codes, const uint64_t* decisions, size_t n_rows, double* out);
        BinFn bin = nullptr;        // numeric features of n_rows rows -> codes (n_rows x n_features)
        std::vector<BlockFn> blocks;  // in model order; out is n_rows x n_outputs
        void* handle = nullptr;
        std::string library;
        ~CompiledForest();
    };

    struct CompileOptions {
        std::string compiler = "cc";
        std::string flags = "-O2";
        std::string work_dir;  // empty: system temp directory
        int jobs = 0;          // translation units compiled in parallel; 0 = hardware threads
        int block_trees = 16;  // trees per block function (larger blocks compile slower, run about the same)
    };

    struct CompileStats {
        double codegen_ms = 0, compile_ms = 0, load_ms = 0;
        size_t source_bytes = 0;
        bool cached = false;
        std::string library;
    };

    // Requires the binned kernel (Model::binned). Throws std::runtime_error on compiler errors.
    std::shared_ptr<const CompiledForest> compile_forest(const Model& model, const CompileOptions& options, CompileStats* stats);

} // namespace lignum
