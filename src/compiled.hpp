#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lignum {

    // Experimental compiled mode: the trees of the binned forest become C code (nested ifs on
    // constant thresholds, leaves in per-block tables), built with the system compiler and loaded
    // as a shared library (dlopen, or LoadLibrary on Windows). The generated library bins the numeric features (same codes as the kernel)
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
        std::string compiler = "auto";  // auto: cc on POSIX; on Windows cl, Visual Studio (vswhere), gcc, clang
        std::string flags;              // empty: -O2 (gcc/clang) or /O2 (cl)
        std::string work_dir;  // empty: system temp directory
        std::string output;    // also copy the library here (to ship it next to the model file)
        int jobs = 0;          // translation units compiled in parallel; 0 = hardware threads
        int block_trees = 16;  // trees per block function (larger blocks compile slower, run about the same)
    };

    struct CompileStats {
        double codegen_ms = 0, compile_ms = 0, load_ms = 0;
        size_t source_bytes = 0;
        bool cached = false;
        std::string library;
        std::string compiler;  // toolchain used (empty when the library came from the cache)
    };

    // Requires the binned kernel (Model::binned). Throws std::runtime_error when the compiler is
    // missing or fails.
    std::shared_ptr<const CompiledForest> compile_forest(const Model& model, const CompileOptions& options, CompileStats* stats);

    // Attaches a library built earlier by compile_forest (e.g. shipped next to the model file).
    // The library records the fingerprint of the model it was generated from and the ABI of the
    // generated functions; a library for another model or lignum version is rejected with
    // std::runtime_error. Loading runs the library's code: only load libraries you trust.
    std::shared_ptr<const CompiledForest> load_compiled(const Model& model, const std::string& library);

    // Stable 64-bit hash of everything the generated code depends on (trees, cuts, leaves,
    // outputs, numeric precision) and of the generator's ABI.
    uint64_t compiled_fingerprint(const Model& model);

} // namespace lignum
