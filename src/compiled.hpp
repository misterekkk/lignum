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
        std::string cache_dir;  // empty: default_cache_dir()
        // Write the library exactly here, always building it, instead of using the cache: a file,
        // or a folder (an existing one, or a path ending in a separator) for compiled_library_name().
        std::string library;
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
    // missing or fails. The code is built in a private temporary folder that is always removed
    // (a failed build keeps only its log, named in the error); the library is moved into place
    // with an atomic rename, so concurrent compiles of one model never see a half-written file.
    std::shared_ptr<const CompiledForest> compile_forest(const Model& model, const CompileOptions& options, CompileStats* stats);

    // Attaches a library built earlier by compile_forest (e.g. shipped next to the model file):
    // a file, or a folder holding compiled_library_name(model). The library records the
    // fingerprint of the model it was generated from and the ABI of the generated functions; a
    // library for another model or lignum version is rejected with std::runtime_error. Loading
    // runs the library's code: only load libraries you trust.
    std::shared_ptr<const CompiledForest> load_compiled(const Model& model, const std::string& library);

    // Per-user cache of compiled libraries: LIGNUM_CACHE_DIR if set, otherwise
    // %LOCALAPPDATA%\lignum\Cache (Windows), ~/Library/Caches/lignum (macOS) or
    // $XDG_CACHE_HOME/lignum, ~/.cache/lignum (Linux).
    std::string default_cache_dir();

    // Removes the compiled libraries in `cache_dir` (empty: default_cache_dir()) and returns how
    // many; libraries a process has loaded on Windows cannot be removed and stay.
    size_t clear_compiled_cache(const std::string& cache_dir);

    // File name of the model's library in a folder: lignum_<fingerprint>_<platform><extension>,
    // e.g. lignum_0123456789abcdef_linux-x86_64.so.
    std::string compiled_library_name(const Model& model);

    // Operating system and CPU architecture of this build, e.g. "linux-x86_64", "macos-arm64".
    const char* compiled_platform();

    // Stable 64-bit hash of everything the generated code depends on (trees, cuts, leaves,
    // outputs, numeric precision) and of the generator's ABI.
    uint64_t compiled_fingerprint(const Model& model);

} // namespace lignum
