#pragma once

#include <lignum/model.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace lignum {

    // Experimental compiled mode: every tree of the binned forest becomes C code (nested ifs with
    // constant thresholds), built with the system compiler and loaded with dlopen. Binning, threads
    // and the summation of leaves stay in lignum, so results stay bit-identical; the generated
    // function only answers "which leaf, for every tree" for one row.
    struct CompiledForest {
        using EvalFn = void (*)(const uint16_t* codes, const uint64_t* decisions, int32_t* leaves);
        EvalFn eval = nullptr;
        void* handle = nullptr;
        std::string library;
        ~CompiledForest();
    };

    struct CompileOptions {
        std::string compiler = "cc";
        std::string flags = "-O2";
        std::string work_dir;  // empty: system temp directory
        int jobs = 0;          // translation units compiled in parallel; 0 = hardware threads
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
