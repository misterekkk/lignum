#include "compiled.hpp"
#include "binned.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
    // Prototype: POSIX only.
#else
    #include <dlfcn.h>
#endif

namespace lignum {

    namespace {

        using Clock = std::chrono::steady_clock;

        double ms_since(Clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        }

        // Exact C literal for a double (hex float).
        std::string literal(double v) {
            if (std::isinf(v)) return v > 0 ? "INFINITY" : "-INFINITY";
            char text[64];
            std::snprintf(text, sizeof(text), "%a", v);
            return text;
        }

        uint32_t n_cuts(const BinnedForest& f, unsigned feature) {
            return f.cut_offsets[feature + 1] - f.cut_offsets[feature];
        }

        // Numeric features whose codes all fit in int16 (bins below 0x7FFF) are read once per row
        // into a local v<feature> = (int32)(int16)code: NaN (0xFFFF) becomes -1, so
        //   code >= T             is  (uint32)v >= T      (NaN above every bin: right)
        //   uint16(code + 1) >= T  is  v >= T - 1          (NaN below every bin: left)
        // without the wrap. Rare wider features read c[feature] with the binned kernel's rule.
        bool local_feature(const BinnedForest& f, const BNode& b, int s) {
            return !((b.cat_slots >> s) & 1) && n_cuts(f, b.features[s]) < 0x7FFF;
        }

        // C expression that is 1 when slot s of b goes right: the binned kernel's decision.
        std::string condition(const BinnedForest& f, const BNode& b, int s) {
            const unsigned feature = b.features[s];
            const unsigned threshold = b.thresholds[s];
            const unsigned shift = (b.nan_left >> s) & 1;
            const std::string code = "c[" + std::to_string(feature) + "]";
            if ((b.cat_slots >> s) & 1) {
                const std::string bit = "((d[" + std::to_string(threshold >> 6) + "] >> " + std::to_string(threshold & 63) + ") & 1)";
                if ((b.xgb_slots >> s) & 1) return "(" + bit + " | ((" + code + " == 0xFFFF) & " + std::to_string(shift) + "))";
                return "!" + bit;
            }
            std::string compare;
            if (local_feature(f, b, s)) {
                const std::string v = "v" + std::to_string(feature);
                compare = shift ? v + " >= " + std::to_string(int(threshold) - 1) : "(uint32_t)" + v + " >= " + std::to_string(threshold);
                if ((b.zero_slots >> s) & 1) return "(" + v + " == " + std::to_string(f.zero_bin[feature]) + " ? " + std::to_string(shift ^ 1) + " : " + compare + ")";
                return compare;
            }
            compare = shift ? "(uint16_t)(" + code + " + 1) >= " + std::to_string(threshold) : code + " >= " + std::to_string(threshold);
            if ((b.zero_slots >> s) & 1) return "(" + code + " == " + std::to_string(f.zero_bin[feature]) + " ? " + std::to_string(shift ^ 1) + " : " + compare + ")";
            return compare;
        }

        // Binning of the numeric features, specialised per feature: cuts in constant tables, the
        // binary search unrolled to its known length (a branch-free count for short features).
        // Gives exactly the kernel's codes: #{cuts <= x}, kNanBin for NaN.
        void binning(std::ostream& out, const BinnedForest& f) {
            constexpr uint32_t kLinear = 8;
            for (size_t j = 0; j < f.n_features; j++) {
                const uint32_t len = n_cuts(f, j);
                if (f.feature_kind[j] != feature_kind::Numeric || len <= kLinear) continue;
                out << "static const double cuts_" << j << "[" << len << "] = {";
                for (uint32_t k = 0; k < len; k++) out << (k ? ", " : "") << literal(f.cuts[f.cut_offsets[j] + k]);
                out << "};\n";
            }
            out << "void lignum_bin(const double* restrict X, size_t stride, size_t n, uint16_t* restrict codes) {\n"
                << "  for (size_t i = 0; i < n; i++) {\n"
                << "    const double* restrict x = X + i * stride;\n"
                << "    uint16_t* restrict c = codes + i * " << f.n_features << ";\n";
            for (size_t j = 0; j < f.n_features; j++) {
                if (f.feature_kind[j] != feature_kind::Numeric) continue;
                const uint32_t len = n_cuts(f, j);
                const double* cuts = &f.cuts[f.cut_offsets[j]];
                out << "    { const double v = x[" << f.feature_of[j] << "]; unsigned b = 0";
                if (len <= kLinear) {
                    for (uint32_t k = 0; k < len; k++) out << " + (" << literal(cuts[k]) << " <= v)";
                    out << "; ";
                } else {
                    out << "; ";
                    for (uint32_t m = len; m > 1; m -= m / 2) out << "b = (cuts_" << j << "[b + " << m / 2 << "] <= v) ? b + " << m / 2 << " : b; ";
                    out << "b += (cuts_" << j << "[b] <= v); ";
                }
                out << "c[" << j << "] = (v != v) ? 0xFFFF : (uint16_t)b; }\n";
            }
            out << "  }\n}\n";
        }

        // Emits trees inline in a block's row loop: nested ifs on constant thresholds, leaves as
        // exact constants added to the running sum of the tree's output.
        class Generator {
            public:
                Generator(const BinnedForest& f, const double* leaf_values, bool float32)
                    : f_(f), leaf_values_(leaf_values), float32_(float32) {}

                const char* acc() const { return float32_ ? "float" : "double"; }

                void tree(std::ostream& out, size_t t, const std::string& sum) {
                    leaf_ = sum + " += ";
                    leaf_end_ = "; goto end_" + std::to_string(t) + ";";
                    ref(out, f_.roots[t], 2);
                    out << "    end_" << t << ":;\n";
                }

                // Marks the features tree t reads through locals.
                void locals(size_t t, std::vector<bool>& used) const { locals_of(f_.roots[t], used); }

            private:
                const BinnedForest& f_;
                const double* leaf_values_;
                bool float32_;
                std::string leaf_, leaf_end_;

                void locals_of(int32_t r, std::vector<bool>& used) const {
                    if (r < 0) return;
                    const BNode& b = f_.nodes[r];
                    for (int s = 0; s < 7; s++) {
                        if (local_feature(f_, b, s)) used[b.features[s]] = true;
                    }
                    for (int c = 0; c < 8; c++) locals_of(b.children[c], used);
                }

                static void indent(std::ostream& out, int depth) {
                    for (int i = 0; i < depth; i++) out << "  ";
                }

                // XGBoost rounds the exact double to float, like the kernel does.
                std::string value(int32_t r) const {
                    return (float32_ ? "(float)" : "") + literal(leaf_values_[(-r) - 1]);
                }

                void ref(std::ostream& out, int32_t r, int depth) {
                    if (r < 0) {
                        indent(out, depth);
                        out << leaf_ << value(r) << leaf_end_ << "\n";
                        return;
                    }
                    const BNode& b = f_.nodes[r];
                    const int32_t* c = b.children;
                    if (c[0] == c[1] && c[0] == c[2] && c[0] == c[3] && c[4] == c[5] && c[4] == c[6] && c[4] == c[7] &&
                        select(out, b, 0, c[4], c[0], depth)) return;
                    branch(out, b, 0, depth, [&](int d) { side(out, b, 2, 4, d); }, [&](int d) { side(out, b, 1, 0, d); });
                }

                // A split whose children are both leaves becomes a select, so a hard to predict last
                // level costs no misprediction.
                bool select(std::ostream& out, const BNode& b, int s, int32_t right, int32_t left, int depth) {
                    if (right >= 0 || left >= 0) return false;
                    indent(out, depth);
                    out << leaf_ << "((" << condition(f_, b, s) << ") ? " << value(right) << " : " << value(left) << ")" << leaf_end_ << "\n";
                    return true;
                }

                // Slot s (1 or 2) with subtrees children[first .. first + 3].
                void side(std::ostream& out, const BNode& b, int s, int first, int depth) {
                    const int32_t* c = &b.children[first];
                    if (c[0] == c[1] && c[0] == c[2] && c[0] == c[3]) {
                        ref(out, c[0], depth);  // the child is a leaf
                        return;
                    }
                    if (c[0] == c[1] && c[2] == c[3] && select(out, b, s, c[2], c[0], depth)) return;
                    branch(out, b, s, depth, [&](int d) { grandchild(out, b, 2 * s + 2, first + 2, d); },
                                             [&](int d) { grandchild(out, b, 2 * s + 1, first, d); });
                }

                void grandchild(std::ostream& out, const BNode& b, int s, int first, int depth) {
                    if (b.children[first] == b.children[first + 1]) {
                        ref(out, b.children[first], depth);
                        return;
                    }
                    if (select(out, b, s, b.children[first + 1], b.children[first], depth)) return;
                    branch(out, b, s, depth, [&](int d) { ref(out, b.children[first + 1], d); },
                                             [&](int d) { ref(out, b.children[first], d); });
                }

                template <typename Right, typename Left>
                void branch(std::ostream& out, const BNode& b, int s, int depth, Right right, Left left) {
                    indent(out, depth);
                    out << "if (" << condition(f_, b, s) << ") {\n";
                    right(depth + 1);
                    indent(out, depth);
                    out << "} else {\n";
                    left(depth + 1);
                    indent(out, depth);
                    out << "}\n";
                }
        };

        // LIGNUM_KEEP(v): an empty asm that makes v live in a register at this point, so the
        // compiler cannot sink the per-row loads back into the branches.
        const char* kPrelude =
            "#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n"
            "#if defined(__GNUC__) || defined(__clang__)\n"
            "#define restrict __restrict__\n#define LIGNUM_KEEP(v) __asm__(\"\" : \"+r\"(v))\n"
            "#else\n#define LIGNUM_KEEP(v)\n#endif\n\n";

        int run(const std::string& command) {
            return std::system(command.c_str());
        }

    } // namespace

    CompiledForest::~CompiledForest() {
    #if !defined(_WIN32)
        if (handle) dlclose(handle);
    #endif
    }

    std::shared_ptr<const CompiledForest> compile_forest(const Model& model, const CompileOptions& options, CompileStats* stats) {
    #if defined(_WIN32)
        (void)model; (void)options; (void)stats;
        throw std::runtime_error("Compiled mode is not supported on Windows yet.");
    #else
        if (!model.binned) throw std::runtime_error("Compiled mode needs the binned kernel (load with binned=True).");
        const BinnedForest& f = *model.binned;
        CompileStats local;
        CompileStats& st = stats ? *stats : local;

        // 1. Generate C. Trees are grouped into blocks; a block function walks its trees for every
        // row of a chunk, so the code of one block stays hot in the instruction cache and branch
        // predictor (one pass over all trees per row thrashes both). Each row's sum is carried
        // across blocks in `out`, in model order, so results stay bit-identical.
        auto t0 = Clock::now();
        const size_t block_trees = std::max<size_t>(1, options.block_trees);
        const size_t n_blocks = std::max<size_t>(1, (f.n_trees + block_trees - 1) / block_trees);
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const size_t parts = std::clamp<size_t>(options.jobs > 0 ? options.jobs : hw, 1, n_blocks);
        const size_t K = model.n_outputs;
        std::vector<std::string> sources(parts);
        Generator gen(f, model.leaf_values, model.float32);
        const char* acc = gen.acc();
        for (size_t p = 0; p < parts; p++) {
            std::ostringstream src;
            src << kPrelude;
            if (p == 0) binning(src, f);
            for (size_t b = n_blocks * p / parts; b < n_blocks * (p + 1) / parts; b++) {
                const size_t begin = b * block_trees, end = std::min(begin + block_trees, f.n_trees);
                std::vector<bool> outputs(K, false), locals(f.n_features, false);
                for (size_t t = begin; t < end; t++) {
                    gen.locals(t, locals);
                    outputs[K > 1 ? model.tree_outputs[t] : 0] = true;
                }
                src << "void lignum_block_" << b << "(const uint16_t* restrict codes, const uint64_t* restrict decisions, size_t n, double* restrict out) {\n"
                    << "  for (size_t i = 0; i < n; i++) {\n"
                    << "    const uint16_t* restrict c = codes + i * " << f.n_features << ";\n"
                    << "    const uint64_t* restrict d = decisions" << (f.decision_words ? " + i * " + std::to_string(f.decision_words) : std::string()) << ";\n"
                    << "    (void)c; (void)d;\n";
                for (size_t k = 0; k < K; k++) {
                    if (outputs[k]) src << "    " << acc << " s" << k << " = (" << acc << ")out[i * " << K << " + " << k << "];\n";
                }
                for (size_t j = 0; j < locals.size(); j++) {
                    if (locals[j]) src << "    int32_t v" << j << " = (int16_t)c[" << j << "]; LIGNUM_KEEP(v" << j << ");\n";
                }
                for (size_t t = begin; t < end; t++) gen.tree(src, t, "s" + std::to_string(K > 1 ? model.tree_outputs[t] : 0));
                for (size_t k = 0; k < K; k++) {
                    if (outputs[k]) src << "    out[i * " << K << " + " << k << "] = (double)s" << k << ";\n";
                }
                src << "  }\n}\n";
            }
            sources[p] = src.str();
        }
        st.codegen_ms = ms_since(t0);

        size_t hash = std::hash<std::string>{}(options.compiler + " " + options.flags);
        st.source_bytes = 0;
        for (const auto& s : sources) {
            hash = hash * 1000003u ^ std::hash<std::string>{}(s);
            st.source_bytes += s.size();
        }

        // 2. Compile (cached by source hash).
        namespace fs = std::filesystem;
        const fs::path dir = options.work_dir.empty() ? fs::temp_directory_path() / "lignum_compiled" : fs::path(options.work_dir);
        fs::create_directories(dir);
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%016zx", hash);
        const fs::path library = dir / (std::string("lignum_") + tag + ".so");
        st.library = library.string();

        t0 = Clock::now();
        st.cached = fs::exists(library);
        if (!st.cached) {
            std::vector<std::string> objects(sources.size());
            std::vector<int> status(sources.size(), 0);
            std::vector<std::thread> workers;
            for (size_t p = 0; p < sources.size(); p++) {
                const fs::path c_file = dir / (std::string("lignum_") + tag + "_" + std::to_string(p) + ".c");
                objects[p] = (dir / (std::string("lignum_") + tag + "_" + std::to_string(p) + ".o")).string();
                std::ofstream(c_file) << sources[p];
                const std::string command = options.compiler + " " + options.flags + " -fPIC -c \"" + c_file.string() + "\" -o \"" + objects[p] + "\"";
                workers.emplace_back([&status, p, command] { status[p] = run(command); });
            }
            for (auto& w : workers) w.join();
            for (int s : status) {
                if (s != 0) throw std::runtime_error("Compiling the generated code failed (see compiler output).");
            }
            std::string link = options.compiler + " -shared -o \"" + library.string() + ".tmp\"";
            for (const auto& o : objects) link += " \"" + o + "\"";
            if (run(link) != 0) throw std::runtime_error("Linking the generated code failed.");
            fs::rename(library.string() + ".tmp", library);
        }
        st.compile_ms = ms_since(t0);

        // 3. Load.
        t0 = Clock::now();
        auto compiled = std::make_shared<CompiledForest>();
        compiled->library = library.string();
        compiled->handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!compiled->handle) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        for (size_t b = 0; b < n_blocks; b++) {
            const std::string name = "lignum_block_" + std::to_string(b);
            auto fn = reinterpret_cast<CompiledForest::BlockFn>(dlsym(compiled->handle, name.c_str()));
            if (!fn) throw std::runtime_error(name + " not found in the compiled library.");
            compiled->blocks.push_back(fn);
        }
        compiled->bin = reinterpret_cast<CompiledForest::BinFn>(dlsym(compiled->handle, "lignum_bin"));
        if (!compiled->bin) throw std::runtime_error("lignum_bin not found in the compiled library.");
        st.load_ms = ms_since(t0);
        return compiled;
    #endif
    }

} // namespace lignum
