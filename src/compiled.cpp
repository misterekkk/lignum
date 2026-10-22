#include "compiled.hpp"
#include "binned.hpp"

#include <algorithm>
#include <chrono>
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

        // C expression that is 1 when slot s of b goes right (same rules as the binned kernel).
        std::string condition(const BinnedForest& f, const BNode& b, int s) {
            const unsigned feature = b.features[s];
            const unsigned threshold = b.thresholds[s];
            const unsigned shift = (b.nan_left >> s) & 1;
            std::ostringstream e;
            if ((b.cat_slots >> s) & 1) {
                const unsigned word = threshold >> 6, bit = threshold & 63;
                if ((b.xgb_slots >> s) & 1) {
                    e << "(((d[" << word << "] >> " << bit << ") & 1) | ((c[" << feature << "] == 0xFFFF) & " << shift << "))";
                } else {
                    e << "!((d[" << word << "] >> " << bit << ") & 1)";
                }
                return e.str();
            }
            const std::string compare = shift ? "(uint16_t)(c[" + std::to_string(feature) + "] + 1) >= " + std::to_string(threshold)
                                              : "c[" + std::to_string(feature) + "] >= " + std::to_string(threshold);
            if ((b.zero_slots >> s) & 1) {
                e << "(c[" << feature << "] == " << f.zero_bin[feature] << " ? " << (shift ^ 1) << " : (" << compare << "))";
            } else {
                e << compare;
            }
            return e.str();
        }

        class Generator {
            public:
                explicit Generator(const BinnedForest& f) : f_(f) {}

                void tree(std::ostream& out, size_t t) {
                    out << "static int32_t tree_" << t << "(const uint16_t* restrict c, const uint64_t* restrict d) {\n";
                    ref(out, f_.roots[t], 1);
                    out << "}\n";
                }

            private:
                const BinnedForest& f_;

                static void indent(std::ostream& out, int depth) {
                    for (int i = 0; i < depth; i++) out << "  ";
                }

                void ref(std::ostream& out, int32_t r, int depth) {
                    if (r < 0) {
                        indent(out, depth);
                        out << "return " << r << ";\n";
                        return;
                    }
                    const BNode& b = f_.nodes[r];
                    branch(out, b, 0, depth, [&](int d) { side(out, b, 2, 4, d); }, [&](int d) { side(out, b, 1, 0, d); });
                }

                // Slot s (1 or 2) with subtrees children[first .. first + 3].
                void side(std::ostream& out, const BNode& b, int s, int first, int depth) {
                    if (b.children[first] == b.children[first + 1] && b.children[first] == b.children[first + 2] &&
                        b.children[first] == b.children[first + 3]) {
                        ref(out, b.children[first], depth);  // the child is a leaf
                        return;
                    }
                    branch(out, b, s, depth, [&](int d) { grandchild(out, b, 2 * s + 2, first + 2, d); },
                                             [&](int d) { grandchild(out, b, 2 * s + 1, first, d); });
                }

                void grandchild(std::ostream& out, const BNode& b, int s, int first, int depth) {
                    if (b.children[first] == b.children[first + 1]) {
                        ref(out, b.children[first], depth);
                        return;
                    }
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

        const char* kPrelude = "#include <stdint.h>\n#if defined(__GNUC__) || defined(__clang__)\n#define restrict __restrict__\n#endif\n\n";

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

        // 1. Generate C: tree functions split into translation units for a parallel build.
        auto t0 = Clock::now();
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const size_t parts = std::clamp<size_t>(options.jobs > 0 ? options.jobs : hw, 1, std::max<size_t>(1, f.n_trees));
        std::vector<std::string> sources(parts + 1);
        Generator gen(f);
        for (size_t p = 0; p < parts; p++) {
            const size_t begin = f.n_trees * p / parts, end = f.n_trees * (p + 1) / parts;
            std::ostringstream src;
            src << kPrelude;
            for (size_t t = begin; t < end; t++) gen.tree(src, t);
            src << "void lignum_eval_" << p << "(const uint16_t* restrict c, const uint64_t* restrict d, int32_t* restrict leaves) {\n";
            for (size_t t = begin; t < end; t++) src << "  leaves[" << t << "] = tree_" << t << "(c, d);\n";
            src << "}\n";
            sources[p] = src.str();
        }
        std::ostringstream main_src;
        main_src << kPrelude;
        for (size_t p = 0; p < parts; p++) {
            main_src << "void lignum_eval_" << p << "(const uint16_t* restrict c, const uint64_t* restrict d, int32_t* restrict leaves);\n";
        }
        main_src << "void lignum_eval(const uint16_t* restrict c, const uint64_t* restrict d, int32_t* restrict leaves) {\n";
        for (size_t p = 0; p < parts; p++) main_src << "  lignum_eval_" << p << "(c, d, leaves);\n";
        main_src << "}\n";
        sources[parts] = main_src.str();
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
        compiled->eval = reinterpret_cast<CompiledForest::EvalFn>(dlsym(compiled->handle, "lignum_eval"));
        if (!compiled->eval) throw std::runtime_error("lignum_eval not found in the compiled library.");
        st.load_ms = ms_since(t0);
        return compiled;
    #endif
    }

} // namespace lignum
