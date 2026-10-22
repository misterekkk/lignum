#include "compiled.hpp"
#include "binned.hpp"

#include <algorithm>
#include <cctype>
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
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
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

        // Features of a block held in locals (the most used ones); the rest read c[feature].
        // M4, Covertype, 1 thread: all (51) 445 ms, 24: 412, 16: 409, 8: 415. x86-64 has half the
        // general registers, hence a smaller default there (CI, EPYC 7763: 16 instead of 10 gives
        // the same times, Covertype 0.795 -> 0.819 s on one thread).
        #ifndef LIGNUM_COMPILED_LOCALS
            #if defined(__x86_64__) || defined(_M_X64)
                #define LIGNUM_COMPILED_LOCALS 10
            #else
                #define LIGNUM_COMPILED_LOCALS 16
            #endif
        #endif
        constexpr size_t kMaxLocals = LIGNUM_COMPILED_LOCALS;

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
        // `locals`: dense features held in locals in this block (others read c[feature]).
        std::string condition(const BinnedForest& f, const BNode& b, int s, const std::vector<bool>& locals) {
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
            if (local_feature(f, b, s) && locals[feature]) {
                const std::string v = "v" + std::to_string(feature);
                compare = shift ? v + " >= " + std::to_string(int(threshold) - 1) : "(uint32_t)" + v + " >= " + std::to_string(threshold);
                if ((b.zero_slots >> s) & 1) return "(" + v + " == " + std::to_string(f.zero_bin[feature]) + " ? " + std::to_string(shift ^ 1) + " : " + compare + ")";
                return compare;
            }
            compare = !shift ? code + " >= " + std::to_string(threshold)
                    : local_feature(f, b, s) ? "(int16_t)" + code + " >= " + std::to_string(int(threshold) - 1)
                                             : "(uint16_t)(" + code + " + 1) >= " + std::to_string(threshold);
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
            out << "LIGNUM_EXPORT void lignum_bin(const double* restrict X, size_t stride, size_t n, uint16_t* restrict codes) {\n"
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

        // Emits trees inline in a block's row loop: nested ifs on constant thresholds, leaves read
        // from the block's table (exact values; XGBoost's rounded to float like the kernel) and
        // added to the running sum of the tree's output.
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

                // Leaves of the current block, read as L[k] (index in the block's table).
                void begin_block() { table_values_.clear(); table_index_.clear(); }
                const std::vector<double>& table() const { return table_values_; }

                // Counts the slots of tree t that can read a local, per dense feature.
                void count_locals(size_t t, std::vector<size_t>& uses) const { locals_of(f_.roots[t], uses); }
                void set_locals(const std::vector<bool>* locals) { locals_ = locals; }

            private:
                const BinnedForest& f_;
                const double* leaf_values_;
                bool float32_;
                std::string leaf_, leaf_end_;
                std::vector<double> table_values_;
                std::unordered_map<int32_t, size_t> table_index_;

                size_t slot(int32_t r) {
                    auto [it, inserted] = table_index_.try_emplace(r, table_values_.size());
                    if (inserted) table_values_.push_back(leaf_values_[(-r) - 1]);
                    return it->second;
                }

                const std::vector<bool>* locals_ = nullptr;

                void locals_of(int32_t r, std::vector<size_t>& uses) const {
                    if (r < 0) return;
                    const BNode& b = f_.nodes[r];
                    for (int s = 0; s < 7; s++) {
                        // Unused slots compare against threshold 0 and point both ways alike.
                        if (local_feature(f_, b, s) && (s == 0 || b.thresholds[s] != 0)) uses[b.features[s]]++;
                    }
                    for (int c = 0; c < 8; c++) locals_of(b.children[c], uses);
                }

                static void indent(std::ostream& out, int depth) {
                    for (int i = 0; i < depth; i++) out << "  ";
                }

                std::string value(int32_t r) { return "L[" + std::to_string(slot(r)) + "]"; }

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
                    // Select the index, then one load.
                    out << leaf_ << "L[(" << condition(f_, b, s, *locals_) << ") ? " << slot(right) << " : " << slot(left) << "]" << leaf_end_ << "\n";
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
                    out << "if (" << condition(f_, b, s, *locals_) << ") {\n";
                    right(depth + 1);
                    indent(out, depth);
                    out << "} else {\n";
                    left(depth + 1);
                    indent(out, depth);
                    out << "}\n";
                }
        };

        // LIGNUM_KEEP(v): an empty asm that makes v live in a register at this point, so the
        // compiler cannot sink the per-row loads back into the branches (MSVC has no inline asm
        // on x64: there it is a no-op, which costs speed, not correctness). LIGNUM_EXPORT makes
        // the entry points visible in a Windows DLL.
        const char* kPrelude =
            "#include <math.h>\n#include <stddef.h>\n#include <stdint.h>\n"
            "#if defined(__GNUC__) || defined(__clang__)\n"
            "#define restrict __restrict__\n#define LIGNUM_KEEP(v) __asm__(\"\" : \"+r\"(v))\n"
            "#else\n#define restrict __restrict\n#define LIGNUM_KEEP(v)\n#endif\n"
            "#if defined(_WIN32)\n#define LIGNUM_EXPORT __declspec(dllexport)\n"
            "#else\n#define LIGNUM_EXPORT __attribute__((visibility(\"default\")))\n#endif\n\n";

        int run(const std::string& command) {
        #if defined(_WIN32)
            // cmd /c drops the first and last quote of the line; wrap it so quoted paths survive.
            return std::system(("\"" + command + "\"").c_str());
        #else
            return std::system(command.c_str());
        #endif
        }

        std::string quote(const std::string& s) { return "\"" + s + "\""; }

    #if defined(_WIN32)
        constexpr const char* kNull = "NUL";
        constexpr const char* kLibraryExtension = ".dll";
    #else
        constexpr const char* kNull = "/dev/null";
        constexpr const char* kLibraryExtension = ".so";
    #endif

        // The C compiler and how to drive it.
        struct Toolchain {
            std::string compiler;
            bool msvc = false;     // cl / clang-cl command line; otherwise gcc / clang
            std::string prefix;    // Windows: sets up the Visual Studio environment first
            std::string name() const { return prefix.empty() ? compiler : compiler + " (Visual Studio)"; }
        };

        bool msvc_style(const std::string& compiler) {
            std::string base = std::filesystem::path(compiler).filename().string();
            std::transform(base.begin(), base.end(), base.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            return base == "cl" || base == "cl.exe" || base == "clang-cl" || base == "clang-cl.exe";
        }

        bool responds(const Toolchain& t) {
            const std::string probe = t.msvc ? t.compiler + " /?" : t.compiler + " --version";
            return run(t.prefix + probe + " > " + kNull + " 2>&1") == 0;
        }

    #if defined(_WIN32)
        // vcvarsall.bat of the newest Visual Studio with the C++ tools, found with vswhere.
        std::string visual_studio_environment() {
            const char* root = std::getenv("ProgramFiles(x86)");
            if (!root) return "";
            const std::filesystem::path vswhere = std::filesystem::path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe";
            if (!std::filesystem::exists(vswhere)) return "";
            const auto out = std::filesystem::temp_directory_path() / "lignum_vswhere.txt";
            const std::string command = quote(vswhere.string()) + " -latest -products * -property installationPath > " + quote(out.string());
            if (run(command) != 0) return "";
            std::string path;
            std::getline(std::ifstream(out), path);
            std::filesystem::remove(out);
            while (!path.empty() && (path.back() == '\r' || path.back() == ' ')) path.pop_back();
            const auto vcvars = std::filesystem::path(path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat";
            if (path.empty() || !std::filesystem::exists(vcvars)) return "";
        #if defined(_M_ARM64)
            const char* arch = "arm64";
        #else
            const char* arch = "x64";
        #endif
            return "call " + quote(vcvars.string()) + " " + arch + " > NUL && ";
        }
    #endif

        // "auto": cc on POSIX; on Windows cl on PATH, Visual Studio (via vswhere), gcc, clang.
        Toolchain find_toolchain(const std::string& requested) {
            std::vector<Toolchain> candidates;
            if (!requested.empty() && requested != "auto") {
                candidates.push_back({requested, msvc_style(requested), ""});
            } else {
            #if defined(_WIN32)
                candidates.push_back({"cl", true, ""});
                const std::string vs = visual_studio_environment();
                if (!vs.empty()) candidates.push_back({"cl", true, vs});
                candidates.push_back({"gcc", false, ""});
                candidates.push_back({"clang", false, ""});
            #else
                candidates.push_back({"cc", false, ""});
            #endif
            }
            for (const auto& t : candidates) {
                if (responds(t)) return t;
            }
            throw std::runtime_error("C compiler '" + (requested.empty() ? std::string("auto") : requested) +
                                     "' not found; install one (gcc, clang, or Visual Studio with C++ tools on Windows) "
                                     "or load a library compiled elsewhere.");
        }

        // Compiles the sources (in parallel) and links them into `library`.
        void build(const Toolchain& t, const std::string& flags, const std::vector<std::filesystem::path>& c_files,
                   const std::filesystem::path& dir, const std::string& tag, const std::filesystem::path& library) {
            if (t.msvc) {
                // One command: /MP compiles the files in parallel, /LD links the DLL.
                std::string command = t.prefix + t.compiler + " /nologo " + (flags.empty() ? "/O2" : flags) + " /LD /MP /Fo" +
                                      quote(dir.string() + "\\\\") + " /Fe" + quote(library.string());  // "dir\\": a lone \ would escape the quote
                for (const auto& c : c_files) command += " " + quote(c.string());
                command += " > " + quote((dir / ("lignum_" + tag + "_build.log")).string()) + " 2>&1";
                if (run(command) != 0) {
                    throw std::runtime_error("Compiling the generated code failed; see " + (dir / ("lignum_" + tag + "_build.log")).string());
                }
                return;
            }
        #if defined(_WIN32)
            const std::string pic;
        #else
            const std::string pic = " -fPIC";
        #endif
            std::vector<std::string> objects(c_files.size());
            std::vector<int> status(c_files.size(), 0);
            std::vector<std::thread> workers;
            for (size_t p = 0; p < c_files.size(); p++) {
                objects[p] = (dir / (c_files[p].stem().string() + ".o")).string();
                const std::string command = t.prefix + t.compiler + " " + (flags.empty() ? "-O2" : flags) + pic + " -c " +
                                            quote(c_files[p].string()) + " -o " + quote(objects[p]);
                workers.emplace_back([&status, p, command] { status[p] = run(command); });
            }
            for (auto& w : workers) w.join();
            for (int s : status) {
                if (s != 0) throw std::runtime_error("Compiling the generated code failed (see compiler output).");
            }
            std::string link = t.prefix + t.compiler + " -shared -o " + quote(library.string());
            for (const auto& o : objects) link += " " + quote(o);
            if (run(link) != 0) throw std::runtime_error("Linking the generated code failed.");
        }

        // Version of the generated functions' interface and semantics; bump on any change to the
        // generator that old libraries would not match.
        constexpr uint32_t kAbi = 1;

        // FNV-1a, 64 bit: stable across runs and platforms of the same byte order.
        class Fnv {
            public:
                void bytes(const void* data, size_t n) {
                    const auto* p = static_cast<const unsigned char*>(data);
                    for (size_t i = 0; i < n; i++) h_ = (h_ ^ p[i]) * 0x100000001b3ULL;
                }
                template <typename T>
                void array(const T* data, size_t n) {
                    value(uint64_t{n});
                    if (n) bytes(data, n * sizeof(T));
                }
                template <typename T>
                void value(T v) { bytes(&v, sizeof(v)); }
                uint64_t get() const { return h_; }

            private:
                uint64_t h_ = 0xcbf29ce484222325ULL;
        };

        std::string hex(uint64_t v) {
            char text[32];
            std::snprintf(text, sizeof(text), "0x%016llxULL", static_cast<unsigned long long>(v));
            return text;
        }

        void* open_handle(const std::string& library) {
        #if defined(_WIN32)
            // LoadLibrary appends ".dll" to a name without an extension; a trailing dot prevents it.
            std::wstring name = std::filesystem::path(library).wstring();
            if (!std::filesystem::path(library).has_extension()) name += L".";
            void* handle = static_cast<void*>(LoadLibraryW(name.c_str()));
            if (!handle) throw std::runtime_error("LoadLibrary failed for " + library + " (error " + std::to_string(GetLastError()) + ")");
        #else
            void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
        #endif
            return handle;
        }

        void* raw_symbol(void* handle, const std::string& name) {
        #if defined(_WIN32)
            return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name.c_str()));
        #else
            return dlsym(handle, name.c_str());
        #endif
        }

        template <typename T>
        const T* symbol(void* handle, const std::string& name, const std::string& library) {
            const auto* p = reinterpret_cast<const T*>(raw_symbol(handle, name));
            if (!p) throw std::runtime_error(name + " not found in " + library + " (not a lignum compiled library?)");
            return p;
        }

        std::shared_ptr<CompiledForest> open_library(const Model& model, const std::string& library) {
            auto compiled = std::make_shared<CompiledForest>();
            compiled->library = library;
            compiled->handle = open_handle(library);
            const uint32_t abi = *symbol<uint32_t>(compiled->handle, "lignum_abi", library);
            if (abi != kAbi) {
                throw std::runtime_error(library + " was generated by another lignum version (ABI " + std::to_string(abi) +
                                         ", expected " + std::to_string(kAbi) + "); compile the model again.");
            }
            if (*symbol<uint64_t>(compiled->handle, "lignum_fingerprint", library) != compiled_fingerprint(model)) {
                throw std::runtime_error(library + " was compiled from a different model; compile this model again.");
            }
            const uint32_t n_blocks = *symbol<uint32_t>(compiled->handle, "lignum_n_blocks", library);
            for (uint32_t b = 0; b < n_blocks; b++) {
                compiled->blocks.push_back(reinterpret_cast<CompiledForest::BlockFn>(
                    const_cast<void*>(static_cast<const void*>(symbol<char>(compiled->handle, "lignum_block_" + std::to_string(b), library)))));
            }
            compiled->bin = reinterpret_cast<CompiledForest::BinFn>(
                const_cast<void*>(static_cast<const void*>(symbol<char>(compiled->handle, "lignum_bin", library))));
            return compiled;
        }

    } // namespace

    CompiledForest::~CompiledForest() {
        if (!handle) return;
    #if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle));
    #else
        dlclose(handle);
    #endif
    }

    uint64_t compiled_fingerprint(const Model& model) {
        if (!model.binned) throw std::runtime_error("Compiled mode needs the binned kernel (load with binned=True).");
        const BinnedForest& f = *model.binned;
        Fnv h;
        h.value(kAbi);
        h.value(model.n_outputs);
        h.value(uint8_t{model.float32});
        h.array(f.nodes, f.n_nodes);
        h.array(f.roots, f.n_trees);
        h.array(f.feature_of, f.n_features);
        h.array(f.feature_kind, f.n_features);
        h.array(f.zero_bin, f.n_features);
        h.array(f.cut_offsets, f.n_features + 1);
        h.array(f.cuts, f.n_cuts);
        h.value(f.decision_words);
        h.array(model.leaf_values, model.n_leaves);
        if (model.n_outputs > 1) h.array(model.tree_outputs, model.n_offsets);
        return h.get();
    }

    std::shared_ptr<const CompiledForest> load_compiled(const Model& model, const std::string& library) {
        if (!model.binned) throw std::runtime_error("Compiled mode needs the binned kernel (load with binned=True).");
        if (!std::filesystem::exists(library)) throw std::runtime_error("Compiled library not found: " + library);
        return open_library(model, library);
    }

    std::shared_ptr<const CompiledForest> compile_forest(const Model& model, const CompileOptions& options, CompileStats* stats) {
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
            if (p == 0) {
                src << "LIGNUM_EXPORT const uint32_t lignum_abi = " << kAbi << ";\n"
                    << "LIGNUM_EXPORT const uint64_t lignum_fingerprint = " << hex(compiled_fingerprint(model)) << ";\n"
                    << "LIGNUM_EXPORT const uint32_t lignum_n_blocks = " << n_blocks << ";\n";
                binning(src, f);
            }
            for (size_t b = n_blocks * p / parts; b < n_blocks * (p + 1) / parts; b++) {
                const size_t begin = b * block_trees, end = std::min(begin + block_trees, f.n_trees);
                std::vector<bool> outputs(K, false), locals(f.n_features, false);
                std::vector<size_t> uses(f.n_features, 0);
                for (size_t t = begin; t < end; t++) {
                    gen.count_locals(t, uses);
                    outputs[K > 1 ? model.tree_outputs[t] : 0] = true;
                }
                // Locals for the most used features only: more than the registers would be
                // spilled to the stack and reloaded anyway.
                std::vector<size_t> order(f.n_features);
                for (size_t j = 0; j < order.size(); j++) order[j] = j;
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b2) { return uses[a] > uses[b2]; });
                for (size_t k = 0; k < std::min<size_t>(kMaxLocals, order.size()); k++) {
                    if (uses[order[k]]) locals[order[k]] = true;
                }
                gen.set_locals(&locals);
                gen.begin_block();
                std::ostringstream body;
                for (size_t t = begin; t < end; t++) gen.tree(body, t, "s" + std::to_string(K > 1 ? model.tree_outputs[t] : 0));
                // Leaves of the block in a table: one load per leaf instead of a 64-bit constant
                // built in the code (mov + 3 movk + fmov). M4, 1 thread: Covertype 488 -> 435 ms,
                // Bank 51.8 -> 46.8 ms. The empty asm hides the contents, so the compiler loads
                // the leaves instead of folding them back into constants.
                src << "static const " << acc << " leaves_" << b << "[" << std::max<size_t>(1, gen.table().size()) << "] = {";
                for (size_t k = 0; k < gen.table().size(); k++) src << (k ? ", " : "") << (model.float32 ? "(float)" : "") << literal(gen.table()[k]);
                src << "};\n"
                    << "LIGNUM_EXPORT void lignum_block_" << b << "(const uint16_t* restrict codes, const uint64_t* restrict decisions, size_t n, double* restrict out) {\n"
                    << "  const " << acc << "* restrict L = leaves_" << b << "; LIGNUM_KEEP(L);\n";
                src << "  for (size_t i = 0; i < n; i++) {\n"
                    << "    const uint16_t* restrict c = codes + i * " << f.n_features << ";\n"
                    << "    const uint64_t* restrict d = decisions" << (f.decision_words ? " + i * " + std::to_string(f.decision_words) : std::string()) << ";\n"
                    << "    (void)c; (void)d;\n";
                for (size_t k = 0; k < K; k++) {
                    if (outputs[k]) src << "    " << acc << " s" << k << " = (" << acc << ")out[i * " << K << " + " << k << "];\n";
                }
                for (size_t j = 0; j < locals.size(); j++) {
                    if (locals[j]) src << "    int32_t v" << j << " = (int16_t)c[" << j << "]; LIGNUM_KEEP(v" << j << ");\n";
                }
                src << body.str();
                for (size_t k = 0; k < K; k++) {
                    if (outputs[k]) src << "    out[i * " << K << " + " << k << "] = (double)s" << k << ";\n";
                }
                src << "  }\n}\n";
            }
            sources[p] = src.str();
        }
        st.codegen_ms = ms_since(t0);

        size_t hash = std::hash<std::string>{}(options.compiler + " " + options.flags + " " + kLibraryExtension);
        st.source_bytes = 0;
        for (const auto& s : sources) {
            hash = hash * 1000003u ^ std::hash<std::string>{}(s);
            st.source_bytes += s.size();
        }

        // 2. Compile (cached by source hash; the toolchain is only looked up on a miss).
        namespace fs = std::filesystem;
        const fs::path dir = options.work_dir.empty() ? fs::temp_directory_path() / "lignum_compiled" : fs::path(options.work_dir);
        fs::create_directories(dir);
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%016zx", hash);
        const fs::path library = dir / (std::string("lignum_") + tag + kLibraryExtension);
        st.library = library.string();

        t0 = Clock::now();
        st.cached = fs::exists(library);
        if (!st.cached) {
            const Toolchain toolchain = find_toolchain(options.compiler);
            std::vector<fs::path> c_files(sources.size());
            for (size_t p = 0; p < sources.size(); p++) {
                c_files[p] = dir / (std::string("lignum_") + tag + "_" + std::to_string(p) + ".c");
                std::ofstream(c_files[p], std::ios::binary) << sources[p];
            }
            // Build under a temporary name and rename, so a half-written library is never cached.
            const fs::path tmp = dir / (std::string("lignum_") + tag + "_tmp" + kLibraryExtension);
            build(toolchain, options.flags, c_files, dir, tag, tmp);
            fs::rename(tmp, library);
            st.compiler = toolchain.name();
        }
        st.compile_ms = ms_since(t0);

        // 3. Copy next to the model if asked, then load.
        if (!options.output.empty()) {
            const fs::path output(options.output);
            if (output.has_parent_path()) fs::create_directories(output.parent_path());
            const fs::path tmp = output.string() + ".tmp";
            fs::copy_file(library, tmp, fs::copy_options::overwrite_existing);
            fs::rename(tmp, output);
            st.library = output.string();
        }
        t0 = Clock::now();
        auto compiled = open_library(model, st.library);
        st.load_ms = ms_since(t0);
        return compiled;
    }

} // namespace lignum
