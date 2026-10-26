#include "catboost.hpp"
#include "parsers/parsers.hpp"
#include "simdjson.h"

#include <lignum/lignum.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#ifdef _OPENMP
    #include <omp.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <intrin.h>
    #include <immintrin.h>
#endif

namespace lignum {

    // ---------------------------------------------------------------------------------------------
    // CityHash64 as in CatBoost's util/digest/city.cpp (Google's first release, MIT licence):
    // categories are hashed with CityHash64(utf8) & 0xffffffff. Later CityHash versions give
    // different values (for over 64 bytes even 1.0.3 does).

    namespace {

        constexpr uint64_t k0 = 0xc3a5c85c97cb3127ULL;
        constexpr uint64_t k1 = 0xb492b66fbe98f273ULL;
        constexpr uint64_t k2 = 0x9ae16a3b2f90404fULL;
        constexpr uint64_t k3 = 0xc949d7c7509e6557ULL;

        inline uint64_t fetch64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
        inline uint32_t fetch32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
        inline uint64_t rotate(uint64_t v, int s) { return s == 0 ? v : ((v >> s) | (v << (64 - s))); }
        inline uint64_t rotate_by_at_least1(uint64_t v, int s) { return (v >> s) | (v << (64 - s)); }
        inline uint64_t shift_mix(uint64_t v) { return v ^ (v >> 47); }

        inline uint64_t hash_len16(uint64_t u, uint64_t v) {
            constexpr uint64_t kMul = 0x9ddfea08eb382d69ULL;
            uint64_t a = (u ^ v) * kMul;
            a ^= (a >> 47);
            uint64_t b = (v ^ a) * kMul;
            b ^= (b >> 47);
            return b * kMul;
        }

        uint64_t hash_len0to16(const char* s, size_t len) {
            if (len > 8) {
                const uint64_t a = fetch64(s);
                const uint64_t b = fetch64(s + len - 8);
                return hash_len16(a, rotate_by_at_least1(b + len, static_cast<int>(len))) ^ b;
            }
            if (len >= 4) {
                const uint64_t a = fetch32(s);
                return hash_len16(len + (a << 3), fetch32(s + len - 4));
            }
            if (len > 0) {
                const uint8_t a = static_cast<uint8_t>(s[0]);
                const uint8_t b = static_cast<uint8_t>(s[len >> 1]);
                const uint8_t c = static_cast<uint8_t>(s[len - 1]);
                const uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
                const uint32_t z = static_cast<uint32_t>(len) + (static_cast<uint32_t>(c) << 2);
                return shift_mix(y * k2 ^ z * k3) * k2;
            }
            return k2;
        }

        uint64_t hash_len17to32(const char* s, size_t len) {
            const uint64_t a = fetch64(s) * k1;
            const uint64_t b = fetch64(s + 8);
            const uint64_t c = fetch64(s + len - 8) * k2;
            const uint64_t d = fetch64(s + len - 16) * k0;
            return hash_len16(rotate(a - b, 43) + rotate(c, 30) + d, a + rotate(b ^ k3, 20) - c + len);
        }

        std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
            a += w;
            b = rotate(b + a + z, 21);
            const uint64_t c = a;
            a += x;
            a += y;
            b += rotate(a, 44);
            return {a + z, b + c};
        }

        std::pair<uint64_t, uint64_t> weak_hash_len32_with_seeds(const char* s, uint64_t a, uint64_t b) {
            return weak_hash_len32_with_seeds(fetch64(s), fetch64(s + 8), fetch64(s + 16), fetch64(s + 24), a, b);
        }

        uint64_t hash_len33to64(const char* s, size_t len) {
            uint64_t z = fetch64(s + 24);
            uint64_t a = fetch64(s) + (len + fetch64(s + len - 16)) * k0;
            uint64_t b = rotate(a + z, 52);
            uint64_t c = rotate(a, 37);
            a += fetch64(s + 8);
            c += rotate(a, 7);
            a += fetch64(s + 16);
            const uint64_t vf = a + z;
            const uint64_t vs = b + rotate(a, 31) + c;
            a = fetch64(s + 16) + fetch64(s + len - 32);
            z = fetch64(s + len - 8);
            b = rotate(a + z, 52);
            c = rotate(a, 37);
            a += fetch64(s + len - 24);
            c += rotate(a, 7);
            a += fetch64(s + len - 16);
            const uint64_t wf = a + z;
            const uint64_t ws = b + rotate(a, 31) + c;
            const uint64_t r = shift_mix((vf + ws) * k2 + (wf + vs) * k0);
            return shift_mix(r * k0 + vs) * k2;
        }

    } // namespace

    uint64_t city_hash64(const char* s, size_t len) {
        if (len <= 32) return len <= 16 ? hash_len0to16(s, len) : hash_len17to32(s, len);
        if (len <= 64) return hash_len33to64(s, len);

        // Over 64 bytes: the end first, then 64-byte chunks with 56 bytes of state.
        uint64_t x = fetch64(s);
        uint64_t y = fetch64(s + len - 16) ^ k1;
        uint64_t z = fetch64(s + len - 56) ^ k0;
        auto v = weak_hash_len32_with_seeds(s + len - 64, len, y);
        auto w = weak_hash_len32_with_seeds(s + len - 32, len * k1, k0);
        z += shift_mix(v.second) * k1;
        x = rotate(z + x, 39) * k1;
        y = rotate(y, 33) * k1;
        len = (len - 1) & ~static_cast<size_t>(63);
        do {
            x = rotate(x + y + v.first + fetch64(s + 16), 37) * k1;
            y = rotate(y + v.second + fetch64(s + 48), 42) * k1;
            x ^= w.second;
            y ^= v.first;
            z = rotate(z ^ w.first, 33);
            v = weak_hash_len32_with_seeds(s, v.second * k1, x + w.first);
            w = weak_hash_len32_with_seeds(s + 32, z + w.second, y);
            std::swap(z, x);
            s += 64;
            len -= 64;
        } while (len != 0);
        return hash_len16(hash_len16(v.first, w.first) + shift_mix(y) * k1 + z, hash_len16(v.second, w.second) + x);
    }

    // ---------------------------------------------------------------------------------------------
    // CatBoost's exp (NCB::FastExpWithInfInplace). On x86 it is herumi's fmath::expd_v, whose SSE2
    // and AVX2 loops do the same double operations as this scalar copy (no FMA): a 2^11-entry table
    // of 2^(i/2048), the index and exponent taken from x * 2048/ln2 + 3 * 2^51, and a cubic
    // correction. The vector loops clamp x to [-708.396..., 709.7827...]; only arguments beyond
    // that (results near 0 or overflow) could differ from CatBoost's scalar tail.

    namespace {

        struct FmathExpTable {
            static constexpr int kBits = 11;
            static constexpr uint64_t kSize = uint64_t{1} << kBits;
            static constexpr uint64_t kAdj = (uint64_t{1} << (kBits + 10)) - (uint64_t{1} << kBits);
            // fmath fills the table with ::pow(2, i/2048) at run time, so it is the platform's pow:
            // CatBoost and lignum share the system libm on Linux and macOS (whose x86_64 pow is
            // not correctly rounded at two entries), while CatBoost's Windows build matches the
            // correctly rounded table and MSVC's pow does not.
            uint64_t table[kSize];
            // ln 2 rounded to double (what ::log(2.0) gives on every platform we test).
            static constexpr double kLn2 = 0.69314718055994530942;
            double a = static_cast<double>(kSize) / kLn2;
            double ra = 1 / a;

            FmathExpTable() {
            #ifdef _WIN32
                static constexpr uint64_t rounded[kSize] = {
                    #include "catboost_exp_table.inc"
                };
                std::memcpy(table, rounded, sizeof(table));
            #else
                for (uint64_t i = 0; i < kSize; i++) {
                    const double v = std::pow(2.0, static_cast<double>(i) * (1.0 / static_cast<double>(kSize)));
                    uint64_t bits;
                    std::memcpy(&bits, &v, 8);
                    table[i] = bits & ((uint64_t{1} << 52) - 1);
                }
            #endif
            }
        };

        // `vector`: the SIMD loop, which clamps x; otherwise fmath::expd, used for the elements
        // after the last full vector, which returns 0 and infinity outside its range.
        [[maybe_unused]] double fmath_expd(double x, const FmathExpTable& c, bool vector) {
            constexpr double kC1 = 1.0, kC2 = 0.16666666685227835064, kC3 = 3.0000000027955394;
            if (vector) {
                x = std::min(x, 709.78272569338397);
                x = std::max(x, -708.39641853226408);
            } else {
                if (x <= -708.39641853226408) return 0;
                if (x >= 709.78271289338397) return std::numeric_limits<double>::infinity();
            }
            const double b = static_cast<double>(uint64_t{3} << 51);
            const double d = x * c.a + b;
            uint64_t bits;
            std::memcpy(&bits, &d, 8);
            const uint64_t iax = c.table[bits & (FmathExpTable::kSize - 1)];
            const double t = (d - b) * c.ra - x;
            const uint64_t u = (((bits + FmathExpTable::kAdj) >> FmathExpTable::kBits) << 52) | iax;
            double scale;
            std::memcpy(&scale, &u, 8);
            const double y = (((kC3 - t) * (t * t)) * kC2 - t) + kC1;
            return y * scale;
        }

    } // namespace

    namespace {

    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        // fmath's loop is 4 wide with AVX2 (as CatBoost dispatches: AVX and AVX2), otherwise 2 (SSE2).
        size_t fmath_width() {
            static const size_t width = [] {
            #if defined(_MSC_VER)
                int info[4];
                __cpuid(info, 1);
                const bool avx = (info[2] & (1 << 28)) && (info[2] & (1 << 27)) && ((_xgetbv(0) & 6) == 6);
                __cpuidex(info, 7, 0);
                return avx && (info[1] & (1 << 5)) ? size_t{4} : size_t{2};
            #else
                __builtin_cpu_init();
                return __builtin_cpu_supports("avx") && __builtin_cpu_supports("avx2") ? size_t{4} : size_t{2};
            #endif
            }();
            return width;
        }
    #endif

    } // namespace

    void catboost_exp(double* x, size_t n) {
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        static const FmathExpTable table;
        const size_t vector_end = n & ~(fmath_width() - 1);
        for (size_t i = 0; i < n; i++) x[i] = fmath_expd(x[i], table, i < vector_end);
    #else
        for (size_t i = 0; i < n; i++) x[i] = std::exp(x[i]);
    #endif
    }

    void catboost_softmax(double* out, size_t n_rows, size_t dim) {
        for (size_t r = 0; r < n_rows; r++) {
            double* v = out + r * dim;
            const double max = *std::max_element(v, v + dim);
            for (size_t d = 0; d < dim; d++) v[d] -= max;
            catboost_exp(v, dim);
            double sum = 0;
            for (size_t d = 0; d < dim; d++) sum += v[d];
            for (size_t d = 0; d < dim; d++) v[d] /= sum;
        }
    }

    uint32_t catboost_hash(std::string_view category) {
        return static_cast<uint32_t>(city_hash64(category.data(), category.size()));
    }

    namespace {

        // CatBoost's CalcHash: combines a projection's hash with one more value.
        inline uint64_t combine_hash(uint64_t a, uint64_t b) {
            constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
            return kMagic * (a + kMagic * b);
        }

        // Slot of a combination hash in a table; the hashes are already well mixed.
        inline uint64_t slot_of(uint64_t key, uint64_t mask) { return (key ^ (key >> 29)) & mask; }

        // Float borders padded with +inf to a power of two, for bucket_search.
        void pad_borders(ObliviousForest& f) {
            for (ObliviousForest::Float& x : f.floats) {
                size_t n = 1;
                while (n < x.borders.size()) n <<= 1;
                x.search = x.borders;
                x.search.resize(n, std::numeric_limits<float>::infinity());
            }
        }

        // Fills the forest's prefix tree from its projections (see ObliviousForest::prefix_cat):
        // projections sorted by category list, one node per category not shared with the previous
        // list. In this order every node's parent is the latest node one level up.
        void index_prefixes(ObliviousForest& f) {
            std::vector<uint32_t> order(f.projections.size());
            for (uint32_t p = 0; p < order.size(); p++) order[p] = p;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return f.projections[a].cats < f.projections[b].cats; });
            f.prefix_cat.clear();
            f.prefix_level.clear();
            f.prefix_projection_begin.assign(1, 0);
            f.prefix_projections.clear();
            f.prefix_levels = 0;
            const std::vector<uint32_t>* previous = nullptr;
            for (uint32_t p : order) {
                const std::vector<uint32_t>& cats = f.projections[p].cats;
                if (cats.empty() || cats.size() > 255) throw std::runtime_error("CatBoost projection with " + std::to_string(cats.size()) + " categories.");
                if (!previous || cats != *previous) {  // new nodes from the first category not shared with the previous list
                    size_t common = 0;
                    while (previous && common + 1 < cats.size() && common < previous->size() && cats[common] == (*previous)[common]) common++;
                    for (size_t level = common; level < cats.size(); level++) {
                        f.prefix_cat.push_back(cats[level]);
                        f.prefix_level.push_back(static_cast<uint8_t>(level));
                        f.prefix_projection_begin.push_back(static_cast<uint32_t>(f.prefix_projections.size()));
                    }
                }
                f.prefix_projections.push_back(p);  // to the latest node: its category list is this one
                f.prefix_projection_begin.back() = static_cast<uint32_t>(f.prefix_projections.size());
                f.prefix_levels = std::max(f.prefix_levels, static_cast<uint32_t>(cats.size()));
                previous = &cats;
            }
        }

        // The condition bytes of the single-row path (see ObliviousForest::condition_begin), the
        // range test of every split, and the runs of trees of one depth.
        void index_conditions(ObliviousForest& f) {
            const size_t n_sources = f.n_sources();
            std::vector<uint32_t> count(n_sources, 0);
            for (size_t k = 0; k < f.split_source.size(); k++) {
                const uint16_t s = f.split_source[k];
                if (s >= n_sources) throw std::runtime_error("CatBoost split on an unknown source.");
                count[s] = std::max<uint32_t>(count[s], f.split_threshold[k]);
            }
            f.condition_begin.assign(1, 0);
            for (size_t s = 0; s < n_sources; s++) f.condition_begin.push_back(f.condition_begin.back() + count[s]);
            f.split_condition.resize(f.split_source.size());
            for (size_t k = 0; k < f.split_source.size(); k++) {
                // A threshold of 0 (`bucket >= 0`, always true) has no byte: it can only come from a
                // damaged file; byte 0 of an empty model keeps it in range.
                const uint16_t t = f.split_threshold[k];
                f.split_condition[k] = t ? f.condition_begin[f.split_source[k]] + t - 1 : 0;
            }
            f.split_range.resize(f.split_source.size());
            for (size_t k = 0; k < f.split_source.size(); k++) {
                f.split_range[k] = f.split_equal[k] ? uint16_t{0} : static_cast<uint16_t>(0xffff - f.split_threshold[k]);
            }
            // Condition bytes pay when sources are few next to splits: every source writes its
            // bytes per row. M4, 500 trees, bytes vs comparing buckets: california (8 sources, 3000
            // splits) 0.62 vs 0.92 µs, covtype (53) 1.02 vs 1.23, bank (268, 2969) 1.60 vs 1.67;
            // adult (364, 2981) 2.25 vs 2.08, diabetes (1228, 2943) 6.7 vs 5.1.
            f.row_conditions = n_sources * 10 <= f.split_source.size();
            f.run_end.clear();
            f.run_depth.clear();
            for (size_t t = 0; t < f.n_trees(); t++) {
                const uint32_t depth = f.split_begin[t + 1] - f.split_begin[t];
                if (depth > 255) throw std::runtime_error("CatBoost tree deeper than 255.");
                if (f.run_depth.empty() || f.run_depth.back() != depth) {
                    f.run_end.push_back(static_cast<uint32_t>(t));
                    f.run_depth.push_back(static_cast<uint8_t>(depth));
                }
                f.run_end.back() = static_cast<uint32_t>(t + 1);
            }
        }

        // Everything the kernel derives from a loaded forest.
        void derive_layout(ObliviousForest& f) {
            pad_borders(f);
            index_prefixes(f);
            index_conditions(f);
        }

        // bucket_of over a feature's padded borders without branches: a fixed number of halvings
        // (log2 of the padded size), each a conditional move. x is not NaN; +inf padding is never
        // below x, so the count is the same. 19% -> 9% of a numeric model's time on 200k rows.
        inline uint16_t bucket_search(const std::vector<float>& search, float x) {
            const float* base = search.data();
            size_t n = search.size();
            while (n > 1) {
                const size_t half = n / 2;
                base = base[half - 1] < x ? base + half : base;
                n -= half;
            }
            return static_cast<uint16_t>((base - search.data()) + (*base < x));
        }

        // Number of strictly increasing borders below x: the bucket with x > borders[k] <=> bucket >= k + 1.
        inline uint16_t bucket_of(const std::vector<float>& borders, float x) {
            return static_cast<uint16_t>(std::lower_bound(borders.begin(), borders.end(), x) - borders.begin());
        }

        // -----------------------------------------------------------------------------------------
        // JSON model

        [[noreturn]] void malformed(const std::string& what) {
            throw std::runtime_error("Malformed CatBoost model: " + what + ".");
        }

        uint64_t get_u64(simdjson::dom::element e) {
            if (e.is_string()) {
                const std::string_view s = e.get_string();
                uint64_t v = 0;
                const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
                if (ec != std::errc() || end != s.data() + s.size()) malformed("bad hash value");
                return v;
            }
            if (e.is_uint64()) return e.get_uint64();
            const int64_t v = e.get_int64();
            return static_cast<uint64_t>(v);
        }

        double get_number(simdjson::dom::element e) {
            if (e.is_double()) return e.get_double();
            if (e.is_int64()) return static_cast<double>(e.get_int64().value());
            if (e.is_uint64()) return static_cast<double>(e.get_uint64().value());
            malformed("expected a number");
        }

        template <typename T>
        T get_index(simdjson::dom::element e, size_t limit, const char* what) {
            const int64_t v = e.get_int64();
            if (v < 0 || static_cast<uint64_t>(v) >= limit) malformed(std::string(what) + " out of range");
            return static_cast<T>(v);
        }

        std::vector<float> float_list(simdjson::dom::array array, const char* what) {
            std::vector<float> values;
            for (simdjson::dom::element e : array) values.push_back(static_cast<float>(get_number(e)));
            for (size_t i = 1; i < values.size(); i++) {
                if (!(values[i - 1] < values[i])) malformed(std::string(what) + " are not strictly increasing");
            }
            return values;
        }

        // The output transform of CatBoost's default predict() for the loss.
        uint8_t transform_of(std::string_view loss, uint32_t dim) {
            if (loss == "Logloss" || loss == "CrossEntropy" || loss == "MultiClassOneVsAll") return transform::Sigmoid;
            if (loss == "MultiClass") return dim > 1 ? transform::Softmax : transform::Identity;
            if (loss == "Poisson" || loss == "Tweedie") return transform::Exp;
            return transform::Identity;
        }

        enum class CtrKind { Borders, Buckets, Counter, Mean };

        CtrKind ctr_kind(std::string_view type) {
            if (type == "Borders") return CtrKind::Borders;
            if (type == "Buckets") return CtrKind::Buckets;
            if (type == "Counter" || type == "FeatureFreq") return CtrKind::Counter;
            if (type == "BinarizedTargetMeanValue" || type == "FloatTargetMeanValue") return CtrKind::Mean;
            throw std::runtime_error("Unsupported CatBoost CTR type '" + std::string(type) + "'.");
        }

        // Learned statistics of one ctr_data entry: hash -> `stride - 1` numbers per entry.
        struct RawTable {
            std::vector<uint64_t> hashes;
            std::vector<double> values;  // entries x width
            size_t width = 0;
            int64_t denominator = 0;
        };

        RawTable read_table(simdjson::dom::object table) {
            RawTable raw;
            const int64_t stride = table["hash_stride"].get_int64();
            if (stride < 2 || stride > 1 << 16) malformed("bad hash_stride");
            raw.width = static_cast<size_t>(stride - 1);
            raw.denominator = table["counter_denominator"].get_int64();
            if (raw.denominator < 0 || raw.denominator > std::numeric_limits<int>::max()) malformed("bad counter_denominator");
            simdjson::dom::array map = table["hash_map"].get_array();
            if (map.size() % static_cast<size_t>(stride) != 0) malformed("hash_map size is not a multiple of hash_stride");
            size_t k = 0;
            for (simdjson::dom::element e : map) {
                if (k++ % static_cast<size_t>(stride) == 0) raw.hashes.push_back(get_u64(e));
                else {
                    // Counts are ints in CatBoost (and so converted below); mean sums are floats.
                    const double v = get_number(e);
                    if (!(std::fabs(v) <= std::numeric_limits<int>::max())) malformed("CTR statistic out of range");
                    raw.values.push_back(v);
                }
            }
            return raw;
        }

        // CatBoost's TModelCtr::Calc in float arithmetic: (count + prior) / (total + prior_denom),
        // then (ctr + shift) * scale.
        struct CtrFormula {
            float prior_num = 0, prior_den = 1, shift = 0, scale = 1;
            float operator()(float count, float total) const {
                const float ctr = (count + prior_num) / (total + prior_den);
                return (ctr + shift) * scale;
            }
        };

        // The CTR value of table entry `entry` (or of a missing combination when entry < 0), the
        // way TStaticCtrProvider::CalcCtrs reads the statistics for each CTR type.
        float ctr_value(CtrKind kind, const CtrFormula& calc, const RawTable& t, int64_t entry, size_t target_border) {
            if (entry < 0) return kind == CtrKind::Counter ? calc(0.0f, static_cast<float>(static_cast<int>(t.denominator))) : calc(0.0f, 0.0f);
            const double* h = &t.values[static_cast<size_t>(entry) * t.width];
            switch (kind) {
                case CtrKind::Counter:
                    return calc(static_cast<float>(static_cast<int>(h[0])), static_cast<float>(static_cast<int>(t.denominator)));
                case CtrKind::Mean:
                    if (t.width < 2) malformed("mean CTR table needs (sum, count)");
                    return calc(static_cast<float>(h[0]), static_cast<float>(static_cast<int>(h[1])));
                case CtrKind::Buckets: {
                    // CatBoost sums ints; int64 gives the same values without overflow.
                    int64_t good = static_cast<int64_t>(h[target_border]), total = 0;
                    for (size_t c = 0; c < t.width; c++) total += static_cast<int64_t>(h[c]);
                    return calc(static_cast<float>(good), static_cast<float>(total));
                }
                case CtrKind::Borders: {
                    // Binary targets: (class 1, class 0 + class 1); more classes: those above the border.
                    int64_t good = 0, total = 0;
                    for (size_t c = 0; c <= target_border; c++) total += static_cast<int64_t>(h[c]);
                    for (size_t c = target_border + 1; c < t.width; c++) good += static_cast<int64_t>(h[c]);
                    return calc(static_cast<float>(good), static_cast<float>(total + good));
                }
            }
            return 0.0f;
        }

        // -----------------------------------------------------------------------------------------
        // Library-neutral description of a CatBoost model, filled by the JSON and .cbm readers.
        // Features refer to CatBoost's feature indices; a split refers to a binary feature: float
        // borders (features in order), then one-hot values, then CTR borders.

        struct SpecFloat {
            int64_t index = 0;
            uint32_t column = 0;
            bool nan_as_true = false;
            std::vector<float> borders;
        };
        struct SpecCat {
            int64_t index = 0;
            uint32_t column = 0;
        };
        struct SpecOneHot {
            int64_t cat = 0;
            std::vector<uint32_t> values;
        };
        struct SpecCtr {
            std::vector<int64_t> cats;                         // hashed in this order
            std::vector<std::pair<int64_t, float>> floats;     // (float feature, border)
            std::vector<std::pair<int64_t, uint32_t>> one_hots;  // (categorical feature, value)
            std::string table;                                 // key into Spec::tables
            CtrKind kind = CtrKind::Borders;
            CtrFormula calc;
            int64_t target_border = 0;
            std::vector<float> borders;
        };
        struct Spec {
            std::vector<SpecFloat> floats;
            std::vector<SpecCat> cats;
            std::vector<SpecOneHot> one_hots;
            std::vector<SpecCtr> ctrs;
            std::map<std::string, RawTable> tables;
            std::vector<uint32_t> tree_depths;
            std::vector<int64_t> splits;  // binary feature of every split, tree after tree
            std::vector<double> leaves;
            uint32_t dim = 1;
            double scale = 1.0;
            std::vector<double> bias;
            std::string loss;
        };

        constexpr size_t kMaxColumns = 1 << 24;

        void check_borders(const std::vector<float>& borders, const char* what) {
            if (borders.size() > 65534) malformed(std::string("too many ") + what);
            for (size_t i = 1; i < borders.size(); i++) {
                if (!(borders[i - 1] < borders[i])) malformed(std::string(what) + " are not strictly increasing");
            }
        }

        Model build_model(Spec& spec) {
            auto forest = std::make_shared<ObliviousForest>();
            ObliviousForest& f = *forest;
            std::vector<std::pair<uint16_t, uint16_t>> binary;  // binary feature -> (source, threshold)
            std::vector<uint8_t> binary_equal;
            uint32_t n_columns = 0;

            std::unordered_map<int64_t, uint32_t> float_source;  // float feature index -> source
            for (SpecFloat& sf : spec.floats) {
                if (sf.column >= kMaxColumns) malformed("feature column out of range");
                n_columns = std::max(n_columns, sf.column + 1);
                if (sf.borders.empty()) continue;
                check_borders(sf.borders, "float feature borders");
                const uint32_t s = static_cast<uint32_t>(f.floats.size());
                if (!float_source.emplace(sf.index, s).second) malformed("duplicate float feature");
                for (size_t k = 0; k < sf.borders.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(0);
                }
                ObliviousForest::Float source;
                source.column = sf.column;
                source.nan_as_true = sf.nan_as_true;
                source.borders = std::move(sf.borders);
                f.floats.push_back(std::move(source));
            }
            std::unordered_map<int64_t, uint32_t> cat_index;  // categorical feature index -> cat_columns index
            for (const SpecCat& sc : spec.cats) {
                if (sc.column >= kMaxColumns) malformed("feature column out of range");
                n_columns = std::max(n_columns, sc.column + 1);
                if (!cat_index.emplace(sc.index, static_cast<uint32_t>(f.cat_columns.size())).second) malformed("duplicate categorical feature");
                f.cat_columns.push_back(sc.column);
            }
            const uint32_t n_floats = static_cast<uint32_t>(f.floats.size());
            std::unordered_map<int64_t, uint32_t> one_hot_source;  // categorical feature index -> one-hot index
            for (SpecOneHot& so : spec.one_hots) {
                auto it = cat_index.find(so.cat);
                if (it == cat_index.end()) malformed("one-hot feature of an unknown categorical feature");
                if (so.values.size() > 65534) malformed("too many one-hot values");
                const uint32_t h = static_cast<uint32_t>(f.one_hots.size());
                if (!one_hot_source.emplace(so.cat, h).second) malformed("duplicate one-hot feature");
                for (size_t k = 0; k < so.values.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(n_floats + h), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(1);
                }
                ObliviousForest::OneHot one_hot;
                one_hot.cat = it->second;
                one_hot.values = std::move(so.values);
                f.one_hots.push_back(std::move(one_hot));
            }

            // CTRs: projections shared by CTRs over the same combination, tables by key.
            std::map<std::string, uint32_t> projection_ids, table_ids;
            std::vector<const RawTable*> raw_tables;
            const uint32_t first_ctr_source = n_floats + static_cast<uint32_t>(f.one_hots.size());
            for (SpecCtr& sc : spec.ctrs) {
                ObliviousForest::Projection projection;
                std::string key;
                for (int64_t cat : sc.cats) {
                    auto it = cat_index.find(cat);
                    if (it == cat_index.end()) malformed("CTR uses an unknown categorical feature");
                    projection.cats.push_back(it->second);
                    key += "c" + std::to_string(cat) + ";";
                }
                for (const auto& [feature, border] : sc.floats) {
                    auto it = float_source.find(feature);
                    if (it == float_source.end()) malformed("CTR uses an unknown float feature");
                    const auto& borders = f.floats[it->second].borders;
                    auto pos = std::lower_bound(borders.begin(), borders.end(), border);
                    if (pos == borders.end() || *pos != border) malformed("CTR uses a float border the model does not have");
                    const auto k = static_cast<uint16_t>(pos - borders.begin() + 1);
                    projection.floats.emplace_back(it->second, k);
                    key += "f" + std::to_string(it->second) + ":" + std::to_string(k) + ";";
                }
                for (const auto& [feature, value] : sc.one_hots) {
                    auto it = one_hot_source.find(feature);
                    if (it == one_hot_source.end()) malformed("CTR uses a one-hot value the model does not have");
                    const auto& values = f.one_hots[it->second].values;
                    auto pos = std::find(values.begin(), values.end(), value);
                    if (pos == values.end()) malformed("CTR uses a one-hot value the model does not have");
                    const auto k = static_cast<uint16_t>(pos - values.begin() + 1);
                    projection.one_hots.emplace_back(n_floats + it->second, k);
                    key += "o" + std::to_string(it->second) + ":" + std::to_string(k) + ";";
                }
                if (projection.cats.empty()) malformed("CTR without categorical features");
                auto [pit, new_projection] = projection_ids.try_emplace(key, static_cast<uint32_t>(f.projections.size()));
                if (new_projection) f.projections.push_back(std::move(projection));

                auto raw_it = spec.tables.find(sc.table);
                if (raw_it == spec.tables.end()) malformed("no CTR table for a CTR");
                auto [tit, new_table] = table_ids.try_emplace(sc.table, static_cast<uint32_t>(raw_tables.size()));
                if (new_table) {
                    raw_tables.push_back(&raw_it->second);
                    ObliviousForest::Table t;
                    t.projection = pit->second;
                    f.projections[pit->second].tables.push_back(static_cast<uint32_t>(f.tables.size()));
                    f.tables.push_back(std::move(t));
                } else if (f.tables[tit->second].projection != pit->second) {
                    malformed("one CTR table for two combinations");
                }

                const RawTable& raw = *raw_tables[tit->second];
                if (raw.width == 0 || raw.values.size() != raw.hashes.size() * raw.width) malformed("CTR table size");
                if (sc.kind == CtrKind::Mean && raw.width < 2) malformed("mean CTR table needs (sum, count)");
                if (sc.target_border < 0 || static_cast<size_t>(sc.target_border) >= std::max<size_t>(raw.width, 1)) {
                    malformed("target_border_idx out of range");
                }
                const auto target_border = static_cast<size_t>(sc.target_border);
                check_borders(sc.borders, "CTR borders");
                ObliviousForest::Ctr ctr;
                ctr.table = tit->second;
                ctr.entry_buckets.resize(raw.hashes.size());
                for (size_t entry = 0; entry < raw.hashes.size(); entry++) {
                    ctr.entry_buckets[entry] = bucket_of(sc.borders, ctr_value(sc.kind, sc.calc, raw, static_cast<int64_t>(entry), target_border));
                }
                ctr.missing_bucket = bucket_of(sc.borders, ctr_value(sc.kind, sc.calc, raw, -1, target_border));
                const uint32_t s = first_ctr_source + static_cast<uint32_t>(f.ctrs.size());
                if (s > 65535) malformed("too many features");
                for (size_t k = 0; k < sc.borders.size(); k++) {
                    binary.emplace_back(static_cast<uint16_t>(s), static_cast<uint16_t>(k + 1));
                    binary_equal.push_back(0);
                }
                f.ctrs.push_back(std::move(ctr));
            }

            // Open-addressing index of every table (load factor <= 1/2). CatBoost marks unused
            // buckets of its own hash view with 2^64 - 1; they are not entries.
            for (size_t t = 0; t < f.tables.size(); t++) {
                ObliviousForest::Table& table = f.tables[t];
                const RawTable& raw = *raw_tables[t];
                uint64_t capacity = 2;
                while (capacity < 2 * raw.hashes.size()) capacity <<= 1;
                table.mask = capacity - 1;
                table.keys.assign(capacity, 0);
                table.slots.assign(capacity, 0);
                for (size_t entry = 0; entry < raw.hashes.size(); entry++) {
                    const uint64_t key = raw.hashes[entry];
                    if (key == ~uint64_t{0}) continue;
                    uint64_t slot = slot_of(key, table.mask);
                    while (table.slots[slot] != 0 && table.keys[slot] != key) slot = (slot + 1) & table.mask;
                    if (table.slots[slot] != 0) continue;  // duplicate hash: CatBoost finds the first
                    table.keys[slot] = key;
                    table.slots[slot] = static_cast<uint32_t>(entry + 1);
                }
            }

            if (spec.dim == 0 || spec.dim > 65535 || spec.bias.size() != spec.dim) malformed("bias");
            f.dim = spec.dim;
            f.scale = spec.scale;
            f.bias = spec.bias;

            f.split_begin.push_back(0);
            size_t split = 0, leaf = 0;
            for (uint32_t depth : spec.tree_depths) {
                if (depth > 16) malformed("tree deeper than 16");
                if (split + depth > spec.splits.size()) malformed("tree splits");
                for (uint32_t d = 0; d < depth; d++, split++) {
                    const int64_t index = spec.splits[split];
                    if (index < 0 || static_cast<size_t>(index) >= binary.size()) malformed("split index out of range");
                    f.split_source.push_back(binary[static_cast<size_t>(index)].first);
                    f.split_threshold.push_back(binary[static_cast<size_t>(index)].second);
                    f.split_equal.push_back(binary_equal[static_cast<size_t>(index)]);
                }
                f.split_begin.push_back(static_cast<uint32_t>(f.split_source.size()));
                f.leaf_begin.push_back(leaf);
                leaf += (size_t{1} << depth) * f.dim;
                if (leaf > spec.leaves.size()) malformed("leaf values size does not match the tree depths");
            }
            if (split != spec.splits.size() || leaf != spec.leaves.size()) malformed("tree sizes do not match the splits and leaves");
            f.leaves = std::move(spec.leaves);
            f.n_columns = n_columns;
            derive_layout(f);

            Model model;
            model.n_outputs = f.dim;
            model.transform = transform_of(spec.loss, f.dim);
            model.n_features = n_columns;
            model.base_scores = f.bias.data();
            model.oblivious = forest;
            model.data = forest;
            return model;
        }

        // -----------------------------------------------------------------------------------------
        // JSON (save_model(format="json"))

        Spec read_json(simdjson::dom::element doc) {
            Spec spec;
            simdjson::dom::object features = doc["features_info"].get_object();
            for (const char* unsupported : {"text_features", "embedding_features", "estimated_features"}) {
                simdjson::dom::array array;
                if (features[unsupported].get_array().get(array) == simdjson::SUCCESS && array.size() > 0) {
                    throw std::runtime_error(std::string("CatBoost models with ") + unsupported + " are not supported.");
                }
            }
            simdjson::dom::array trees;
            if (doc["oblivious_trees"].get_array().get(trees) != simdjson::SUCCESS) {
                throw std::runtime_error("Only CatBoost models with symmetric (oblivious) trees are supported; "
                                         "this one was trained with grow_policy Depthwise or Lossguide.");
            }
            simdjson::dom::array float_features;
            if (features["float_features"].get_array().get(float_features) == simdjson::SUCCESS) {
                for (simdjson::dom::object ff : float_features) {
                    SpecFloat sf;
                    sf.index = ff["feature_index"].get_int64();
                    sf.column = get_index<uint32_t>(ff["flat_feature_index"], kMaxColumns, "flat_feature_index");
                    simdjson::dom::array borders;
                    if (ff["borders"].get_array().get(borders) == simdjson::SUCCESS) sf.borders = float_list(borders, "float feature borders");
                    std::string_view nan_treatment = "AsIs";
                    (void)ff["nan_value_treatment"].get_string().get(nan_treatment);
                    sf.nan_as_true = nan_treatment == "AsTrue";
                    spec.floats.push_back(std::move(sf));
                }
            }
            simdjson::dom::array cat_features;
            if (features["categorical_features"].get_array().get(cat_features) == simdjson::SUCCESS) {
                for (simdjson::dom::object cf : cat_features) {
                    SpecCat sc;
                    sc.index = cf["feature_index"].get_int64();
                    sc.column = get_index<uint32_t>(cf["flat_feature_index"], kMaxColumns, "flat_feature_index");
                    spec.cats.push_back(sc);
                    simdjson::dom::array values;
                    if (cf["values"].get_array().get(values) == simdjson::SUCCESS && values.size() > 0) {
                        SpecOneHot so;
                        so.cat = sc.index;
                        for (simdjson::dom::element v : values) so.values.push_back(static_cast<uint32_t>(v.get_int64().value()));
                        spec.one_hots.push_back(std::move(so));
                    }
                }
            }
            simdjson::dom::object ctr_data;
            const bool has_ctr_data = doc["ctr_data"].get_object().get(ctr_data) == simdjson::SUCCESS;
            simdjson::dom::array ctrs;
            if (features["ctrs"].get_array().get(ctrs) == simdjson::SUCCESS) {
                for (simdjson::dom::object cj : ctrs) {
                    SpecCtr sc;
                    for (simdjson::dom::object e : cj["elements"].get_array()) {
                        const std::string_view kind = e["combination_element"].get_string();
                        if (kind == "cat_feature_value") sc.cats.push_back(e["cat_feature_index"].get_int64());
                        else if (kind == "float_feature") sc.floats.emplace_back(e["float_feature_index"].get_int64(), static_cast<float>(get_number(e["border"])));
                        else if (kind == "cat_feature_exact_value") sc.one_hots.emplace_back(e["cat_feature_index"].get_int64(), static_cast<uint32_t>(e["value"].get_int64().value()));
                        else throw std::runtime_error("Unsupported CatBoost CTR element '" + std::string(kind) + "'.");
                    }
                    sc.table = std::string(cj["identifier"].get_string().value());
                    if (!spec.tables.count(sc.table)) {
                        simdjson::dom::object table;
                        if (!has_ctr_data || ctr_data[sc.table].get_object().get(table) != simdjson::SUCCESS) malformed("ctr_data has no table for " + sc.table);
                        spec.tables.emplace(sc.table, read_table(table));
                    }
                    sc.kind = ctr_kind(cj["ctr_type"].get_string());
                    sc.calc.prior_num = static_cast<float>(get_number(cj["prior_numerator"]));
                    sc.calc.prior_den = static_cast<float>(get_number(cj["prior_denomerator"]));
                    sc.calc.shift = static_cast<float>(get_number(cj["shift"]));
                    sc.calc.scale = static_cast<float>(get_number(cj["scale"]));
                    sc.target_border = cj["target_border_idx"].get_int64();
                    sc.borders = float_list(cj["borders"].get_array(), "CTR borders");
                    spec.ctrs.push_back(std::move(sc));
                }
            }
            simdjson::dom::array scale_and_bias = doc["scale_and_bias"].get_array();
            if (scale_and_bias.size() != 2) malformed("scale_and_bias");
            spec.scale = get_number(scale_and_bias.at(0));
            for (simdjson::dom::element b : scale_and_bias.at(1).get_array()) spec.bias.push_back(get_number(b));
            spec.dim = static_cast<uint32_t>(std::min<size_t>(spec.bias.size(), 1 << 20));
            for (simdjson::dom::object tree : trees) {
                // A tree of depth 0 (a single leaf) has "splits": null.
                simdjson::dom::array splits;
                uint32_t depth = 0;
                if (tree["splits"].get_array().get(splits) == simdjson::SUCCESS) {
                    if (splits.size() > 16) malformed("tree deeper than 16");
                    depth = static_cast<uint32_t>(splits.size());
                    for (simdjson::dom::object split : splits) spec.splits.push_back(split["split_index"].get_int64());
                } else if (!tree["splits"].is_null()) {
                    malformed("splits");
                }
                spec.tree_depths.push_back(depth);
                for (simdjson::dom::element v : tree["leaf_values"].get_array()) spec.leaves.push_back(get_number(v));
            }
            simdjson::dom::element loss_type;
            if (doc.at_pointer("/model_info/params/loss_function/type").get(loss_type) == simdjson::SUCCESS) spec.loss = std::string(loss_type.get_string().value());
            return spec;
        }

        // -----------------------------------------------------------------------------------------
        // .cbm: "CBM1", a size, the TModelCore FlatBuffer, then the CTR tables (a count, then a size
        // and a TCtrValueTable FlatBuffer each). Read by hand (no flatbuffers dependency), every
        // offset checked against the buffer.

        class Flat {
            public:
                Flat(const uint8_t* data, size_t size) : data_(data), size_(size) {}

                template <typename T>
                T read(size_t pos) const {
                    if (pos > size_ || size_ - pos < sizeof(T)) malformed("FlatBuffer offset out of range");
                    T v;
                    std::memcpy(&v, data_ + pos, sizeof(T));
                    return v;
                }
                size_t size() const { return size_; }
                const uint8_t* at(size_t pos) const { return data_ + pos; }

            private:
                const uint8_t* data_;
                size_t size_;
        };

        struct FlatVector {
            const Flat* flat = nullptr;
            size_t start = 0;  // first element
            uint32_t length = 0;

            size_t element(size_t k, size_t width) const { return start + k * width; }
        };

        struct FlatTable {
            const Flat* flat = nullptr;
            size_t pos = 0;
            size_t vtable = 0;
            uint16_t vtable_size = 0;

            // Position of field i, or 0 when absent.
            size_t field(int i) const {
                const size_t entry = 4 + 2 * static_cast<size_t>(i);
                if (entry + 2 > vtable_size) return 0;
                const uint16_t offset = flat->read<uint16_t>(vtable + entry);
                return offset ? pos + offset : 0;
            }
            template <typename T>
            T scalar(int i, T fallback) const {
                const size_t p = field(i);
                return p ? flat->read<T>(p) : fallback;
            }
            bool has(int i) const { return field(i) != 0; }
            // The target of the uoffset stored at `at`.
            size_t follow(size_t at) const {
                const uint32_t offset = flat->read<uint32_t>(at);
                if (offset == 0 || at + offset >= flat->size()) malformed("FlatBuffer offset out of range");
                return at + offset;
            }
            FlatVector vector(int i, size_t width) const {
                FlatVector v;
                v.flat = flat;
                const size_t p = field(i);
                if (!p) return v;
                const size_t target = follow(p);
                v.length = flat->read<uint32_t>(target);
                v.start = target + 4;
                if (v.length > (flat->size() - v.start) / std::max<size_t>(width, 1)) malformed("FlatBuffer vector out of range");
                return v;
            }
            std::string_view string(int i) const {
                const FlatVector v = vector(i, 1);
                return v.length ? std::string_view(reinterpret_cast<const char*>(flat->at(v.start)), v.length) : std::string_view();
            }
        };

        FlatTable table_at(const Flat& flat, size_t pos) {
            FlatTable t;
            t.flat = &flat;
            t.pos = pos;
            const int64_t vtable = static_cast<int64_t>(pos) - flat.read<int32_t>(pos);
            if (vtable < 0 || static_cast<uint64_t>(vtable) + 4 > flat.size()) malformed("FlatBuffer vtable out of range");
            t.vtable = static_cast<size_t>(vtable);
            t.vtable_size = flat.read<uint16_t>(t.vtable);
            if (t.vtable_size < 4 || t.vtable + t.vtable_size > flat.size()) malformed("FlatBuffer vtable out of range");
            return t;
        }
        FlatTable root(const Flat& flat) { return table_at(flat, static_cast<size_t>(flat.read<uint32_t>(0))); }
        // Field i of t as a table; `present` is false when it is absent.
        FlatTable subtable(const FlatTable& t, int i, bool* present = nullptr) {
            const size_t p = t.field(i);
            if (present) *present = p != 0;
            if (!p) {
                if (!present) malformed("missing FlatBuffer table");
                return FlatTable{};
            }
            return table_at(*t.flat, t.follow(p));
        }
        FlatTable table_in(const FlatVector& v, size_t k) {
            const size_t at = v.element(k, 4);
            FlatTable dummy;
            dummy.flat = v.flat;
            return table_at(*v.flat, dummy.follow(at));
        }
        template <typename T>
        std::vector<T> scalars(const FlatTable& t, int i) {
            const FlatVector v = t.vector(i, sizeof(T));
            std::vector<T> out(v.length);
            for (uint32_t k = 0; k < v.length; k++) out[k] = v.flat->template read<T>(v.element(k, sizeof(T)));
            return out;
        }

        // TModelCtrBase as a key: the same in the model's CTR features and in its CTR tables.
        std::string ctr_base_key(const FlatTable& base, SpecCtr* ctr) {
            bool has_combination = false;
            const FlatTable combination = subtable(base, 0, &has_combination);
            std::string key;
            if (has_combination) {
                for (int32_t cat : scalars<int32_t>(combination, 0)) {
                    key += "c" + std::to_string(cat) + ";";
                    if (ctr) ctr->cats.push_back(cat);
                }
                const FlatVector floats = combination.vector(1, 8);
                for (uint32_t k = 0; k < floats.length; k++) {
                    const int32_t index = floats.flat->read<int32_t>(floats.element(k, 8));
                    const float border = floats.flat->read<float>(floats.element(k, 8) + 4);
                    uint32_t bits;
                    std::memcpy(&bits, &border, 4);
                    key += "f" + std::to_string(index) + ":" + std::to_string(bits) + ";";
                    if (ctr) ctr->floats.emplace_back(index, border);
                }
                const FlatVector one_hots = combination.vector(2, 8);
                for (uint32_t k = 0; k < one_hots.length; k++) {
                    const int32_t index = one_hots.flat->read<int32_t>(one_hots.element(k, 8));
                    const int32_t value = one_hots.flat->read<int32_t>(one_hots.element(k, 8) + 4);
                    key += "o" + std::to_string(index) + ":" + std::to_string(value) + ";";
                    if (ctr) ctr->one_hots.emplace_back(index, static_cast<uint32_t>(value));
                }
            }
            const int8_t type = base.scalar<int8_t>(1, 0);
            key += "t" + std::to_string(type) + ":" + std::to_string(base.scalar<int32_t>(2, 0));
            if (ctr) {
                switch (type) {
                    case 0: ctr->kind = CtrKind::Borders; break;
                    case 1: ctr->kind = CtrKind::Buckets; break;
                    case 2: case 3: ctr->kind = CtrKind::Mean; break;
                    case 4: case 5: ctr->kind = CtrKind::Counter; break;
                    default: throw std::runtime_error("Unsupported CatBoost CTR type " + std::to_string(type) + ".");
                }
            }
            return key;
        }

        size_t read_size(const Flat& file, size_t& pos) {
            uint64_t size = file.read<uint32_t>(pos);
            pos += 4;
            if (size == 0xffffffffu) {
                size = file.read<uint64_t>(pos);
                pos += 8;
            }
            if (size > file.size() - pos) malformed("section size out of range");
            return static_cast<size_t>(size);
        }

        Spec read_cbm(const std::string& bytes) {
            const Flat file(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
            if (bytes.size() < 8 || bytes.compare(0, 4, "CBM1") != 0) malformed("not a CatBoost .cbm file");
            size_t pos = 4;
            const size_t core_size = read_size(file, pos);
            const Flat core(file.at(pos), core_size);
            pos += core_size;

            Spec spec;
            const FlatTable model = root(core);
            if (model.string(0) != "FlabuffersModel_v1") malformed("unknown .cbm core format");
            const FlatVector info = model.vector(2, 4);
            for (uint32_t k = 0; k < info.length; k++) {
                const FlatTable kv = table_in(info, k);
                if (kv.string(0) != "params") continue;
                // The training parameters, as JSON: only the loss decides the output transform.
                simdjson::dom::parser parser;
                simdjson::dom::element params;
                simdjson::dom::element type;
                const std::string_view text = kv.string(1);
                if (parser.parse(text.data(), text.size()).get(params) == simdjson::SUCCESS &&
                    params.at_pointer("/loss_function/type").get(type) == simdjson::SUCCESS && type.is_string()) {
                    spec.loss = std::string(type.get_string().value());
                }
            }
            bool has_ctr_part = false;
            const FlatVector parts = model.vector(3, 4);
            for (uint32_t k = 0; k < parts.length; k++) {
                const size_t at = parts.element(k, 4);
                FlatTable dummy;
                dummy.flat = &core;
                const size_t target = dummy.follow(at);
                const uint32_t length = core.read<uint32_t>(target);
                if (length > core.size() - target - 4) malformed("FlatBuffer string out of range");
                const std::string_view part(reinterpret_cast<const char*>(core.at(target + 4)), length);
                if (part == "static_provider_v1") has_ctr_part = true;
                else throw std::runtime_error("CatBoost models with " + std::string(part) + " (text or embedding features) are not supported.");
            }

            const FlatTable trees = subtable(model, 1);
            spec.dim = static_cast<uint32_t>(std::clamp<int32_t>(trees.scalar<int32_t>(0, 1), 0, 1 << 20));
            for (int i : {10, 11}) {
                if (trees.vector(i, 4).length > 0) {
                    throw std::runtime_error("Only CatBoost models with symmetric (oblivious) trees are supported; "
                                             "this one was trained with grow_policy Depthwise or Lossguide.");
                }
            }
            for (int i : {12, 13, 18}) {
                if (trees.vector(i, 4).length > 0) throw std::runtime_error("CatBoost models with text, embedding or estimated features are not supported.");
            }
            for (int32_t s : scalars<int32_t>(trees, 1)) spec.splits.push_back(s);
            for (int32_t d : scalars<int32_t>(trees, 2)) {
                if (d < 0 || d > 16) malformed("tree deeper than 16");
                spec.tree_depths.push_back(static_cast<uint32_t>(d));
            }
            const FlatVector cats = trees.vector(4, 4);
            for (uint32_t k = 0; k < cats.length; k++) {
                const FlatTable c = table_in(cats, k);
                SpecCat sc;
                sc.index = c.scalar<int32_t>(0, -1);
                const int32_t column = c.scalar<int32_t>(1, -1);
                if (column < 0) malformed("feature column out of range");
                sc.column = static_cast<uint32_t>(column);
                spec.cats.push_back(sc);
            }
            const FlatVector floats = trees.vector(5, 4);
            for (uint32_t k = 0; k < floats.length; k++) {
                const FlatTable fl = table_in(floats, k);
                SpecFloat sf;
                sf.index = fl.scalar<int32_t>(1, -1);
                const int32_t column = fl.scalar<int32_t>(2, -1);
                if (column < 0) malformed("feature column out of range");
                sf.column = static_cast<uint32_t>(column);
                sf.borders = scalars<float>(fl, 3);
                sf.nan_as_true = fl.scalar<int8_t>(5, 0) == 2;  // ENanValueTreatment::AsTrue
                spec.floats.push_back(std::move(sf));
            }
            const FlatVector one_hots = trees.vector(6, 4);
            for (uint32_t k = 0; k < one_hots.length; k++) {
                const FlatTable oh = table_in(one_hots, k);
                SpecOneHot so;
                so.cat = oh.scalar<int32_t>(0, -1);
                for (int32_t v : scalars<int32_t>(oh, 1)) so.values.push_back(static_cast<uint32_t>(v));
                spec.one_hots.push_back(std::move(so));
            }
            const FlatVector ctrs = trees.vector(7, 4);
            for (uint32_t k = 0; k < ctrs.length; k++) {
                const FlatTable feature = table_in(ctrs, k);
                const FlatTable ctr = subtable(feature, 0);
                SpecCtr sc;
                sc.table = ctr_base_key(subtable(ctr, 0), &sc);
                sc.target_border = ctr.scalar<int32_t>(1, 0);
                sc.calc.prior_num = ctr.scalar<float>(2, 0.0f);
                sc.calc.prior_den = ctr.scalar<float>(3, 1.0f);
                sc.calc.shift = ctr.scalar<float>(4, 0.0f);
                sc.calc.scale = ctr.scalar<float>(5, 1.0f);
                sc.borders = scalars<float>(feature, 1);
                spec.ctrs.push_back(std::move(sc));
            }
            spec.leaves = scalars<double>(trees, 8);
            spec.scale = trees.scalar<double>(14, 1.0);
            spec.bias = scalars<double>(trees, 16);
            if (spec.bias.empty()) spec.bias.assign(spec.dim, trees.scalar<double>(15, 0.0));

            // CTR tables: TBucket {ui64 hash; ui32 index} (packed) -> statistics in the blob.
            if (has_ctr_part) {
                const size_t n_tables = read_size(file, pos);
                for (size_t t = 0; t < n_tables; t++) {
                    const size_t size = read_size(file, pos);
                    const Flat buffer(file.at(pos), size);
                    pos += size;
                    const FlatTable table = root(buffer);
                    SpecCtr probe;
                    const std::string key = ctr_base_key(subtable(table, 0), &probe);
                    const FlatVector buckets = table.vector(1, 1);
                    const FlatVector blob = table.vector(2, 1);
                    RawTable raw;
                    raw.denominator = table.scalar<int32_t>(3, 0);
                    if (raw.denominator < 0) malformed("bad counter_denominator");
                    const int32_t classes = table.scalar<int32_t>(4, 0);
                    size_t entries = 0;
                    if (probe.kind == CtrKind::Mean) {
                        raw.width = 2;
                        entries = blob.length / 8;
                        for (size_t e = 0; e < entries; e++) {
                            raw.values.push_back(static_cast<double>(buffer.read<float>(blob.start + 8 * e)));
                            raw.values.push_back(static_cast<double>(buffer.read<int32_t>(blob.start + 8 * e + 4)));
                        }
                    } else {
                        raw.width = probe.kind == CtrKind::Counter ? 1 : static_cast<size_t>(classes);
                        if (raw.width == 0 || raw.width > (1 << 16)) malformed("bad TargetClassesCount");
                        entries = blob.length / (4 * raw.width);
                        for (size_t e = 0; e < entries * raw.width; e++) raw.values.push_back(static_cast<double>(buffer.read<int32_t>(blob.start + 4 * e)));
                    }
                    raw.hashes.assign(entries, ~uint64_t{0});
                    for (size_t b = 0; b + 12 <= buckets.length; b += 12) {
                        const uint64_t hash = buffer.read<uint64_t>(buckets.start + b);
                        const uint32_t index = buffer.read<uint32_t>(buckets.start + b + 8);
                        if (hash == ~uint64_t{0}) continue;
                        if (index >= entries) malformed("CTR bucket index out of range");
                        raw.hashes[index] = hash;
                    }
                    spec.tables[key] = std::move(raw);
                }
            }
            return spec;
        }

    } // namespace

    namespace parsers {

        Model load_catboost_json(const std::string& filepath) {
            simdjson::dom::parser parser;
            simdjson::dom::element doc;
            if (auto error = parser.load(filepath).get(doc)) {
                throw std::runtime_error("Cannot parse CatBoost JSON " + filepath + ": " + simdjson::error_message(error));
            }
            try {
                Spec spec = read_json(doc);
                return build_model(spec);
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed CatBoost model " + filepath + ": " + e.what());
            }
        }

        Model load_catboost_cbm(const std::string& filepath) {
            try {
                Spec spec = read_cbm(read_file(filepath));
                return build_model(spec);
            } catch (const simdjson::simdjson_error& e) {
                throw std::runtime_error("Malformed CatBoost model " + filepath + ": " + e.what());
            }
        }

    } // namespace parsers

    // ---------------------------------------------------------------------------------------------
    // Kernel

    namespace {

        constexpr size_t kBlock = 128;
        // Batches below this many rows take the per-row path (see predict_row).
        #ifndef LIGNUM_CATBOOST_ROW_PATH
            #define LIGNUM_CATBOOST_ROW_PATH 4
        #endif
        constexpr size_t kRowPathMax = LIGNUM_CATBOOST_ROW_PATH;

        // Scratch of one thread: buckets of every source for a block (column-major, so the
        // per-tree loops over rows vectorise), category hashes, table entries, leaf indices.
        // Kept per thread and only grown, so a single row allocates and clears nothing. Arrays
        // are [source][row] with a stride of the block's row count.
        struct Scratch {
            std::vector<uint16_t> buckets;
            std::vector<uint32_t> hashes;
            std::vector<uint32_t> entries;
            std::vector<uint64_t> combination;
            std::vector<uint64_t> prefixes;  // [level][row]: hash of the current prefix-tree node per level
            std::vector<uint16_t> index;
            std::vector<double> sums;
            std::vector<uint8_t> conditions;  // single-row path

            void prepare(const ObliviousForest& f, size_t rows) {
                auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
                grow(buckets, f.n_sources() * rows);
                grow(hashes, f.cat_columns.size() * rows);
                grow(entries, f.tables.size() * rows);
                grow(combination, rows);
                grow(prefixes, f.prefix_levels * rows);
                grow(index, rows);
                grow(sums, rows * f.dim);
                if (rows == 1 && f.row_conditions) grow(conditions, f.condition_begin.back() + 8);  // + 8: stores of 8 bytes
            }
        };

        Scratch& thread_scratch() {
            thread_local Scratch scratch;
            return scratch;
        }

        // Leaf indices (16 bits: depth <= 16) of one tree for a block of rows. With the depth known
        // at compile time, all levels are done in one pass over the rows with the index in a
        // register. Both split kinds are one unsigned range test, `(uint16_t)(bucket - t) <= range`:
        // range 0xffff - t for `bucket >= t`, 0 for the one-hot `bucket == t`. (Two comparisons
        // joined with `&` stopped the loop from vectorising: 3x slower.)
        template <int Depth>
        void tree_index(const uint16_t* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                        const uint8_t* equal, size_t rows, uint16_t* index) {
            const uint16_t* b[Depth];
            uint16_t lo[Depth], range[Depth];
            for (int d = 0; d < Depth; d++) {
                b[d] = buckets + source[d] * stride;
                lo[d] = threshold[d];
                range[d] = equal[d] ? uint16_t{0} : static_cast<uint16_t>(0xffff - threshold[d]);
            }
            for (size_t r = 0; r < rows; r++) {
                uint16_t v = 0;
                for (int d = 0; d < Depth; d++) v |= static_cast<uint16_t>((static_cast<uint16_t>(b[d][r] - lo[d]) <= range[d]) << d);
                index[r] = v;
            }
        }

        // Any depth, one pass over the rows per level.
        void tree_index_any(const uint16_t* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                            const uint8_t* equal, uint32_t depth, size_t rows, uint16_t* index) {
            std::fill(index, index + rows, uint16_t{0});
            for (uint32_t d = 0; d < depth; d++) {
                const uint16_t* b = buckets + source[d] * stride;
                const uint16_t t = threshold[d];
                if (equal[d]) {
                    for (size_t r = 0; r < rows; r++) index[r] |= static_cast<uint16_t>((b[r] == t) << d);
                } else {
                    for (size_t r = 0; r < rows; r++) index[r] |= static_cast<uint16_t>((b[r] >= t) << d);
                }
            }
        }

        void tree_index_dispatch(const uint16_t* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                                 const uint8_t* equal, uint32_t depth, size_t rows, uint16_t* index) {
            switch (depth) {
                case 0: std::fill(index, index + rows, uint16_t{0}); break;
                case 1: tree_index<1>(buckets, stride, source, threshold, equal, rows, index); break;
                case 2: tree_index<2>(buckets, stride, source, threshold, equal, rows, index); break;
                case 3: tree_index<3>(buckets, stride, source, threshold, equal, rows, index); break;
                case 4: tree_index<4>(buckets, stride, source, threshold, equal, rows, index); break;
                case 5: tree_index<5>(buckets, stride, source, threshold, equal, rows, index); break;
                case 6: tree_index<6>(buckets, stride, source, threshold, equal, rows, index); break;
                case 7: tree_index<7>(buckets, stride, source, threshold, equal, rows, index); break;
                case 8: tree_index<8>(buckets, stride, source, threshold, equal, rows, index); break;
                case 9: tree_index<9>(buckets, stride, source, threshold, equal, rows, index); break;
                case 10: tree_index<10>(buckets, stride, source, threshold, equal, rows, index); break;
                default: tree_index_any(buckets, stride, source, threshold, equal, depth, rows, index); break;
            }
        }

        // Phase 2 for a block: leaf index per tree, leaves added in model order (one double addition
        // per tree and output). Dim outputs per row, 0: any, from the model. The sums are a local
        // array: on the heap the compiler cannot keep them apart from the leaves (7 classes: 0.149
        // instead of 0.100 s for 200k rows).
        template <int Dim>
        void phase2(const ObliviousForest& f, const uint16_t* buckets, size_t stride, size_t rows, Scratch& s, double* out) {
            double local[Dim > 0 ? kBlock * Dim : 1];
            const size_t dim = Dim > 0 ? Dim : f.dim;
            double* sums = Dim > 0 ? local : s.sums.data();
            std::fill(sums, sums + rows * dim, 0.0);
            uint16_t* index = s.index.data();
            for (size_t t = 0; t < f.n_trees(); t++) {
                const uint32_t k = f.split_begin[t];
                tree_index_dispatch(buckets, stride, f.split_source.data() + k, f.split_threshold.data() + k, f.split_equal.data() + k,
                                    f.split_begin[t + 1] - k, rows, index);
                const double* leaves = f.leaves.data() + f.leaf_begin[t];
                if constexpr (Dim > 0) {
                    for (size_t r = 0; r < rows; r++) {
                        const double* l = leaves + index[r] * Dim;
                        for (int d = 0; d < Dim; d++) sums[r * Dim + d] += l[d];
                    }
                } else {
                    for (size_t r = 0; r < rows; r++) {
                        const double* l = leaves + index[r] * dim;
                        for (size_t d = 0; d < dim; d++) sums[r * dim + d] += l[d];
                    }
                }
            }
            for (size_t r = 0; r < rows; r++) {
                for (size_t d = 0; d < dim; d++) out[r * dim + d] = f.scale * sums[r * dim + d] + f.bias[d];
            }
        }

        void predict_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s) {
            const size_t F = f.floats.size(), H = f.one_hots.size();
            const size_t stride = rows;
            s.prepare(f, rows);
            uint16_t* buckets = s.buckets.data();

            // Phase 1: buckets of every source.
            for (size_t k = 0; k < F; k++) {
                const ObliviousForest::Float& ff = f.floats[k];
                const uint16_t nan_bucket = ff.nan_as_true ? static_cast<uint16_t>(ff.borders.size()) : 0;
                uint16_t* b = buckets + k * stride;
                for (size_t r = 0; r < rows; r++) {
                    const float x = static_cast<float>(X[r * n_features + ff.column]);
                    b[r] = std::isnan(x) ? nan_bucket : bucket_search(ff.search, x);
                }
            }
            for (size_t c = 0; c < f.cat_columns.size(); c++) {
                uint32_t* h = s.hashes.data() + c * stride;
                const uint32_t column = f.cat_columns[c];
                for (size_t r = 0; r < rows; r++) h[r] = static_cast<uint32_t>(X[r * n_features + column]);
            }
            for (size_t k = 0; k < H; k++) {
                const ObliviousForest::OneHot& oh = f.one_hots[k];
                const uint32_t* h = s.hashes.data() + oh.cat * stride;
                uint16_t* b = buckets + (F + k) * stride;
                for (size_t r = 0; r < rows; r++) {
                    uint16_t v = 0;
                    for (size_t j = 0; j < oh.values.size(); j++) v = oh.values[j] == h[r] ? static_cast<uint16_t>(j + 1) : v;
                    b[r] = v;
                }
            }
            // CTRs: the category hashes of every prefix-tree node (a prefix shared by projections is
            // hashed once), then per projection its float / one-hot conditions, one lookup per table
            // and a bucket per CTR.
            for (size_t n = 0; n < f.prefix_cat.size(); n++) {
                const uint32_t level = f.prefix_level[n];
                uint64_t* hash = s.prefixes.data() + level * stride;
                const uint64_t* parent = level ? hash - stride : nullptr;
                const uint32_t* h = s.hashes.data() + f.prefix_cat[n] * stride;
                for (size_t r = 0; r < rows; r++) {
                    hash[r] = combine_hash(parent ? parent[r] : 0, static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(h[r]))));
                }
                for (uint32_t i = f.prefix_projection_begin[n]; i < f.prefix_projection_begin[n + 1]; i++) {
                    const ObliviousForest::Projection& proj = f.projections[f.prefix_projections[i]];
                    const uint64_t* combination = hash;
                    if (!proj.floats.empty() || !proj.one_hots.empty()) {
                        uint64_t* c = s.combination.data();
                        std::copy(hash, hash + rows, c);
                        for (const auto& [source, threshold] : proj.floats) {
                            const uint16_t* b = buckets + source * stride;
                            for (size_t r = 0; r < rows; r++) c[r] = combine_hash(c[r], b[r] >= threshold ? 1 : 0);
                        }
                        for (const auto& [source, value] : proj.one_hots) {
                            const uint16_t* b = buckets + source * stride;
                            for (size_t r = 0; r < rows; r++) c[r] = combine_hash(c[r], b[r] == value ? 1 : 0);
                        }
                        combination = c;
                    }
                    for (uint32_t t : proj.tables) {
                        const ObliviousForest::Table& table = f.tables[t];
                        uint32_t* entries = s.entries.data() + t * stride;
                        for (size_t r = 0; r < rows; r++) {
                            uint64_t slot = slot_of(combination[r], table.mask);
                            uint32_t found = 0;
                            while (table.slots[slot] != 0) {
                                if (table.keys[slot] == combination[r]) { found = table.slots[slot]; break; }
                                slot = (slot + 1) & table.mask;
                            }
                            entries[r] = found;
                        }
                    }
                }
            }
            for (size_t k = 0; k < f.ctrs.size(); k++) {
                const ObliviousForest::Ctr& ctr = f.ctrs[k];
                const uint32_t* entries = s.entries.data() + ctr.table * stride;
                uint16_t* b = buckets + (F + H + k) * stride;
                for (size_t r = 0; r < rows; r++) b[r] = entries[r] ? ctr.entry_buckets[entries[r] - 1] : ctr.missing_bucket;
            }

            switch (f.dim) {
                case 1: phase2<1>(f, buckets, stride, rows, s, out); break;
                case 2: phase2<2>(f, buckets, stride, rows, s, out); break;
                case 3: phase2<3>(f, buckets, stride, rows, s, out); break;
                case 4: phase2<4>(f, buckets, stride, rows, s, out); break;
                case 5: phase2<5>(f, buckets, stride, rows, s, out); break;
                case 6: phase2<6>(f, buckets, stride, rows, s, out); break;
                case 7: phase2<7>(f, buckets, stride, rows, s, out); break;
                case 8: phase2<8>(f, buckets, stride, rows, s, out); break;
                case 9: phase2<9>(f, buckets, stride, rows, s, out); break;
                case 10: phase2<10>(f, buckets, stride, rows, s, out); break;
                default: phase2<0>(f, buckets, stride, rows, s, out); break;
            }
        }

        // Trees [t0, t1) of one row, all of depth Depth (known at compile time: a tree's leaf index
        // is Depth steps without a loop). Bytes: each split is one byte of the row's condition array
        // (ObliviousForest::condition_begin); otherwise it compares the source's bucket with one
        // unsigned range test. Dim = 0: any number of outputs. The sums carry over from the
        // previous run, so leaves are still added in model order.
        // M4, 500 trees of depth 6: california 1.58 -> 0.62 µs per row (bytes); the chain of 500
        // dependent additions alone takes 0.25 µs.
        template <int Depth, int Dim, bool Bytes>
        void row_trees(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* io) {
            const size_t dim = Dim > 0 ? Dim : f.dim;
            double local[Dim > 0 ? Dim : 1];
            double* sums = Dim > 0 ? local : io;
            if constexpr (Dim > 0) std::copy(io, io + Dim, local);
            const double* leaves = f.leaves.data() + f.leaf_begin[t0];
            const size_t leaves_per_tree = dim << Depth;
            for (size_t t = t0, k = f.split_begin[t0]; t < t1; t++, k += Depth, leaves += leaves_per_tree) {
                uint32_t index = 0;
                for (int d = 0; d < Depth; d++) {
                    if constexpr (Bytes) {
                        index |= static_cast<uint32_t>(bits[f.split_condition[k + d]]) << d;
                    } else {
                        const uint16_t v = static_cast<uint16_t>(buckets[f.split_source[k + d]] - f.split_threshold[k + d]);
                        index |= static_cast<uint32_t>(v <= f.split_range[k + d]) << d;
                    }
                }
                const double* l = leaves + index * dim;
                for (size_t d = 0; d < dim; d++) sums[d] += l[d];
            }
            if constexpr (Dim > 0) std::copy(local, local + Dim, io);
        }

        // Trees [t0, t1) of any depths (deeper than the specialised ones).
        void row_trees_any(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, double* sums) {
            const size_t dim = f.dim;
            for (size_t t = t0; t < t1; t++) {
                uint32_t index = 0;
                for (uint32_t k = f.split_begin[t], bit = 0; k < f.split_begin[t + 1]; k++, bit++) {
                    const uint16_t v = static_cast<uint16_t>(buckets[f.split_source[k]] - f.split_threshold[k]);
                    index |= static_cast<uint32_t>(v <= f.split_range[k]) << bit;
                }
                const double* l = f.leaves.data() + f.leaf_begin[t] + index * dim;
                for (size_t d = 0; d < dim; d++) sums[d] += l[d];
            }
        }

        template <int Depth, bool Bytes>
        void row_trees_dim(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* sums) {
            switch (f.dim) {
                case 1: row_trees<Depth, 1, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 2: row_trees<Depth, 2, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 3: row_trees<Depth, 3, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 4: row_trees<Depth, 4, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 5: row_trees<Depth, 5, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 6: row_trees<Depth, 6, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 7: row_trees<Depth, 7, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 8: row_trees<Depth, 8, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 9: row_trees<Depth, 9, Bytes>(f, t0, t1, buckets, bits, sums); break;
                case 10: row_trees<Depth, 10, Bytes>(f, t0, t1, buckets, bits, sums); break;
                default: row_trees<Depth, 0, Bytes>(f, t0, t1, buckets, bits, sums); break;
            }
        }

        template <int Depth>
        void row_trees_kind(const ObliviousForest& f, size_t t0, size_t t1, const uint16_t* buckets, const uint8_t* bits, double* sums) {
            if (bits) row_trees_dim<Depth, true>(f, t0, t1, buckets, bits, sums);
            else row_trees_dim<Depth, false>(f, t0, t1, buckets, bits, sums);
        }

        // All trees of one row, run by run. bits: the row's condition bytes, or null to compare buckets.
        void row_trees_all(const ObliviousForest& f, const uint16_t* buckets, const uint8_t* bits, double* out, Scratch& s) {
            double* sums = s.sums.data();
            std::fill(sums, sums + f.dim, 0.0);
            for (size_t r = 0, t0 = 0; r < f.run_end.size(); t0 = f.run_end[r], r++) {
                const size_t t1 = f.run_end[r];
                switch (f.run_depth[r]) {
                    case 0: row_trees_kind<0>(f, t0, t1, buckets, bits, sums); break;
                    case 1: row_trees_kind<1>(f, t0, t1, buckets, bits, sums); break;
                    case 2: row_trees_kind<2>(f, t0, t1, buckets, bits, sums); break;
                    case 3: row_trees_kind<3>(f, t0, t1, buckets, bits, sums); break;
                    case 4: row_trees_kind<4>(f, t0, t1, buckets, bits, sums); break;
                    case 5: row_trees_kind<5>(f, t0, t1, buckets, bits, sums); break;
                    case 6: row_trees_kind<6>(f, t0, t1, buckets, bits, sums); break;
                    case 7: row_trees_kind<7>(f, t0, t1, buckets, bits, sums); break;
                    case 8: row_trees_kind<8>(f, t0, t1, buckets, bits, sums); break;
                    case 9: row_trees_kind<9>(f, t0, t1, buckets, bits, sums); break;
                    case 10: row_trees_kind<10>(f, t0, t1, buckets, bits, sums); break;
                    default: row_trees_any(f, t0, t1, buckets, sums); break;
                }
            }
            for (size_t d = 0; d < f.dim; d++) out[d] = f.scale * sums[d] + f.bias[d];
        }

        // One row without the block layout: a few rows would pay a loop over rows per split.
        void predict_row(const ObliviousForest& f, const double* x, double* out, Scratch& s) {
            const size_t F = f.floats.size(), H = f.one_hots.size();
            s.prepare(f, 1);
            uint16_t* buckets = s.buckets.data();
            uint32_t* hashes = s.hashes.data();
            uint32_t* entries = s.entries.data();
            for (size_t k = 0; k < F; k++) {
                const ObliviousForest::Float& ff = f.floats[k];
                const float v = static_cast<float>(x[ff.column]);
                buckets[k] = std::isnan(v) ? (ff.nan_as_true ? static_cast<uint16_t>(ff.borders.size()) : 0) : bucket_search(ff.search, v);
            }
            for (size_t c = 0; c < f.cat_columns.size(); c++) hashes[c] = static_cast<uint32_t>(x[f.cat_columns[c]]);
            for (size_t k = 0; k < H; k++) {
                const ObliviousForest::OneHot& oh = f.one_hots[k];
                const uint32_t h = hashes[oh.cat];
                uint16_t v = 0;
                for (size_t j = 0; j < oh.values.size(); j++) v = oh.values[j] == h ? static_cast<uint16_t>(j + 1) : v;
                buckets[F + k] = v;
            }
            uint64_t* prefix = s.prefixes.data();
            for (size_t n = 0; n < f.prefix_cat.size(); n++) {
                const uint32_t level = f.prefix_level[n];
                prefix[level] = combine_hash(level ? prefix[level - 1] : 0, static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(hashes[f.prefix_cat[n]]))));
                for (uint32_t i = f.prefix_projection_begin[n]; i < f.prefix_projection_begin[n + 1]; i++) {
                    const ObliviousForest::Projection& proj = f.projections[f.prefix_projections[i]];
                    uint64_t hash = prefix[level];
                    for (const auto& [source, threshold] : proj.floats) hash = combine_hash(hash, buckets[source] >= threshold ? 1 : 0);
                    for (const auto& [source, value] : proj.one_hots) hash = combine_hash(hash, buckets[source] == value ? 1 : 0);
                    for (uint32_t t : proj.tables) {
                        const ObliviousForest::Table& table = f.tables[t];
                        uint64_t slot = slot_of(hash, table.mask);
                        uint32_t found = 0;
                        while (table.slots[slot] != 0) {
                            if (table.keys[slot] == hash) { found = table.slots[slot]; break; }
                            slot = (slot + 1) & table.mask;
                        }
                        entries[t] = found;
                    }
                }
            }
            for (size_t k = 0; k < f.ctrs.size(); k++) {
                const ObliviousForest::Ctr& ctr = f.ctrs[k];
                const uint32_t e = entries[ctr.table];
                buckets[F + H + k] = e ? ctr.entry_buckets[e - 1] : ctr.missing_bucket;
            }

            // Condition bytes: per source a run of `bucket >= j + 1` (or `== j + 1` for one-hot),
            // written 8 bytes at a time (most CTR sources have a few conditions, where a loop per
            // source costs more than the work). A store may run up to 7 bytes into the next
            // source's run, which is written afterwards, or into the padding.
            if (!f.row_conditions) {
                row_trees_all(f, buckets, nullptr, out, s);
                return;
            }
            static_assert(std::endian::native == std::endian::little, "condition bytes are written as little-endian words");
            uint8_t* bits = s.conditions.data();
            constexpr uint64_t kOnes = 0x0101010101010101ULL;
            for (size_t k = 0; k + 1 < f.condition_begin.size(); k++) {
                const uint32_t v = buckets[k];
                uint8_t* b = bits + f.condition_begin[k];
                const uint32_t n = f.condition_begin[k + 1] - f.condition_begin[k];
                const bool one_hot = k >= F && k < F + H;
                for (uint32_t j = 0; j < n; j += 8, b += 8) {
                    uint64_t word;
                    if (one_hot) {  // byte v - 1 - j
                        word = v > j && v - j <= 8 ? uint64_t{1} << (8 * (v - 1 - j)) : 0;
                    } else {        // bytes below v - j
                        const uint32_t below = v > j ? std::min<uint32_t>(v - j, 8) : 0;
                        word = below == 8 ? kOnes : kOnes & ((uint64_t{1} << (8 * below)) - 1);
                    }
                    std::memcpy(b, &word, 8);  // little-endian byte order: byte i is bits 8i..8i+7
                }
            }
            row_trees_all(f, buckets, bits, out, s);
        }

        // Categorical columns must hold category hashes; checked before any thread starts.
        void check_categories(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features) {
            for (uint32_t column : f.cat_columns) {
                for (size_t i = 0; i < n_samples; i++) {
                    const double v = X[i * n_features + column];
                    if (!(v >= 0.0 && v <= 4294967295.0 && v == std::floor(v))) {
                        throw std::invalid_argument("Column " + std::to_string(column) +
                                                    " is categorical: it must hold category hashes (lignum.catboost_hash), got " + std::to_string(v) + ".");
                    }
                }
            }
        }

    } // namespace

    void predict_oblivious(const ObliviousForest& f, const double* X, size_t n_samples, size_t n_features, double* out,
                           int n_threads, Strategy strategy) {
        check_categories(f, X, n_samples, n_features);
        if (n_samples < kRowPathMax) {
            Scratch& s = thread_scratch();
            for (size_t i = 0; i < n_samples; i++) predict_row(f, X + i * n_features, out + i * f.dim, s);
            return;
        }
        const size_t n_blocks = (n_samples + kBlock - 1) / kBlock;
        const bool serial = strategy == Strategy::Serial || n_threads <= 1 || n_samples < 8;
        if (serial || n_blocks == 1) {
            Scratch& s = thread_scratch();
            for (size_t b = 0; b < n_blocks; b++) {
                const size_t i = b * kBlock;
                predict_block(f, X + i * n_features, n_features, std::min(kBlock, n_samples - i), out + i * f.dim, s);
            }
            return;
        }
        #pragma omp parallel num_threads(n_threads)
        {
            Scratch& s = thread_scratch();
            #pragma omp for schedule(dynamic)
            for (int64_t b = 0; b < static_cast<int64_t>(n_blocks); b++) {
                const size_t i = static_cast<size_t>(b) * kBlock;
                predict_block(f, X + i * n_features, n_features, std::min(kBlock, n_samples - i), out + i * f.dim, s);
            }
        }
    }

} // namespace lignum

namespace lignum {

    // ---------------------------------------------------------------------------------------------
    // lignum's binary format: the forest as arrays, each an 8-byte count and its elements, padded
    // to 64 bytes. Loading copies them into vectors and checks every index before any prediction.

    namespace {

        constexpr size_t pad64(size_t n) { return (n + 63) & ~size_t{63}; }

        class ArrayWriter {
            public:
                explicit ArrayWriter(std::ostream& out) : out_(out) {}
                template <typename T>
                void operator()(const std::vector<T>& v) {
                    const uint64_t count = v.size();
                    out_.write(reinterpret_cast<const char*>(&count), 8);
                    out_.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
                    const size_t used = 8 + v.size() * sizeof(T);
                    static const char zeros[64] = {};
                    out_.write(zeros, static_cast<std::streamsize>(pad64(used) - used));
                }
            private:
                std::ostream& out_;
        };

        class ArrayReader {
            public:
                ArrayReader(const char* data, size_t size, size_t offset) : data_(data), size_(size), offset_(offset) {}
                template <typename T>
                void operator()(std::vector<T>& v) {
                    uint64_t count = 0;
                    if (offset_ > size_ || size_ - offset_ < 8) bad();
                    std::memcpy(&count, data_ + offset_, 8);
                    if (count > (size_ - offset_ - 8) / sizeof(T)) bad();
                    v.resize(static_cast<size_t>(count));
                    if (count) std::memcpy(v.data(), data_ + offset_ + 8, static_cast<size_t>(count) * sizeof(T));
                    offset_ += pad64(8 + static_cast<size_t>(count) * sizeof(T));
                }
                [[noreturn]] static void bad() { throw std::runtime_error("Binary file is truncated or corrupted."); }
            private:
                const char* data_;
                size_t size_;
                size_t offset_;
        };

        // Visits every array of the forest in file order (the same code writes and reads).
        template <typename Forest, typename Visit>
        void visit_arrays(Forest& f, Visit&& visit, std::vector<uint32_t>& u32, std::vector<double>& f64) {
            visit(u32);  // n_columns, dim
            visit(f64);  // scale
            visit(f.bias);
            std::vector<uint32_t> float_column, border_begin;
            std::vector<uint8_t> nan_as_true;
            std::vector<float> borders;
            std::vector<uint32_t> one_hot_cat, value_begin, values;
            std::vector<uint32_t> proj_cat_begin, proj_cats, proj_float_begin, proj_float_source, proj_one_hot_begin, proj_one_hot_source,
                                  proj_table_begin, proj_tables;
            std::vector<uint16_t> proj_float_threshold, proj_one_hot_value;
            std::vector<uint32_t> table_projection, ctr_table;
            std::vector<uint64_t> table_mask, slot_begin, keys, entry_begin;
            std::vector<uint32_t> slots;
            std::vector<uint16_t> ctr_missing, entry_buckets;
            constexpr bool writing = std::is_const_v<Forest>;
            if constexpr (writing) {
                border_begin.push_back(0);
                for (const auto& x : f.floats) {
                    float_column.push_back(x.column);
                    nan_as_true.push_back(x.nan_as_true);
                    borders.insert(borders.end(), x.borders.begin(), x.borders.end());
                    border_begin.push_back(static_cast<uint32_t>(borders.size()));
                }
                value_begin.push_back(0);
                for (const auto& x : f.one_hots) {
                    one_hot_cat.push_back(x.cat);
                    values.insert(values.end(), x.values.begin(), x.values.end());
                    value_begin.push_back(static_cast<uint32_t>(values.size()));
                }
                proj_cat_begin.push_back(0); proj_float_begin.push_back(0); proj_one_hot_begin.push_back(0); proj_table_begin.push_back(0);
                for (const auto& p : f.projections) {
                    proj_cats.insert(proj_cats.end(), p.cats.begin(), p.cats.end());
                    for (const auto& [s, t] : p.floats) { proj_float_source.push_back(s); proj_float_threshold.push_back(t); }
                    for (const auto& [s, v] : p.one_hots) { proj_one_hot_source.push_back(s); proj_one_hot_value.push_back(v); }
                    proj_tables.insert(proj_tables.end(), p.tables.begin(), p.tables.end());
                    proj_cat_begin.push_back(static_cast<uint32_t>(proj_cats.size()));
                    proj_float_begin.push_back(static_cast<uint32_t>(proj_float_source.size()));
                    proj_one_hot_begin.push_back(static_cast<uint32_t>(proj_one_hot_source.size()));
                    proj_table_begin.push_back(static_cast<uint32_t>(proj_tables.size()));
                }
                slot_begin.push_back(0);
                for (const auto& t : f.tables) {
                    table_projection.push_back(t.projection);
                    table_mask.push_back(t.mask);
                    keys.insert(keys.end(), t.keys.begin(), t.keys.end());
                    slots.insert(slots.end(), t.slots.begin(), t.slots.end());
                    slot_begin.push_back(keys.size());
                }
                entry_begin.push_back(0);
                for (const auto& c : f.ctrs) {
                    ctr_table.push_back(c.table);
                    ctr_missing.push_back(c.missing_bucket);
                    entry_buckets.insert(entry_buckets.end(), c.entry_buckets.begin(), c.entry_buckets.end());
                    entry_begin.push_back(entry_buckets.size());
                }
            }
            visit(float_column); visit(nan_as_true); visit(border_begin); visit(borders);
            visit(f.cat_columns);
            visit(one_hot_cat); visit(value_begin); visit(values);
            visit(proj_cat_begin); visit(proj_cats); visit(proj_float_begin); visit(proj_float_source); visit(proj_float_threshold);
            visit(proj_one_hot_begin); visit(proj_one_hot_source); visit(proj_one_hot_value); visit(proj_table_begin); visit(proj_tables);
            visit(table_projection); visit(table_mask); visit(slot_begin); visit(keys); visit(slots);
            visit(ctr_table); visit(ctr_missing); visit(entry_begin); visit(entry_buckets);
            visit(f.split_begin); visit(f.split_source); visit(f.split_threshold); visit(f.split_equal); visit(f.leaf_begin); visit(f.leaves);
            if constexpr (!writing) {
                auto range = [](const auto& begin, size_t n, size_t total) {
                    if (begin.size() != n + 1 || begin[0] != 0 || begin[n] != total) ArrayReader::bad();
                    for (size_t i = 0; i < n; i++) if (begin[i] > begin[i + 1]) ArrayReader::bad();
                };
                const size_t F = float_column.size();
                if (nan_as_true.size() != F) ArrayReader::bad();
                range(border_begin, F, borders.size());
                f.floats.resize(F);
                for (size_t i = 0; i < F; i++) {
                    f.floats[i].column = float_column[i];
                    f.floats[i].nan_as_true = nan_as_true[i] != 0;
                    f.floats[i].borders.assign(borders.begin() + border_begin[i], borders.begin() + border_begin[i + 1]);
                }
                const size_t H = one_hot_cat.size();
                range(value_begin, H, values.size());
                f.one_hots.resize(H);
                for (size_t i = 0; i < H; i++) {
                    f.one_hots[i].cat = one_hot_cat[i];
                    f.one_hots[i].values.assign(values.begin() + value_begin[i], values.begin() + value_begin[i + 1]);
                }
                const size_t P = proj_cat_begin.empty() ? 0 : proj_cat_begin.size() - 1;
                range(proj_cat_begin, P, proj_cats.size());
                range(proj_float_begin, P, proj_float_source.size());
                range(proj_one_hot_begin, P, proj_one_hot_source.size());
                range(proj_table_begin, P, proj_tables.size());
                if (proj_float_threshold.size() != proj_float_source.size() || proj_one_hot_value.size() != proj_one_hot_source.size()) ArrayReader::bad();
                f.projections.resize(P);
                for (size_t p = 0; p < P; p++) {
                    auto& proj = f.projections[p];
                    proj.cats.assign(proj_cats.begin() + proj_cat_begin[p], proj_cats.begin() + proj_cat_begin[p + 1]);
                    for (uint32_t k = proj_float_begin[p]; k < proj_float_begin[p + 1]; k++) proj.floats.emplace_back(proj_float_source[k], proj_float_threshold[k]);
                    for (uint32_t k = proj_one_hot_begin[p]; k < proj_one_hot_begin[p + 1]; k++) proj.one_hots.emplace_back(proj_one_hot_source[k], proj_one_hot_value[k]);
                    proj.tables.assign(proj_tables.begin() + proj_table_begin[p], proj_tables.begin() + proj_table_begin[p + 1]);
                }
                const size_t T = table_projection.size();
                if (table_mask.size() != T || slots.size() != keys.size()) ArrayReader::bad();
                range(slot_begin, T, keys.size());
                f.tables.resize(T);
                for (size_t t = 0; t < T; t++) {
                    f.tables[t].projection = table_projection[t];
                    f.tables[t].mask = table_mask[t];
                    f.tables[t].keys.assign(keys.begin() + static_cast<ptrdiff_t>(slot_begin[t]), keys.begin() + static_cast<ptrdiff_t>(slot_begin[t + 1]));
                    f.tables[t].slots.assign(slots.begin() + static_cast<ptrdiff_t>(slot_begin[t]), slots.begin() + static_cast<ptrdiff_t>(slot_begin[t + 1]));
                }
                const size_t C = ctr_table.size();
                if (ctr_missing.size() != C) ArrayReader::bad();
                range(entry_begin, C, entry_buckets.size());
                f.ctrs.resize(C);
                for (size_t c = 0; c < C; c++) {
                    f.ctrs[c].table = ctr_table[c];
                    f.ctrs[c].missing_bucket = ctr_missing[c];
                    f.ctrs[c].entry_buckets.assign(entry_buckets.begin() + static_cast<ptrdiff_t>(entry_begin[c]),
                                                   entry_buckets.begin() + static_cast<ptrdiff_t>(entry_begin[c + 1]));
                }
            }
        }

        // Every index the kernel follows, so a damaged file is rejected instead of read out of bounds.
        void validate(const ObliviousForest& f) {
            auto bad = [] { ArrayReader::bad(); };
            if (f.dim == 0 || f.bias.size() != f.dim) bad();
            const size_t F = f.floats.size(), C = f.cat_columns.size(), H = f.one_hots.size();
            if (f.n_sources() > 65535) bad();
            for (const auto& x : f.floats) {
                if (x.column >= f.n_columns || x.borders.empty() || x.borders.size() > 65534) bad();
                for (size_t i = 1; i < x.borders.size(); i++) if (!(x.borders[i - 1] < x.borders[i])) bad();
            }
            for (uint32_t c : f.cat_columns) if (c >= f.n_columns) bad();
            for (const auto& x : f.one_hots) if (x.cat >= C || x.values.size() > 65534) bad();
            for (size_t p = 0; p < f.projections.size(); p++) {
                const auto& proj = f.projections[p];
                for (uint32_t c : proj.cats) if (c >= C) bad();
                for (const auto& [s, t] : proj.floats) if (s >= F || t > f.floats[s].borders.size()) bad();
                for (const auto& [s, v] : proj.one_hots) if (s < F || s >= F + H || v > f.one_hots[s - F].values.size()) bad();
                for (uint32_t t : proj.tables) if (t >= f.tables.size() || f.tables[t].projection != p) bad();
            }
            std::vector<size_t> entries(f.tables.size(), 0);
            for (size_t t = 0; t < f.tables.size(); t++) {
                const auto& table = f.tables[t];
                if (table.projection >= f.projections.size() || table.mask == ~uint64_t{0} || ((table.mask + 1) & table.mask) != 0 ||
                    table.keys.size() != table.mask + 1 || table.slots.size() != table.keys.size()) bad();
                const auto& owners = f.projections[table.projection].tables;
                if (std::find(owners.begin(), owners.end(), t) == owners.end()) bad();
                size_t empty = 0;
                for (uint32_t s : table.slots) { entries[t] = std::max<size_t>(entries[t], s); empty += s == 0; }
                if (empty == 0) bad();  // the lookup stops at an empty slot
            }
            for (const auto& c : f.ctrs) if (c.table >= f.tables.size() || c.entry_buckets.size() < entries[c.table]) bad();
            const size_t N = f.leaf_begin.size();
            if (f.split_begin.size() != N + 1 || f.split_begin[0] != 0 || f.split_begin[N] != f.split_source.size() ||
                f.split_threshold.size() != f.split_source.size() || f.split_equal.size() != f.split_source.size()) bad();
            for (size_t t = 0; t < N; t++) {
                if (f.split_begin[t] > f.split_begin[t + 1]) bad();
                const size_t depth = f.split_begin[t + 1] - f.split_begin[t];
                if (depth > 16 || f.leaf_begin[t] > f.leaves.size() || ((size_t{1} << depth) * f.dim) > f.leaves.size() - f.leaf_begin[t]) bad();
            }
            for (uint16_t s : f.split_source) if (s >= f.n_sources()) bad();
            for (uint8_t e : f.split_equal) if (e > 1) bad();
        }

    } // namespace

    void save_oblivious(std::ostream& out, const ObliviousForest& f) {
        std::vector<uint32_t> u32 = {f.n_columns, f.dim};
        std::vector<double> f64 = {f.scale};
        ArrayWriter writer(out);
        visit_arrays(f, writer, u32, f64);
    }

    std::shared_ptr<ObliviousForest> load_oblivious(const char* data, size_t size, size_t offset) {
        auto f = std::make_shared<ObliviousForest>();
        std::vector<uint32_t> u32;
        std::vector<double> f64;
        ArrayReader reader(data, size, offset);
        visit_arrays(*f, reader, u32, f64);
        if (u32.size() != 2 || f64.size() != 1) ArrayReader::bad();
        f->n_columns = u32[0];
        f->dim = u32[1];
        f->scale = f64[0];
        validate(*f);
        derive_layout(*f);
        return f;
    }

} // namespace lignum
