#include "catboost.hpp"
#include "catboost_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility>

namespace lignum {

    // ---------------------------------------------------------------------------------------------
    // CityHash64 as in CatBoost's util/digest/city.cpp (Google's first release, MIT licence):
    // categories are hashed with CityHash64(utf8) & 0xffffffff. Later CityHash versions give
    // different values (for over 64 bytes even 1.0.3 does).
    // Derived from CityHash, Copyright (c) 2011 Google, Inc., by Geoff Pike and Jyrki Alakuijala,
    // under the MIT License; the full notice is in NOTICE.

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

    uint32_t catboost_hash(std::string_view category) {
        return static_cast<uint32_t>(city_hash64(category.data(), category.size()));
    }

    // ---------------------------------------------------------------------------------------------
    // Derived from fmath (https://github.com/herumi/fmath) by herumi, under the BSD 3-Clause
    // License; the full notice is in NOTICE.
    // CatBoost's exp (NCB::FastExpWithInfInplace). On x86 it is herumi's fmath::expd_v, whose SSE2
    // and AVX2 loops do the same double operations as this scalar copy (no FMA): a 2^11-entry table
    // of 2^(i/2048), the index and exponent taken from x * 2048/ln2 + 3 * 2^51, and a cubic
    // correction.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    namespace {

        struct FmathExpTable {
            static constexpr int kBits = 11;
            static constexpr uint64_t kSize = uint64_t{1} << kBits;
            static constexpr uint64_t kAdj = (uint64_t{1} << (kBits + 10)) - (uint64_t{1} << kBits);
            uint64_t table[kSize];
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

        double fmath_expd(double x, const FmathExpTable& c, bool vector) {
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

        // fmath's loop is 4 wide with AVX2, otherwise 2 (SSE2).
        size_t fmath_width() {
            static const size_t width = catboost_detail::cpu_has_avx2() ? size_t{4} : size_t{2};
            return width;
        }

    } // namespace
#endif

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

} // namespace lignum
