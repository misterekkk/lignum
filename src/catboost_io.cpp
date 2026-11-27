#include "catboost.hpp"
#include "catboost_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
                    ctr_missing.push_back(c.entry_buckets[0]);  // the file keeps it apart from the entries
                    entry_buckets.insert(entry_buckets.end(), c.entry_buckets.begin() + 1, c.entry_buckets.end());
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
                    std::vector<uint16_t>& b = f.ctrs[c].entry_buckets;
                    b.assign(1, ctr_missing[c]);
                    b.insert(b.end(), entry_buckets.begin() + static_cast<ptrdiff_t>(entry_begin[c]),
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
                if (x.column >= f.n_columns || x.borders.empty() || x.borders.size() > 65534 || std::isnan(x.borders[0])) bad();
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
            for (const auto& c : f.ctrs) if (c.table >= f.tables.size() || c.entry_buckets.size() < entries[c.table] + 1) bad();
            const size_t N = f.leaf_begin.size();
            if (f.split_begin.size() != N + 1 || f.split_begin[0] != 0 || f.split_begin[N] != f.split_source.size() ||
                f.split_threshold.size() != f.split_source.size() || f.split_equal.size() != f.split_source.size()) bad();
            // The leaves of the trees follow each other, as the parser writes them: the kernel for
            // trees of one depth steps from tree to tree by their size instead of reading leaf_begin
            // (a gap or an overlap read past the leaves; found by tests/fuzz).
            size_t leaf = 0;
            for (size_t t = 0; t < N; t++) {
                if (f.split_begin[t] > f.split_begin[t + 1]) bad();
                const size_t depth = f.split_begin[t + 1] - f.split_begin[t];
                if (depth > 16 || f.leaf_begin[t] != leaf || ((size_t{1} << depth) * f.dim) > f.leaves.size() - leaf) bad();
                leaf += (size_t{1} << depth) * f.dim;
            }
            if (leaf != f.leaves.size()) bad();
            for (uint16_t s : f.split_source) if (s >= f.n_sources()) bad();
            for (uint8_t e : f.split_equal) if (e > 1) bad();
            std::vector<uint16_t> max_t(f.n_sources(), 0);
            for (size_t k = 0; k < f.split_source.size(); k++) {
                const uint16_t s = f.split_source[k], t = f.split_threshold[k];
                const uint8_t eq = f.split_equal[k];
                if (t == 0) bad();
                if (s < F) {
                    if (eq != 0 || t > f.floats[s].borders.size()) bad();
                } else if (s < F + H) {
                    if (eq != 1 || t > f.one_hots[s - F].values.size()) bad();
                } else {
                    if (eq != 0 || t > 65534) bad();
                }
                if (t > max_t[s]) max_t[s] = t;
            }
            uint64_t cond_sum = 0;
            for (uint16_t mt : max_t) cond_sum += mt;
            if (cond_sum > (uint64_t{64} << 20)) bad();
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
        catboost_detail::derive_layout(*f);
        return f;
    }

} // namespace lignum
