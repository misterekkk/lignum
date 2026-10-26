// The CatBoost block kernel, included once per instruction set with LIGNUM_CATBOOST_ISA naming its
// namespace (generic in catboost.cpp; avx2 in catboost_kernel_avx2.cpp, compiled for AVX2). All
// of it is internal to that namespace, so no function is shared between the builds.
// Needs catboost_kernel.hpp, <algorithm>, <cmath>, <cstring>, <limits> and <vector>.

#ifndef LIGNUM_CATBOOST_ISA
    #error "define LIGNUM_CATBOOST_ISA before including catboost_kernel.inc"
#endif

namespace lignum::catboost_kernel::LIGNUM_CATBOOST_ISA {

    namespace {

        // CatBoost's CalcHash: combines a projection's hash with one more value.
        inline uint64_t combine_hash(uint64_t a, uint64_t b) {
            constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
            return kMagic * (a + kMagic * b);
        }

        // Slot of a combination hash in a table; the hashes are already well mixed.
        inline uint64_t slot_of(uint64_t key, uint64_t mask) { return (key ^ (key >> 29)) & mask; }

        // The way (0, 1 or 2 above) a feature's buckets are computed.
        inline int float_mode(const ObliviousForest::Float& ff) {
            int mode = kFloatMode == 3 ? (ff.borders.size() <= kSmallFeature ? kSmallFeatureMode : 2) : kFloatMode;
            return mode == 1 && ff.borders.size() > kCountBorders ? 0 : mode;
        }

        // bucket_of over a feature's padded borders: a fixed number of halvings (log2 of the padded
        // size). Written as a conditional move, which ARM compilers keep (csel) and x86 ones turn
        // into a data-dependent branch, so only for features with many borders. x is not NaN; +inf
        // padding is never below x, so the count is the same.
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

        // The borders below x in two levels: c coarse borders (the last of each block of 16) are
        // below x, so are the 16 c borders of those blocks, plus the ones of block c below x.
        // Both are sums of comparisons, without selects. x is not NaN.
        inline uint16_t bucket_two_level(const ObliviousForest::Float& ff, float x) {
            uint32_t blocks = 0;
            for (const float border : ff.coarse) blocks += x > border;
            const float* fine = ff.fine.data() + 16 * static_cast<size_t>(blocks);
            uint32_t below = 0;
            for (int j = 0; j < 16; j++) below += x > fine[j];
            return static_cast<uint16_t>(16 * blocks + below);
        }

        // The bucket of a float feature's value that is not NaN, as float_mode chooses.
        inline uint16_t float_bucket(const ObliviousForest::Float& ff, float x) {
            switch (float_mode(ff)) {
                case 2: return bucket_two_level(ff, x);
                case 1: {
                    uint32_t below = 0;
                    for (const float border : ff.borders) below += x > border;
                    return static_cast<uint16_t>(below);
                }
                default: return bucket_search(ff.search, x);
            }
        }

        // Number of strictly increasing borders below x: the bucket with x > borders[k] <=> bucket >= k + 1.
        inline uint16_t bucket_of(const std::vector<float>& borders, float x) {
            return static_cast<uint16_t>(std::lower_bound(borders.begin(), borders.end(), x) - borders.begin());
        }

        // Leaf indices (16 bits: depth <= 16) of one tree for a block of rows, from buckets of type B
        // (uint16_t, or uint8_t when every source fits). With the depth known at compile time, all
        // levels are done in one pass over the rows with the index in a register. Both split kinds
        // are one unsigned range test, (B)(bucket - t) <= range: range max - t for `bucket >= t`,
        // 0 for the one-hot `bucket == t`. (Two comparisons joined with `&` stopped the loop from
        // vectorising: 3x slower.)
        template <int Depth, typename B, typename I>
        LIGNUM_HOT_LOOP void tree_index(const B* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                                        const uint8_t* equal, size_t rows, I* index) {
            static_assert(Depth <= 8 * static_cast<int>(sizeof(I)), "leaf index too narrow for the depth");
            const B* b[Depth];
            B lo[Depth], range[Depth];
            for (int d = 0; d < Depth; d++) {
                b[d] = buckets + source[d] * stride;
                lo[d] = static_cast<B>(threshold[d]);
                range[d] = equal[d] ? B{0} : static_cast<B>(static_cast<B>(~B{0}) - threshold[d]);
            }
            for (size_t r = 0; r < rows; r++) {
                I v = 0;
                for (int d = 0; d < Depth; d++) v |= static_cast<I>((static_cast<B>(b[d][r] - lo[d]) <= range[d]) << d);
                index[r] = v;
            }
        }

        // Any depth, one pass over the rows per level.
        template <typename B, typename I>
        void tree_index_any(const B* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                            const uint8_t* equal, uint32_t depth, size_t rows, I* index) {
            std::fill(index, index + rows, I{0});
            for (uint32_t d = 0; d < depth; d++) {
                const B* b = buckets + source[d] * stride;
                const uint16_t t = threshold[d];
                if (equal[d]) {
                    for (size_t r = 0; r < rows; r++) index[r] |= static_cast<I>((b[r] == t) << d);
                } else {
                    for (size_t r = 0; r < rows; r++) index[r] |= static_cast<I>((b[r] >= t) << d);
                }
            }
        }

        // Leaf indices of a tree whose splits are all `bucket >= t` (see kPlainSplits).
        template <int Depth, typename B, typename I>
        LIGNUM_HOT_LOOP void tree_index_ge(const B* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                                           size_t rows, I* index) {
            static_assert(Depth <= 8 * static_cast<int>(sizeof(I)), "leaf index too narrow for the depth");
            const B* b[Depth];
            B lo[Depth];
            for (int d = 0; d < Depth; d++) {
                b[d] = buckets + source[d] * stride;
                lo[d] = static_cast<B>(threshold[d]);
            }
            for (size_t r = 0; r < rows; r++) {
                I v = 0;
                for (int d = Depth - 1; d >= 0; d--) v = static_cast<I>(v + v + (b[d][r] >= lo[d]));
                index[r] = v;
            }
        }

        template <typename B, typename I>
        void tree_index_ge_dispatch(const B* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                                    const uint8_t* equal, uint32_t depth, size_t rows, I* index) {
            switch (depth) {
                case 1: tree_index_ge<1>(buckets, stride, source, threshold, rows, index); break;
                case 2: tree_index_ge<2>(buckets, stride, source, threshold, rows, index); break;
                case 3: tree_index_ge<3>(buckets, stride, source, threshold, rows, index); break;
                case 4: tree_index_ge<4>(buckets, stride, source, threshold, rows, index); break;
                case 5: tree_index_ge<5>(buckets, stride, source, threshold, rows, index); break;
                case 6: tree_index_ge<6>(buckets, stride, source, threshold, rows, index); break;
                case 7: tree_index_ge<7>(buckets, stride, source, threshold, rows, index); break;
                case 8: tree_index_ge<8>(buckets, stride, source, threshold, rows, index); break;
                default:
                    if constexpr (sizeof(I) > 1) {
                        if (depth == 9) tree_index_ge<9>(buckets, stride, source, threshold, rows, index);
                        else if (depth == 10) tree_index_ge<10>(buckets, stride, source, threshold, rows, index);
                        else tree_index_any(buckets, stride, source, threshold, equal, depth, rows, index);
                    } else {
                        tree_index_any(buckets, stride, source, threshold, equal, depth, rows, index);  // depth 0
                    }
                    break;
            }
        }

        // Byte indices (I = uint8_t) are only used when every tree has depth <= 8.
        template <typename B, typename I>
        void tree_index_dispatch(const B* buckets, size_t stride, const uint16_t* source, const uint16_t* threshold,
                                 const uint8_t* equal, uint32_t depth, size_t rows, I* index) {
            switch (depth) {
                case 0: std::fill(index, index + rows, I{0}); break;
                case 1: tree_index<1>(buckets, stride, source, threshold, equal, rows, index); break;
                case 2: tree_index<2>(buckets, stride, source, threshold, equal, rows, index); break;
                case 3: tree_index<3>(buckets, stride, source, threshold, equal, rows, index); break;
                case 4: tree_index<4>(buckets, stride, source, threshold, equal, rows, index); break;
                case 5: tree_index<5>(buckets, stride, source, threshold, equal, rows, index); break;
                case 6: tree_index<6>(buckets, stride, source, threshold, equal, rows, index); break;
                case 7: tree_index<7>(buckets, stride, source, threshold, equal, rows, index); break;
                case 8: tree_index<8>(buckets, stride, source, threshold, equal, rows, index); break;
                default:
                    if constexpr (sizeof(I) > 1) {
                        if (depth == 9) tree_index<9>(buckets, stride, source, threshold, equal, rows, index);
                        else if (depth == 10) tree_index<10>(buckets, stride, source, threshold, equal, rows, index);
                        else tree_index_any(buckets, stride, source, threshold, equal, depth, rows, index);
                    }
                    break;
            }
        }

        // Adds the leaves of G trees to each row's sums, tree after tree (index: G arrays of rows).
        template <int G, int Dim, typename I>
        LIGNUM_HOT_LOOP inline void add_leaves(const double* const* leaves, const I* index, size_t rows, size_t dim, double* sums) {
            if constexpr (Dim > 0) {
                for (size_t r = 0; r < rows; r++) {
                    double acc[Dim];
                    for (int d = 0; d < Dim; d++) acc[d] = sums[r * Dim + d];
                    for (int g = 0; g < G; g++) {
                        const double* l = leaves[g] + index[g * rows + r] * Dim;
                        for (int d = 0; d < Dim; d++) acc[d] += l[d];
                    }
                    for (int d = 0; d < Dim; d++) sums[r * Dim + d] = acc[d];
                }
            } else {
                for (size_t r = 0; r < rows; r++) {
                    for (int g = 0; g < G; g++) {
                        const double* l = leaves[g] + index[g * rows + r] * dim;
                        for (size_t d = 0; d < dim; d++) sums[r * dim + d] += l[d];
                    }
                }
            }
        }

        // Phase 2 for a block: leaf index per tree, leaves added in model order (one double addition
        // per tree and output), kGroup trees at a time. Dim outputs per row, 0: any, from the model.
        // The sums are a local array: on the heap the compiler cannot keep them apart from the
        // leaves (7 classes: 0.149 instead of 0.100 s for 200k rows).
        template <int Dim, typename B, typename I>
        void phase2(const ObliviousForest& f, const B* buckets, size_t stride, size_t rows, Scratch& s, double* out) {
            double local[Dim > 0 ? kBlock * Dim : 1];
            const size_t dim = Dim > 0 ? Dim : f.dim;
            double* sums = Dim > 0 ? local : s.sums.data();
            std::fill(sums, sums + rows * dim, 0.0);
            I* index = reinterpret_cast<I*>(s.index.data());  // 16-bit storage; bytes may alias it
            const double* leaves[kGroup];
            auto tree = [&](size_t t, int g) {
                const uint32_t k = f.split_begin[t], depth = f.split_begin[t + 1] - k;
                leaves[g] = f.leaves.data() + f.leaf_begin[t];
                if (kPlainSplits && !f.tree_equal[t]) {
                    tree_index_ge_dispatch(buckets, stride, f.split_source.data() + k, f.split_threshold.data() + k,
                                           f.split_equal.data() + k, depth, rows, index + g * rows);
                } else {
                    tree_index_dispatch(buckets, stride, f.split_source.data() + k, f.split_threshold.data() + k,
                                        f.split_equal.data() + k, depth, rows, index + g * rows);
                }
            };
            size_t t = 0;
            for (; t + kGroup <= f.n_trees(); t += kGroup) {
                for (int g = 0; g < kGroup; g++) tree(t + g, g);
                add_leaves<kGroup, Dim, I>(leaves, index, rows, dim, sums);
            }
            for (; t < f.n_trees(); t++) {
                tree(t, 0);
                add_leaves<1, Dim, I>(leaves, index, rows, dim, sums);
            }
            for (size_t r = 0; r < rows; r++) {
                for (size_t d = 0; d < dim; d++) out[r * dim + d] = f.scale * sums[r * dim + d] + f.bias[d];
            }
        }

        template <typename B, typename I>
        void phase2_dim(const ObliviousForest& f, const B* buckets, size_t stride, size_t rows, Scratch& s, double* out) {
            switch (f.dim) {
                case 1: phase2<1, B, I>(f, buckets, stride, rows, s, out); break;
                case 2: phase2<2, B, I>(f, buckets, stride, rows, s, out); break;
                case 3: phase2<3, B, I>(f, buckets, stride, rows, s, out); break;
                case 4: phase2<4, B, I>(f, buckets, stride, rows, s, out); break;
                case 5: phase2<5, B, I>(f, buckets, stride, rows, s, out); break;
                case 6: phase2<6, B, I>(f, buckets, stride, rows, s, out); break;
                case 7: phase2<7, B, I>(f, buckets, stride, rows, s, out); break;
                case 8: phase2<8, B, I>(f, buckets, stride, rows, s, out); break;
                case 9: phase2<9, B, I>(f, buckets, stride, rows, s, out); break;
                case 10: phase2<10, B, I>(f, buckets, stride, rows, s, out); break;
                default: phase2<0, B, I>(f, buckets, stride, rows, s, out); break;
            }
        }

        // Buckets of one float feature for a block of rows (float_mode). Counting compares without
        // selects, so NaN (below every border) needs no test; AsTrue puts it above afterwards.
        template <typename B>
        LIGNUM_HOT_LOOP void float_buckets(const ObliviousForest::Float& ff, const double* X, size_t n_features, size_t rows, B* b, Scratch& s) {
            const B nan_bucket = ff.nan_as_true ? static_cast<B>(ff.borders.size()) : B{0};
            const int mode = float_mode(ff);
            const bool count = mode == 1;
            if (mode == 0) {
                for (size_t r = 0; r < rows; r++) {
                    const float x = static_cast<float>(X[r * n_features + ff.column]);
                    b[r] = std::isnan(x) ? nan_bucket : static_cast<B>(bucket_search(ff.search, x));
                }
                return;
            }
            float* x = s.values.data();
            for (size_t r = 0; r < rows; r++) x[r] = static_cast<float>(X[r * n_features + ff.column]);
            if (count) {  // border by border over the rows: compares and adds the compiler vectorises
                std::fill(b, b + rows, B{0});
                for (const float border : ff.borders) {
                    for (size_t r = 0; r < rows; r++) b[r] = static_cast<B>(b[r] + (x[r] > border));
                }
            } else {      // two levels: coarse borders over the rows, then each row's block of 16
                uint32_t* blocks = s.counts.data();
                std::fill(blocks, blocks + rows, 0u);
                for (const float border : ff.coarse) {
                    for (size_t r = 0; r < rows; r++) blocks[r] += x[r] > border;
                }
                for (size_t r = 0; r < rows; r++) {
                    const float* fine = ff.fine.data() + 16 * static_cast<size_t>(blocks[r]);
                    uint32_t below = 0;
                    for (int j = 0; j < 16; j++) below += x[r] > fine[j];
                    b[r] = static_cast<B>(16 * blocks[r] + below);
                }
            }
            if (ff.nan_as_true) {
                for (size_t r = 0; r < rows; r++) b[r] = std::isnan(x[r]) ? nan_bucket : b[r];
            }
        }

        template <typename B>
        void predict_block_typed(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s, B* buckets) {
            const size_t F = f.floats.size(), H = f.one_hots.size();
            const size_t stride = rows;

            // Phase 1: buckets of every source.
            for (size_t k = 0; k < F; k++) float_buckets(f.floats[k], X, n_features, rows, buckets + k * stride, s);
            for (size_t c = 0; c < f.cat_columns.size(); c++) {
                uint32_t* h = s.hashes.data() + c * stride;
                const uint32_t column = f.cat_columns[c];
                for (size_t r = 0; r < rows; r++) h[r] = static_cast<uint32_t>(X[r * n_features + column]);
            }
            for (size_t k = 0; k < H; k++) {
                const ObliviousForest::OneHot& oh = f.one_hots[k];
                const uint32_t* h = s.hashes.data() + oh.cat * stride;
                B* b = buckets + (F + k) * stride;
                for (size_t r = 0; r < rows; r++) {
                    B v = 0;
                    for (size_t j = 0; j < oh.values.size(); j++) v = oh.values[j] == h[r] ? static_cast<B>(j + 1) : v;
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
                            const B* b = buckets + source * stride;
                            for (size_t r = 0; r < rows; r++) c[r] = combine_hash(c[r], b[r] >= threshold ? 1 : 0);
                        }
                        for (const auto& [source, value] : proj.one_hots) {
                            const B* b = buckets + source * stride;
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
                B* b = buckets + (F + H + k) * stride;
                for (size_t r = 0; r < rows; r++) b[r] = static_cast<B>(entries[r] ? ctr.entry_buckets[entries[r] - 1] : ctr.missing_bucket);
            }

            if (kByteIndex && f.max_depth <= 8) phase2_dim<B, uint8_t>(f, buckets, stride, rows, s, out);
            else phase2_dim<B, uint16_t>(f, buckets, stride, rows, s, out);
        }

    } // namespace

    void predict_block(const ObliviousForest& f, const double* X, size_t n_features, size_t rows, double* out, Scratch& s) {
        if (kSmallBuckets && f.small_buckets) {
            predict_block_typed<uint8_t>(f, X, n_features, rows, out, s, s.small.data());
        } else {
            predict_block_typed<uint16_t>(f, X, n_features, rows, out, s, s.buckets.data());
        }
    }


} // namespace lignum::catboost_kernel::LIGNUM_CATBOOST_ISA
