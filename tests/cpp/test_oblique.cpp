// In-memory building (lignum/build.hpp) and miter v1.1 integration:
// oblique splits, quantile Q16 quantizers, native CTR tables, multiclass + softmax,
// numerically stable sigmoid, symmetric oblivious miter trees, format 0.10, and compiled mode.
#include <lignum/build.hpp>
#include <lignum/lignum.hpp>

#include "binned.hpp"
#include "columns.hpp"
#include "compiled.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

lignum::ir::Node leaf(double v) {
    lignum::ir::Node n;
    n.kind = lignum::ir::Node::Kind::Leaf;
    n.value = v;
    return n;
}

lignum::ir::Node oblique(uint8_t k, uint32_t f0, uint32_t f1, uint32_t f2, int16_t w0, int16_t w1, int16_t w2,
                         int32_t thr, bool default_right, uint32_t left, uint32_t right) {
    lignum::ir::Node n;
    n.kind = lignum::ir::Node::Kind::Oblique;
    n.oblique.k = k;
    n.oblique.features[0] = f0;
    n.oblique.features[1] = f1;
    n.oblique.features[2] = f2;
    n.oblique.weights[0] = w0;
    n.oblique.weights[1] = w1;
    n.oblique.weights[2] = w2;
    n.oblique.threshold = thr;
    n.default_right = default_right;
    n.left = left;
    n.right = right;
    return n;
}

lignum::ir::Node numeric(uint32_t f, double thr, bool default_right, uint32_t left, uint32_t right) {
    lignum::ir::Node n;
    n.kind = lignum::ir::Node::Kind::Numeric;
    n.feature = f;
    n.value = thr;
    n.default_right = default_right;
    n.left = left;
    n.right = right;
    return n;
}

lignum::ir::Node categorical(uint32_t f, std::vector<uint32_t> cats, uint32_t left, uint32_t right) {
    lignum::ir::Node n;
    n.kind = lignum::ir::Node::Kind::Categorical;
    n.feature = f;
    n.categories = std::move(cats);
    n.default_right = true;
    n.left = left;
    n.right = right;
    return n;
}

lignum::ir::Node ctr_split(uint32_t table_idx, float thr, uint32_t left, uint32_t right) {
    lignum::ir::Node n;
    n.kind = lignum::ir::Node::Kind::Ctr;
    n.feature = table_idx;
    n.value = static_cast<double>(thr);
    n.left = left;
    n.right = right;
    return n;
}

// Three columns; quantizer lo = 0, scale = 1, so code(x) = floor(x + 0.5) clamped to [0, 65534]
// and the centred code is c = code - 32768.
lignum::ir::Forest base_forest() {
    lignum::ir::Forest f;
    f.n_features = 3;
    f.base_score = 0.5;
    f.quantizers.assign(3, lignum::FeatureQuantizer{0.0, 1.0});
    f.quantizer_constant.assign(3, 0);
    return f;
}

double predict_one(const lignum::Model& m, std::vector<double> row) {
    double out = 0.0;
    m.predict_raw(row.data(), 1, row.size(), &out, 1);
    return out;
}

// Reference CTR table builder matching miter's open-addressing layout.
uint64_t combine_cat_hash_ref(uint64_t a, uint64_t b) {
    constexpr uint64_t kMagic = 0x4906ba494954cb65ULL;
    return kMagic * (a + kMagic * b);
}

lignum::ir::CtrTable make_ctr_table(std::vector<uint32_t> cols, float default_value,
                                    const std::vector<std::pair<std::vector<uint32_t>, float>>& entries) {
    lignum::ir::CtrTable t;
    t.columns = std::move(cols);
    t.default_value = default_value;
    if (entries.empty()) return t;
    size_t cap = 4;
    while (cap < entries.size() * 2 + 2) cap <<= 1;
    t.mask = static_cast<uint32_t>(cap - 1);
    t.keys.assign(cap, 0ULL);
    t.slots.assign(cap, 0U);
    t.values.reserve(entries.size());
    for (const auto& [hashes, val] : entries) {
        uint64_t h = hashes[0];
        for (size_t k = 1; k < hashes.size(); k++) h = combine_cat_hash_ref(h, hashes[k]);
        uint64_t idx = (h ^ (h >> 29U)) & t.mask;
        while (t.slots[idx] != 0) idx = (idx + 1U) & t.mask;
        t.values.push_back(val);
        t.keys[idx] = h;
        t.slots[idx] = static_cast<uint32_t>(t.values.size());
    }
    return t;
}

// Helper columnar source over row-major double matrix for testing ColumnSource / predict_columns.
struct RowMajorColumnSource : lignum::ColumnSource {
    const double* X = nullptr;
    size_t n_features = 0;

    void read(size_t column, size_t offset, size_t count, double* out) const override {
        for (size_t i = 0; i < count; i++) out[i] = X[(offset + i) * n_features + column];
    }
};

// Random asymmetric forests of oblique, axis and categorical splits (complete depth-5 trees);
// every 7th tree has no oblique split (BNode walk in the same model).
lignum::ir::Forest random_forest(uint64_t seed) {
    std::mt19937_64 rng(seed);
    lignum::ir::Forest f;
    f.n_features = 6;  // column 5 is categorical (codes 0..9)
    f.base_score = -0.25;
    f.quantizers.resize(6);
    f.quantizer_constant.assign(6, 0);
    for (int j = 0; j < 6; j++) f.quantizers[static_cast<size_t>(j)] = lignum::FeatureQuantizer{-5.0 - j, 6000.0 + 500.0 * j};
    f.quantizer_constant[4] = 1;
    for (int ti = 0; ti < 60; ti++) {
        lignum::ir::Tree t;
        const bool plain = ti % 7 == 3;
        for (uint32_t i = 0; i < 63; i++) {
            if (i >= 31) {
                t.nodes.push_back(leaf(std::uniform_real_distribution<double>(-1, 1)(rng)));
                continue;
            }
            const uint32_t l = 2 * i + 1, r = 2 * i + 2;
            const int kind = static_cast<int>(rng() % 5);
            if (kind <= 1 && !plain) {
                const uint8_t k = static_cast<uint8_t>(2 + rng() % 2);
                const uint32_t f0 = rng() % 5, f1 = rng() % 5;
                const uint32_t f2 = k == 3 ? static_cast<uint32_t>(rng() % 5) : f0;
                const auto w = [&] { return static_cast<int16_t>(static_cast<int>(rng() % 32769) - 16384); };
                const int16_t w0 = w(), w1 = w(), w2 = k == 3 ? w() : int16_t{0};
                t.nodes.push_back(oblique(k, f0, f1, f2, w0, w1, w2,
                                          static_cast<int32_t>(static_cast<int64_t>(rng() % (1u << 27)) - (1 << 26)),
                                          rng() % 2 == 0, l, r));
            } else if (kind == 2) {
                std::vector<uint32_t> cats;
                for (uint32_t v = 0; v < 10; v++) {
                    if (rng() % 2) cats.push_back(v);
                }
                t.nodes.push_back(categorical(5, std::move(cats), l, r));
            } else {
                t.nodes.push_back(numeric(static_cast<uint32_t>(rng() % 5),
                                          0.5 * static_cast<double>(static_cast<int>(rng() % 17) - 8),
                                          rng() % 2 == 0, l, r));
            }
        }
        // Vary depth slightly on tree 0 so SymForest detector does not misclassify an asymmetric forest.
        if (ti == 0 && t.nodes[2].kind != lignum::ir::Node::Kind::Leaf) {
            t.nodes[2] = numeric(0, 0.25, true, 5, 6);
        }
        f.trees.push_back(t);
    }
    return f;
}

std::vector<double> random_rows(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<double> X(n * 6);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 5; j++) {
            const auto r = rng() % 10;
            X[i * 6 + j] = r == 0 ? kNan
                                  : (r < 4 ? 0.5 * static_cast<double>(static_cast<int>(rng() % 17) - 8)
                                           : std::normal_distribution<double>(0, 4)(rng));
        }
        X[i * 6 + 5] = (rng() % 13 == 0) ? kNan : static_cast<double>(rng() % 12);
    }
    return X;
}

std::vector<char> file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

TEST(Oblique, ThreeInputSplitRoutesExactly) {
    auto f = base_forest();
    lignum::ir::Tree t;
    t.nodes = {oblique(3, 0, 1, 2, 2, -3, 1, 10, true, 1, 2), leaf(-1.0), leaf(+1.0)};
    f.trees.push_back(t);
    const auto m = lignum::build_model(f, /*binned=*/false);
    EXPECT_EQ(m.n_oblique, 1U);
    EXPECT_EQ(m.source, lignum::model_source::Miter);

    const auto P = [](double a, double b, double c) {
        const auto code = [](double x) { return static_cast<int32_t>(std::floor(x + 0.5)) - 32768; };
        return 2 * code(a) - 3 * code(b) + code(c);
    };
    const double a = 32768.0 + 5.0, b = 32768.0, c = 32768.0;  // P = 10
    ASSERT_EQ(P(a, b, c), 10);
    EXPECT_EQ(predict_one(m, {a, b, c}), 0.5 + 1.0);
    EXPECT_EQ(predict_one(m, {a, b, c - 1.0}), 0.5 - 1.0);
    EXPECT_EQ(predict_one(m, {32772.5, b, c}), 1.5);
    EXPECT_EQ(predict_one(m, {32772.4999, b, c}), -0.5);
    for (int q = 0; q < 3; q++) {
        std::vector<double> row = {0.0, 1e9, 0.0};
        EXPECT_EQ(predict_one(m, row), -0.5);
        row[static_cast<size_t>(q)] = kNan;
        EXPECT_EQ(predict_one(m, row), 1.5) << "NaN in input " << q;
    }
    const double lo_c = -32768.0, hi_c = 65534.0 - 32768.0;
    EXPECT_EQ(predict_one(m, {-1e300, 0.0, 0.0}),
              (2 * lo_c - 3 * lo_c + lo_c >= 10) ? 1.5 : -0.5);
    EXPECT_EQ(predict_one(m, {std::numeric_limits<double>::infinity(), 0.0, 0.0}),
              (2 * hi_c - 3 * lo_c + lo_c >= 10) ? 1.5 : -0.5);
}

TEST(Oblique, TwoInputsConstantColumnAndMixedTree) {
    auto f = base_forest();
    f.quantizer_constant[1] = 1;
    lignum::ir::Tree t;
    t.nodes = {oblique(2, 0, 1, 0, 1, 1, 0, -2, false, 1, 2), leaf(10.0), numeric(2, 7.25, true, 3, 4),
               leaf(20.0), leaf(30.0)};
    f.trees.push_back(t);
    const auto m = lignum::build_model(f, false);
    EXPECT_EQ(predict_one(m, {65533.0, 123.0, 0.0}), 0.5 + 10.0);
    EXPECT_EQ(predict_one(m, {1e9, 123.0, 7.25}), 0.5 + 30.0);
    EXPECT_EQ(predict_one(m, {1e9, 123.0, 7.0}), 0.5 + 20.0);
    EXPECT_EQ(predict_one(m, {1e9, 123.0, kNan}), 0.5 + 30.0);
    EXPECT_EQ(predict_one(m, {kNan, 123.0, 9.0}), 0.5 + 10.0);
}

TEST(Oblique, BatchesMatchSingleRowsAndThreads) {
    std::mt19937_64 rng(7);
    auto f = base_forest();
    f.n_features = 4;
    f.quantizers.assign(4, lignum::FeatureQuantizer{-10.0, 3000.0});
    f.quantizer_constant.assign(4, 0);
    for (int ti = 0; ti < 40; ti++) {
        lignum::ir::Tree t;
        for (uint32_t i = 0; i < 31; i++) {
            if (i >= 15) {
                t.nodes.push_back(leaf(std::uniform_real_distribution<double>(-1, 1)(rng)));
                continue;
            }
            const uint32_t l = 2 * i + 1, r = 2 * i + 2;
            if (rng() % 2) {
                const uint8_t k = static_cast<uint8_t>(2 + rng() % 2);
                const uint32_t f0 = rng() % 4, f1 = rng() % 4;
                const uint32_t f2 = k == 3 ? static_cast<uint32_t>(rng() % 4) : f0;
                const auto w = [&] { return static_cast<int16_t>(static_cast<int>(rng() % 32769) - 16384); };
                const int16_t w0 = w(), w1 = w(), w2 = k == 3 ? w() : int16_t{0};
                t.nodes.push_back(oblique(k, f0, f1, f2, w0, w1, w2,
                                          static_cast<int32_t>(static_cast<int64_t>(rng() % (1u << 28)) - (1 << 27)),
                                          rng() % 2 == 0, l, r));
            } else {
                t.nodes.push_back(numeric(static_cast<uint32_t>(rng() % 4), std::normal_distribution<double>(0, 3)(rng),
                                          rng() % 2 == 0, l, r));
            }
        }
        f.trees.push_back(t);
    }
    const auto m = lignum::build_model(f, false);
    constexpr size_t n = 1003;
    std::vector<double> X(n * 4);
    for (size_t i = 0; i < X.size(); i++) {
        X[i] = (i % 37 == 0) ? kNan : std::normal_distribution<double>(0, 4)(rng);
    }
    std::vector<double> batch(n), threaded(n);
    m.predict_raw(X.data(), n, 4, batch.data(), 1);
    m.predict_raw(X.data(), n, 4, threaded.data(), 4);
    for (size_t i = 0; i < n; i++) {
        double one = 0.0;
        m.predict_raw(&X[i * 4], 1, 4, &one, 1);
        ASSERT_EQ(std::bit_cast<uint64_t>(batch[i]), std::bit_cast<uint64_t>(one)) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(threaded[i]), std::bit_cast<uint64_t>(one)) << i;
    }
}

TEST(Oblique, BinnedUsesONodes) {
    auto f = base_forest();
    lignum::ir::Tree t;
    t.nodes = {oblique(2, 0, 1, 0, 1, 1, 0, 0, false, 1, 2), leaf(1.0), leaf(2.0)};
    f.trees.push_back(t);
    const auto m = lignum::build_model(f, /*binned=*/true);
    ASSERT_NE(m.binned, nullptr);
    EXPECT_EQ(m.binned->n_onodes, 1U);
    EXPECT_EQ(m.binned->n_q16, 2U);
}

TEST(Oblique, BinnedEqualsKNodeBitwise) {
    const auto f = random_forest(2027);
    const auto knode = lignum::build_model(f, false);
    const auto binned = lignum::build_model(f, true);
    ASSERT_NE(binned.binned, nullptr);
    ASSERT_GT(binned.binned->n_onodes, 0U);
    ASSERT_GT(binned.binned->n_nodes, 0U);

    constexpr size_t n = 2003;
    const std::vector<double> X = random_rows(n, 99);
    for (const int threads : {1, 3}) {
        std::vector<double> a(n), b(n);
        knode.predict_raw(X.data(), n, 6, a.data(), threads);
        binned.predict_raw(X.data(), n, 6, b.data(), threads);
        for (size_t i = 0; i < n; i++) {
            ASSERT_EQ(std::bit_cast<uint64_t>(a[i]), std::bit_cast<uint64_t>(b[i])) << "row " << i << " threads " << threads;
        }
    }
    for (size_t i = 0; i < 50; i++) {
        double a = 0.0, b = 0.0;
        knode.predict_raw(&X[i * 6], 1, 6, &a, 1);
        binned.predict_raw(&X[i * 6], 1, 6, &b, 1);
        ASSERT_EQ(std::bit_cast<uint64_t>(a), std::bit_cast<uint64_t>(b)) << "single row " << i;
    }
}

TEST(Oblique, NumericOnlyForestBuildsTheBinnedKernel) {
    lignum::ir::Forest f;
    f.n_features = 2;
    lignum::ir::Tree t;
    t.nodes = {numeric(1, 0.5, false, 1, 2), leaf(-2.0), leaf(3.0)};
    f.trees.push_back(t);
    const auto m = lignum::build_model(f, true);
    EXPECT_NE(m.binned, nullptr);
    EXPECT_EQ(predict_one(m, {0.0, 0.5}), 3.0);
    EXPECT_EQ(predict_one(m, {0.0, std::nextafter(0.5, 0.0)}), -2.0);
    EXPECT_EQ(predict_one(m, {0.0, kNan}), -2.0);
}

TEST(Oblique, InvalidForestsAreRejected) {
    const auto with = [](lignum::ir::Node root) {
        auto f = base_forest();
        lignum::ir::Tree t;
        t.nodes = {root, leaf(1.0), leaf(2.0)};
        f.trees.push_back(t);
        return f;
    };
    EXPECT_THROW(lignum::build_model(with(oblique(4, 0, 1, 2, 1, 1, 1, 0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(3, 0, 1, 3, 1, 1, 1, 0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(2, 0, 1, 2, 1, 1, 0, 0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(3, 0, 1, 2, 20000, 1, 1, 0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(3, 0, 1, 2, 0, 0, 0, 0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(3, 0, 1, 2, 1, 1, 1, 0, false, 0, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(oblique(3, 0, 1, 2, 1, 1, 1, 0, false, 1, 9))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(numeric(5, 0.0, false, 1, 2))), std::invalid_argument);
    EXPECT_THROW(lignum::build_model(with(numeric(0, kNan, false, 1, 2))), std::invalid_argument);
    auto no_quant = with(oblique(3, 0, 1, 2, 1, 1, 1, 0, false, 1, 2));
    no_quant.quantizers.clear();
    no_quant.quantizer_constant.clear();
    EXPECT_THROW(lignum::build_model(no_quant), std::invalid_argument);
    auto shared = with(numeric(0, 0.0, false, 1, 1));
    EXPECT_THROW(lignum::build_model(shared), std::invalid_argument);
}

TEST(Oblique, SaveAndLoadKeepPredictionsBitwise) {
    const auto f = random_forest(31);
    const auto model = lignum::build_model(f, true);
    constexpr size_t n = 1001;
    const std::vector<double> X = random_rows(n, 5);
    std::vector<double> ref(n);
    model.predict_raw(X.data(), n, 6, ref.data(), 1);
    const auto path = (std::filesystem::temp_directory_path() / "lignum_oblique_io.bin").string();
    model.save(path);
    const auto bytes = file_bytes(path);
    ASSERT_GT(bytes.size(), 64U);
    EXPECT_EQ(bytes[7], 10);
    for (const bool binned : {true, false}) {
        const auto loaded = lignum::load(path, "auto", binned);
        EXPECT_EQ(loaded.source, lignum::model_source::Miter);
        EXPECT_EQ(loaded.n_oblique, model.n_oblique);
        EXPECT_EQ(loaded.binned != nullptr, binned);
        if (binned) EXPECT_EQ(loaded.binned->n_onodes, model.binned->n_onodes);
        std::vector<double> got(n);
        loaded.predict_raw(X.data(), n, 6, got.data(), 2);
        for (size_t i = 0; i < n; i++) {
            ASSERT_EQ(std::bit_cast<uint64_t>(got[i]), std::bit_cast<uint64_t>(ref[i])) << "row " << i << " binned " << binned;
        }
    }
}

TEST(Oblique, CorruptedObliqueFilesAreRejectedOrSafe) {
    const auto model = lignum::build_model(random_forest(41), true);
    const auto path = (std::filesystem::temp_directory_path() / "lignum_oblique_fuzz.bin").string();
    model.save(path);
    const std::vector<char> original = file_bytes(path);
    const size_t last_onode = original.size() - 64;
    const auto rejected = [&](const std::vector<char>& bytes, const std::string& what) {
        const auto bad = (std::filesystem::temp_directory_path() / "lignum_oblique_bad.bin").string();
        write_bytes(bad, bytes);
        EXPECT_THROW(lignum::load(bad, "auto", true), std::runtime_error) << what;
    };
    {
        std::vector<char> bytes = original;
        const uint16_t big = 0x7FFF;
        std::memcpy(bytes.data() + last_onode + 2, &big, sizeof(big));
        rejected(bytes, "ONode code column out of range");
    }
    {
        std::vector<char> bytes = original;
        const int32_t child = 1 << 30;
        std::memcpy(bytes.data() + last_onode + 48, &child, sizeof(child));
        rejected(bytes, "ONode child out of range");
    }
    {
        std::vector<char> bytes = original;
        const int16_t w = 20000;
        std::memcpy(bytes.data() + last_onode + 6, &w, sizeof(w));
        rejected(bytes, "ONode weight out of range");
    }
    std::mt19937_64 rng(4242);
    std::vector<double> X = random_rows(9, 1);
    for (int trial = 0; trial < 400; trial++) {
        std::vector<char> bytes = original;
        const int edits = 1 + static_cast<int>(rng() % 4);
        for (int e = 0; e < edits; e++) bytes[rng() % bytes.size()] = static_cast<char>(rng());
        const auto bad = (std::filesystem::temp_directory_path() / "lignum_oblique_bad.bin").string();
        write_bytes(bad, bytes);
        for (const bool binned : {true, false}) {
            try {
                const auto m = lignum::load(bad, "auto", binned);
                if (m.n_features > 6) continue;
                std::vector<double> out(9 * m.output_width(true));
                m.predict_raw(X.data(), 9, 6, out.data(), 1);
            } catch (const std::exception&) {
            }
        }
    }
}

TEST(Oblique, FileWithOutOfRangeObliqueWeightIsRejected) {
    auto f = base_forest();
    lignum::ir::Tree t;
    t.nodes = {oblique(3, 0, 1, 2, 1234, -2345, 3456, 0, false, 1, 2), leaf(1.0), leaf(2.0)};
    f.trees.push_back(t);
    const auto path = (std::filesystem::temp_directory_path() / "lignum_oblique_weight.bin").string();
    lignum::build_model(f, false).save(path);
    std::vector<char> bytes = file_bytes(path);
    const int16_t pattern[3] = {1234, -2345, 3456};
    const auto it = std::search(bytes.begin(), bytes.end(), reinterpret_cast<const char*>(pattern),
                                reinterpret_cast<const char*>(pattern) + sizeof(pattern));
    ASSERT_NE(it, bytes.end());
    const int16_t big = 20000;
    std::memcpy(&*it, &big, sizeof(big));
    write_bytes(path, bytes);
    for (const bool binned : {true, false}) EXPECT_THROW(lignum::load(path, "auto", binned), std::runtime_error);
}

TEST(Oblique, TreeKindMustMatchObliqueSlots) {
    auto f = base_forest();
    lignum::ir::Tree t;
    t.nodes = {oblique(2, 0, 1, 0, 1, 1, 0, 0, false, 1, 2), leaf(1.0), leaf(2.0)};
    f.trees.push_back(t);
    lignum::ir::Tree plain;
    plain.nodes = {numeric(2, 0.5, false, 1, 2), leaf(3.0), leaf(4.0)};
    f.trees.push_back(plain);
    const auto path = (std::filesystem::temp_directory_path() / "lignum_oblique_kind.bin").string();
    lignum::build_model(f, false).save(path);
    const std::vector<char> original = file_bytes(path);
    const auto get64 = [&](size_t pos) { uint64_t v; std::memcpy(&v, original.data() + pos, sizeof(v)); return v; };
    const auto get32 = [&](size_t pos) { uint32_t v; std::memcpy(&v, original.data() + pos, sizeof(v)); return v; };
    const auto align = [](uint64_t n) { return (n + 63) & ~uint64_t{63}; };
    const uint64_t kinds = 128 + align(get64(88) + get32(104) + get32(108)) + align(get64(24) * 64) + align(get64(32) * 4) +
                           align(get64(40) * 8);
    ASSERT_EQ(original[kinds], static_cast<char>(lignum::tree_kind::Oblique));
    ASSERT_EQ(original[kinds + 1], 0);
    for (const auto& [pos, value] : {std::pair<uint64_t, char>{kinds, 0}, std::pair<uint64_t, char>{kinds + 1, lignum::tree_kind::Oblique}}) {
        std::vector<char> bytes = original;
        bytes[pos] = value;
        write_bytes(path, bytes);
        for (const bool binned : {true, false}) EXPECT_THROW(lignum::load(path, "auto", binned), std::runtime_error) << pos;
    }
}

TEST(Oblique, FuzzRegressionsAreRejectedSafely) {
    const std::filesystem::path reg_dir = std::filesystem::path(__FILE__).parent_path().parent_path() / "fuzz" / "regressions" / "lignum";
    ASSERT_TRUE(std::filesystem::exists(reg_dir));
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(reg_dir)) {
        if (entry.path().filename().string().rfind("oblique-", 0) != 0) continue;
        count++;
        for (const bool binned : {true, false}) {
            EXPECT_THROW(lignum::load(entry.path().string(), "auto", binned), std::runtime_error)
                << entry.path().filename().string() << " binned=" << binned;
        }
    }
    EXPECT_EQ(count, 6);
}

TEST(Oblique, CompiledEqualsKNodeBitwise) {
    const auto f = random_forest(2027);
    const auto knode = lignum::build_model(f, false);
    auto model = lignum::build_model(f, true);
    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_oblique_compiled").string();
    std::filesystem::remove_all(o.cache_dir);
    o.jobs = 2;
    o.block_trees = 16;
    model.compiled = lignum::compile_forest(model, o, nullptr);
    ASSERT_NE(model.compiled, nullptr);

    constexpr size_t n = 2003;
    std::vector<double> X = random_rows(n, 77);
    const double specials[] = {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 1e300, -1e300, 0.0, -0.0};
    for (size_t i = 0; i < 60; i++) X[i * 6 + i % 5] = specials[i % 6];
    for (const int threads : {1, 3}) {
        std::vector<double> a(n), b(n);
        knode.predict_raw(X.data(), n, 6, a.data(), threads);
        model.predict_raw(X.data(), n, 6, b.data(), threads);
        for (size_t i = 0; i < n; i++) {
            ASSERT_EQ(std::bit_cast<uint64_t>(a[i]), std::bit_cast<uint64_t>(b[i])) << "row " << i << " threads " << threads;
        }
    }
    for (size_t i = 0; i < 64; i++) {
        double a = 0.0, b = 0.0;
        knode.predict_raw(&X[i * 6], 1, 6, &a, 1);
        model.predict_raw(&X[i * 6], 1, 6, &b, 1);
        ASSERT_EQ(std::bit_cast<uint64_t>(a), std::bit_cast<uint64_t>(b)) << "single row " << i;
    }

    const uint64_t base = lignum::compiled_fingerprint(model);
    auto changed = f;
    for (auto& t : changed.trees) {
        for (auto& nd : t.nodes) {
            if (nd.kind == lignum::ir::Node::Kind::Oblique) {
                nd.oblique.weights[0] = static_cast<int16_t>(nd.oblique.weights[0] == 7 ? 8 : 7);
                goto done_weight;
            }
        }
    }
done_weight:
    EXPECT_NE(lignum::compiled_fingerprint(lignum::build_model(changed, true)), base);
    changed = f;
    for (auto& t : changed.trees) {
        for (auto& nd : t.nodes) {
            if (nd.kind == lignum::ir::Node::Kind::Oblique) {
                nd.oblique.threshold += 1;
                goto done_threshold;
            }
        }
    }
done_threshold:
    EXPECT_NE(lignum::compiled_fingerprint(lignum::build_model(changed, true)), base);
    changed = f;
    changed.quantizers[0].scale *= 2.0;
    EXPECT_NE(lignum::compiled_fingerprint(lignum::build_model(changed, true)), base);
}

// ---- miter v1.1 features ----

TEST(Oblique, StableSigmoidMatchesMiterExactly) {
    auto f = base_forest();
    f.transform = lignum::transform::Sigmoid;
    f.base_score = 0.0;
    lignum::ir::Tree t;
    t.nodes = {oblique(2, 0, 1, 0, 1, 0, 0, 0, false, 1, 2), leaf(-1000.0), leaf(1000.0)};
    lignum::ir::Tree t2;
    t2.nodes = {numeric(2, 0.5, false, 1, 2), leaf(-1.25), leaf(2.75)};
    f.trees = {t, t2};

    const auto knode = lignum::build_model(f, false);
    auto binned = lignum::build_model(f, true);
    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_miter_sigmoid").string();
    std::filesystem::remove_all(o.cache_dir);
    auto compiled = lignum::build_model(f, true);
    compiled.compiled = lignum::compile_forest(compiled, o, nullptr);

    std::vector<double> X = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 40000.0, 0.0, 0.0, 40000.0, 0.0, 1.0};
    auto miter_sig = [](double z) {
        if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
        const double ez = std::exp(z);
        return ez / (1.0 + ez);
    };
    const double margins[4] = {-1001.25, -997.25, 998.75, 1002.75};
    std::vector<double> p0(4), p1(4), p2(4);
    knode.predict(X.data(), 4, 3, p0.data(), 1);
    binned.predict(X.data(), 4, 3, p1.data(), 1);
    compiled.predict(X.data(), 4, 3, p2.data(), 1);
    for (size_t i = 0; i < 4; i++) {
        const double exp_p = miter_sig(margins[i]);
        EXPECT_EQ(std::bit_cast<uint64_t>(p0[i]), std::bit_cast<uint64_t>(exp_p)) << i;
        EXPECT_EQ(std::bit_cast<uint64_t>(p1[i]), std::bit_cast<uint64_t>(exp_p)) << i;
        EXPECT_EQ(std::bit_cast<uint64_t>(p2[i]), std::bit_cast<uint64_t>(exp_p)) << i;
    }
}

TEST(Oblique, MulticlassAndSoftmaxMatchBitwise) {
    auto f = random_forest(909);
    constexpr size_t K = 3;
    f.base_score = 0.0;
    f.base_scores = {0.1, -0.3, 0.25};
    f.transform = lignum::transform::Softmax;
    f.tree_outputs.resize(f.trees.size());
    for (size_t t = 0; t < f.trees.size(); t++) f.tree_outputs[t] = static_cast<uint16_t>(t % K);

    const auto knode = lignum::build_model(f, false);
    const auto binned = lignum::build_model(f, true);
    ASSERT_NE(binned.binned, nullptr);
    ASSERT_GT(binned.binned->n_onodes, 0U);

    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_miter_multi").string();
    std::filesystem::remove_all(o.cache_dir);
    auto compiled = lignum::build_model(f, true);
    compiled.compiled = lignum::compile_forest(compiled, o, nullptr);

    const auto path = (std::filesystem::temp_directory_path() / "lignum_miter_multi.bin").string();
    binned.save(path);
    const auto loaded = lignum::load(path, "auto", true);

    constexpr size_t n = 123;
    const auto X = random_rows(n, 333);
    std::vector<double> r0(n * K), r1(n * K), r2(n * K), r3(n * K);
    std::vector<double> p0(n * K), p1(n * K), p2(n * K), p3(n * K);
    knode.predict_raw(X.data(), n, 6, r0.data(), 1);
    binned.predict_raw(X.data(), n, 6, r1.data(), 2);
    compiled.predict_raw(X.data(), n, 6, r2.data(), 2);
    loaded.predict_raw(X.data(), n, 6, r3.data(), 2);
    knode.predict(X.data(), n, 6, p0.data(), 1);
    binned.predict(X.data(), n, 6, p1.data(), 2);
    compiled.predict(X.data(), n, 6, p2.data(), 2);
    loaded.predict(X.data(), n, 6, p3.data(), 2);

    for (size_t i = 0; i < n; i++) {
        // Compute reference softmax with inv_sum = 1.0 / sum_exp matching miter.
        double wmax = std::max({r0[i * K], r0[i * K + 1], r0[i * K + 2]});
        double e[3] = {std::exp(r0[i * K] - wmax), std::exp(r0[i * K + 1] - wmax), std::exp(r0[i * K + 2] - wmax)};
        double inv_sum = 1.0 / (e[0] + e[1] + e[2]);
        for (size_t k = 0; k < K; k++) {
            const size_t idx = i * K + k;
            ASSERT_EQ(std::bit_cast<uint64_t>(r0[idx]), std::bit_cast<uint64_t>(r1[idx])) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(r0[idx]), std::bit_cast<uint64_t>(r2[idx])) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(r0[idx]), std::bit_cast<uint64_t>(r3[idx])) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(e[k] * inv_sum)) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p1[idx])) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p2[idx])) << idx;
            ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p3[idx])) << idx;
        }
    }
}

TEST(Oblique, QuantileQuantizerMatchesKNodeBinnedCompiledAndIO) {
    auto f = random_forest(1203);
    // Give columns 0 and 2 piecewise-linear quantile knots (3 and 17 knots); keep others affine / constant.
    f.quantizer_knots.resize(6);
    f.quantizer_knots[0] = {-10.0, 0.0, 10.0};
    f.quantizer_knots[2].resize(17);
    for (int k = 0; k < 17; k++) f.quantizer_knots[2][static_cast<size_t>(k)] = -8.0 + 0.1 * k * k;

    const auto knode = lignum::build_model(f, false);
    const auto binned = lignum::build_model(f, true);
    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_quantile_q16").string();
    std::filesystem::remove_all(o.cache_dir);
    auto compiled = lignum::build_model(f, true);
    compiled.compiled = lignum::compile_forest(compiled, o, nullptr);

    const auto path = (std::filesystem::temp_directory_path() / "lignum_quantile_io.bin").string();
    binned.save(path);
    const auto loaded = lignum::load(path, "auto", true);

    constexpr size_t n = 1029;
    std::vector<double> X = random_rows(n, 71);
    // Include exact knot values, bounds, midpoints, and NaNs.
    for (size_t k = 0; k < 17; k++) {
        X[k * 6 + 2] = f.quantizer_knots[2][k];
        X[(k + 20) * 6 + 2] = f.quantizer_knots[2][k] + 0.05;
    }
    X[0] = -10.0; X[6] = 0.0; X[12] = 10.0; X[18] = kNan;

    std::vector<double> a(n), b(n), c(n), d(n);
    knode.predict_raw(X.data(), n, 6, a.data(), 1);
    binned.predict_raw(X.data(), n, 6, b.data(), 2);
    compiled.predict_raw(X.data(), n, 6, c.data(), 2);
    loaded.predict_raw(X.data(), n, 6, d.data(), 2);
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(std::bit_cast<uint64_t>(a[i]), std::bit_cast<uint64_t>(b[i])) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(a[i]), std::bit_cast<uint64_t>(c[i])) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(a[i]), std::bit_cast<uint64_t>(d[i])) << i;
    }

    // Fingerprint changes when a quantile knot changes.
    auto altered = f;
    altered.quantizer_knots[2][8] += 0.01;
    EXPECT_NE(lignum::compiled_fingerprint(lignum::build_model(altered, true)), lignum::compiled_fingerprint(binned));

    // Invalid knots rejected.
    auto bad = f;
    bad.quantizer_knots[0] = {1.0, 1.0};
    EXPECT_THROW(lignum::build_model(bad), std::invalid_argument);
    bad.quantizer_knots[0] = {1.0};
    EXPECT_THROW(lignum::build_model(bad), std::invalid_argument);
}

TEST(Oblique, NativeCtrTablesWorkAcrossAllKernelsAndIO) {
    lignum::ir::Forest f;
    f.n_features = 5;  // cols 0, 1: numeric/oblique; cols 2, 3, 4: cat hash columns as doubles
    f.base_score = 0.2;
    f.transform = lignum::transform::Sigmoid;
    f.quantizers.assign(5, lignum::FeatureQuantizer{-5.0, 6000.0});
    f.quantizer_constant.assign(5, 0);

    f.ctr_tables.push_back(make_ctr_table({2}, 0.25f, {{{10U}, 0.8f}, {{20U}, 0.1f}}));
    f.ctr_tables.push_back(make_ctr_table({2, 3, 4}, 0.5f, {{{10U, 20U, 30U}, 0.9f}, {{0x9e3779b9U, 1U, 2U}, 0.15f}}));
    f.ctr_tables.push_back(make_ctr_table({4}, 0.42f, {}));  // empty table -> default_value

    lignum::ir::Tree t1;
    t1.nodes = {
        ctr_split(0, 0.5f, 1, 2),
        oblique(2, 0, 1, 0, 3, -2, 0, 100, false, 3, 4),
        ctr_split(1, 0.7f, 5, 6),
        leaf(-0.5), leaf(0.4), leaf(0.1), leaf(0.9),
    };
    lignum::ir::Tree t2;  // CTR + numeric only (BNode walk in non-sym binned)
    t2.nodes = {ctr_split(2, 0.40f, 1, 2), leaf(-1.0), numeric(0, 0.5, true, 3, 4), leaf(0.2), leaf(0.6)};
    f.trees = {t1, t2};

    const auto knode = lignum::build_model(f, false);
    const auto binned = lignum::build_model(f, true);
    ASSERT_NE(binned.binned, nullptr);
    EXPECT_TRUE(binned.binned->has_ctr_features);

    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_ctr_compiled").string();
    std::filesystem::remove_all(o.cache_dir);
    auto compiled = lignum::build_model(f, true);
    compiled.compiled = lignum::compile_forest(compiled, o, nullptr);

    const auto path = (std::filesystem::temp_directory_path() / "lignum_ctr_io.bin").string();
    binned.save(path);
    const auto loaded = lignum::load(path, "auto", true);

    std::vector<double> X = {
        0.0, 0.0, 10.0, 20.0, 30.0,     // tbl0=0.8 (>=0.5), tbl1=0.9 (>=0.7), tbl2=0.42 (>=0.40)
        1.0, -1.0, 20.0, 0.0, 0.0,      // tbl0=0.1 (<0.5) -> oblique
        kNan, 2.0, 99.0, 5.0, 6.0,      // tbl0=default(0.25), oblique NaN -> default left
        0.5, 0.5, kNan, 1.0, 2.0,       // cat0 NaN -> 0x9e3779b9 -> tbl1=0.15
        -2.0, 3.0, -5.0, 1.0, 2.0,      // negative hash -> kMissingCatHash -> tbl1=0.15
    };
    const size_t n = X.size() / 5;
    std::vector<double> p0(n), p1(n), p2(n), p3(n), p4(n);
    knode.predict(X.data(), n, 5, p0.data(), 1);
    binned.predict(X.data(), n, 5, p1.data(), 2);
    compiled.predict(X.data(), n, 5, p2.data(), 2);
    loaded.predict(X.data(), n, 5, p3.data(), 2);

    RowMajorColumnSource col_src;
    col_src.X = X.data();
    col_src.n_features = 5;
    lignum::predict_columns(binned, col_src, n, 5, p4.data(), false, 1, lignum::Strategy::Serial);

    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[i]), std::bit_cast<uint64_t>(p1[i])) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[i]), std::bit_cast<uint64_t>(p2[i])) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[i]), std::bit_cast<uint64_t>(p3[i])) << i;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[i]), std::bit_cast<uint64_t>(p4[i])) << i;
    }

    // Malformed CTR tables (including a full probe table with no empty 0 slot) must be rejected.
    auto full_tbl = f;
    full_tbl.ctr_tables[0].slots.assign(full_tbl.ctr_tables[0].slots.size(), 1U);
    EXPECT_THROW(lignum::build_model(full_tbl), std::invalid_argument);
    auto non_pow2 = f;
    non_pow2.ctr_tables[0].keys.resize(3);
    non_pow2.ctr_tables[0].slots.resize(3, 0);
    non_pow2.ctr_tables[0].mask = 2;
    EXPECT_THROW(lignum::build_model(non_pow2), std::invalid_argument);
}

TEST(Oblique, SymmetricObliqueKernelMatchesKNodeAndCompiledBitwise) {
    // Build a forest of level-wise symmetric trees (depth 6, alternating oblique, axis, categorical, CTR).
    std::mt19937_64 rng(555);
    lignum::ir::Forest f;
    f.n_features = 7;  // 0..4 numeric/oblique, 5 categorical, 6 cat hash for CTR
    f.base_scores = {0.1, -0.2, 0.05};
    f.transform = lignum::transform::Softmax;
    f.quantizers.assign(7, lignum::FeatureQuantizer{-4.0, 5000.0});
    f.quantizer_constant.assign(7, 0);
    f.quantizer_knots.resize(7);
    f.quantizer_knots[1] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    f.ctr_tables.push_back(make_ctr_table({6}, 0.3f, {{{1U}, 0.8f}, {{2U}, 0.1f}, {{3U}, 0.55f}}));

    for (int ti = 0; ti < 18; ti++) {
        const uint32_t depth = 6;
        const uint32_t n_int = (1U << depth) - 1U;
        const uint32_t n_tot = (1U << (depth + 1U)) - 1U;
        lignum::ir::Tree t;
        t.nodes.resize(n_tot);
        std::vector<lignum::ir::Node> level_specs(depth);
        for (uint32_t lv = 0; lv < depth; lv++) {
            lignum::ir::Node spec;
            const uint32_t mode = (lv + static_cast<uint32_t>(ti)) % 4U;
            if (mode == 0) {
                spec = oblique(3, 0, 1, 2, 5, -3, 2, 1000 * static_cast<int32_t>(lv + 1) - 3000, lv % 2 == 0, 0, 0);
            } else if (mode == 1) {
                spec = numeric(lv % 5U, 0.25 * static_cast<double>(static_cast<int>(lv) - 2), lv % 2 == 1, 0, 0);
            } else if (mode == 2) {
                spec = categorical(5, {0, 2, 4, 6}, 0, 0);
            } else {
                spec = ctr_split(0, 0.4f, 0, 0);
            }
            level_specs[lv] = std::move(spec);
        }
        for (uint32_t lv = 0; lv < depth; lv++) {
            const uint32_t start = (1U << lv) - 1U;
            const uint32_t count = 1U << lv;
            for (uint32_t k = 0; k < count; k++) {
                const uint32_t id = start + k;
                t.nodes[id] = level_specs[lv];
                t.nodes[id].left = 2U * id + 1U;
                t.nodes[id].right = 2U * id + 2U;
            }
        }
        for (uint32_t id = n_int; id < n_tot; id++) {
            t.nodes[id] = leaf(std::uniform_real_distribution<double>(-0.5, 0.5)(rng));
        }
        f.trees.push_back(std::move(t));
        f.tree_outputs.push_back(static_cast<uint16_t>(ti % 3));
    }

    const auto knode = lignum::build_model(f, false);
    const auto binned = lignum::build_model(f, true);
    ASSERT_NE(binned.binned, nullptr);
    ASSERT_NE(binned.binned->sym, nullptr);
    EXPECT_EQ(binned.binned->sym->trees.size(), 18U);
    EXPECT_EQ(binned.binned->sym->trees[0].depth, 6U);

    lignum::CompileOptions o;
    o.cache_dir = (std::filesystem::temp_directory_path() / "lignum_test_sym_compiled").string();
    std::filesystem::remove_all(o.cache_dir);
    auto compiled = lignum::build_model(f, true);
    compiled.compiled = lignum::compile_forest(compiled, o, nullptr);

    const auto path = (std::filesystem::temp_directory_path() / "lignum_sym_io.bin").string();
    binned.save(path);
    const auto loaded = lignum::load(path, "auto", true);
    ASSERT_NE(loaded.binned, nullptr);
    ASSERT_NE(loaded.binned->sym, nullptr);

    constexpr size_t n = 259;
    std::vector<double> X(n * 7);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < 5; j++) {
            X[i * 7 + j] = (rng() % 11 == 0) ? kNan : std::normal_distribution<double>(0.0, 2.5)(rng);
        }
        X[i * 7 + 5] = (rng() % 9 == 0) ? kNan : static_cast<double>(rng() % 8);
        X[i * 7 + 6] = (rng() % 5 == 0) ? kNan : static_cast<double>(1 + (rng() % 4));
    }

    std::vector<double> p0(n * 3), p1(n * 3), p2(n * 3), p3(n * 3), p4(n * 3);
    knode.predict(X.data(), n, 7, p0.data(), 1);
    binned.predict(X.data(), n, 7, p1.data(), 2);
    compiled.predict(X.data(), n, 7, p2.data(), 2);
    loaded.predict(X.data(), n, 7, p3.data(), 2);

    RowMajorColumnSource col_src;
    col_src.X = X.data();
    col_src.n_features = 7;
    lignum::predict_columns(binned, col_src, n, 7, p4.data(), false, 1, lignum::Strategy::Serial);

    for (size_t idx = 0; idx < n * 3; idx++) {
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p1[idx])) << idx;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p2[idx])) << idx;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p3[idx])) << idx;
        ASSERT_EQ(std::bit_cast<uint64_t>(p0[idx]), std::bit_cast<uint64_t>(p4[idx])) << idx;
    }
    for (size_t i = 0; i < 16; i++) {
        double one0[3]{}, one1[3]{}, one2[3]{};
        knode.predict(&X[i * 7], 1, 7, one0, 1);
        binned.predict(&X[i * 7], 1, 7, one1, 1);
        compiled.predict(&X[i * 7], 1, 7, one2, 1);
        for (size_t k = 0; k < 3; k++) {
            ASSERT_EQ(std::bit_cast<uint64_t>(one0[k]), std::bit_cast<uint64_t>(one1[k])) << i << "," << k;
            ASSERT_EQ(std::bit_cast<uint64_t>(one0[k]), std::bit_cast<uint64_t>(one2[k])) << i << "," << k;
        }
    }
}

} // namespace
