#include "test_util.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using lignum::test::predict_raw;
using lignum::test::write_temp;

TEST(BinaryIO, RoundTripPreservesPredictions) {
    const std::string json = R"({"objective": "binary sigmoid:1", "tree_info": [
        {"tree_structure": {"split_feature": 1, "threshold": 0.25, "decision_type": "<=", "default_left": true,
                            "missing_type": "NaN", "left_child": {"leaf_value": -0.75}, "right_child": {"leaf_value": 0.5}}},
        {"tree_structure": {"leaf_value": 0.125}}]})";
    auto from_json = lignum::load(write_temp("io_src.json", json), "lightgbm");

    auto bin_path = (std::filesystem::temp_directory_path() / "lignum_test_io.bin").string();
    from_json.save(bin_path);
    auto from_bin = lignum::load(bin_path, "auto");

    std::vector<double> X = {0.0, 0.1, 0.0, 0.9, 0.0, 0.25, 0.0, 0.3, 0.0, -1.0};
    EXPECT_EQ(predict_raw(from_json, X, 2), predict_raw(from_bin, X, 2));

    std::vector<double> a(5), b(5);
    from_json.predict(X.data(), 5, 2, a.data(), 1);
    from_bin.predict(X.data(), 5, 2, b.data(), 1);
    EXPECT_EQ(a, b);
}

TEST(BinaryIO, RejectsNonLignumFile) {
    auto path = write_temp("io_garbage.bin", "definitely not a lignum model");
    EXPECT_THROW(lignum::load(path, "auto"), std::invalid_argument);
}

namespace {

    // Numeric, zero-as-missing and categorical (bitset) splits: every section of the format, the
    // binned ones included, is present.
    const char* kSectionsModel = R"({"objective": "regression", "tree_info": [
      {"tree_structure": {"split_feature": 0, "threshold": "1||5||100||900", "decision_type": "==", "default_left": false, "missing_type": "None",
        "left_child": {"split_feature": 1, "threshold": 0.5, "decision_type": "<=", "default_left": true, "missing_type": "Zero",
                       "left_child": {"leaf_value": 0.1}, "right_child": {"leaf_value": 0.2}},
        "right_child": {"leaf_value": 0.3}}}]})";

    std::vector<char> read_bytes(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    template <typename T>
    T get(const std::vector<char>& bytes, size_t pos) {
        T v;
        std::memcpy(&v, bytes.data() + pos, sizeof(T));
        return v;
    }

    template <typename T>
    void put(std::vector<char>& bytes, size_t pos, T v) {
        std::memcpy(bytes.data() + pos, &v, sizeof(T));
    }

    size_t align64(size_t n) { return (n + 63) & ~size_t{63}; }

    // Byte offsets of the format 0.8 (src/io.cpp): FileHeader fields, the BinnedFileHeader and
    // the binned kernel's BinnedCatFeature section.
    struct Layout {
        static constexpr size_t kCatWords = 16, kNodes = 24, kOffsets = 32, kLeaves = 40;
        size_t binned_header = 0;
        size_t cat_features = 0;

        explicit Layout(const std::vector<char>& b) {
            const uint64_t n_cat_words = get<uint64_t>(b, kCatWords), n_nodes = get<uint64_t>(b, kNodes);
            const uint64_t n_offsets = get<uint64_t>(b, kOffsets), n_leaves = get<uint64_t>(b, kLeaves);
            const uint16_t n_outputs = get<uint16_t>(b, 60);
            const uint64_t names_bytes = get<uint64_t>(b, 64 + 24);
            const uint32_t objective_bytes = get<uint32_t>(b, 64 + 40);
            size_t pos = 128 + align64(names_bytes + objective_bytes);
            pos += align64(n_nodes * 64) + align64(n_offsets * 4) + align64(n_leaves * 8) + align64(n_offsets) +
                   align64(n_outputs * 8) + (n_outputs > 1 ? align64(n_offsets * 2) : 0) + align64(n_cat_words * 4);
            binned_header = pos;
            const uint64_t bn = get<uint64_t>(b, pos), bt = get<uint64_t>(b, pos + 8), bf = get<uint64_t>(b, pos + 16);
            const uint64_t bc = get<uint64_t>(b, pos + 24);
            pos += 64;
            pos += align64(bn * 64) + align64(bt * 4) + align64(bt) + align64(bf * 4) + align64(bf) + align64(bf * 2) +
                   align64((bf + 1) * 4) + align64(bc * 8);
            cat_features = pos;
        }
    };

    // Writes `bytes` and expects loading to reject them: with and without the binned kernel, or
    // only with it for damage in the binned sections (binned=false does not read them).
    void expect_rejected(const std::vector<char>& bytes, const std::string& what, bool binned_only = false) {
        const auto path = (std::filesystem::temp_directory_path() / "lignum_test_io_counts.bin").string();
        std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        for (bool binned : {true, false}) {
            if (binned_only && !binned) continue;
            EXPECT_THROW(lignum::load(path, "auto", binned), std::runtime_error) << what << ", binned=" << binned;
        }
    }

} // namespace

TEST(BinaryIO, CountsThatWrapAroundAreRejected) {
    // A count such as n_leaves + 2^61 makes n_leaves * 8 wrap to the original size: every size check
    // passed and every leaf index looked in range, so prediction read outside the mapped file.
    const auto path = (std::filesystem::temp_directory_path() / "lignum_test_io_sections.bin").string();
    lignum::load(write_temp("io_sections.json", kSectionsModel), "lightgbm").save(path);
    const std::vector<char> original = read_bytes(path);
    ASSERT_NO_THROW(lignum::load(path, "auto"));
    const Layout layout(original);
    ASSERT_EQ(get<uint64_t>(original, layout.binned_header + 32), 1u);  // one categorical feature

    std::vector<size_t> counts = {Layout::kCatWords, Layout::kNodes, Layout::kOffsets, Layout::kLeaves};
    for (size_t k = 0; k < 6; k++) counts.push_back(layout.binned_header + 8 * k);  // BinnedFileHeader counts
    for (size_t pos : counts) {
        for (int shift : {58, 59, 60, 61, 62, 63}) {
            std::vector<char> bytes = original;
            put<uint64_t>(bytes, pos, get<uint64_t>(bytes, pos) + (uint64_t{1} << shift));
            expect_rejected(bytes, "count at byte " + std::to_string(pos) + " + 2^" + std::to_string(shift), pos >= layout.binned_header);
        }
    }

    std::vector<char> bytes = original;
    put<uint64_t>(bytes, Layout::kCatWords, 0);  // no word 0 for the branch-free lookups
    expect_rejected(bytes, "n_cat_words = 0");

    bytes = original;
    put<uint64_t>(bytes, layout.cat_features + 16, ~uint64_t{0} - 3);  // table_offset + size wraps around
    expect_rejected(bytes, "categorical table offset near 2^64", true);
}

TEST(BinaryIO, TransformMustMatchTheModel) {
    // Hinge is XGBoost's (float32) and SignedSquare LightGBM's (double): another pairing, or an
    // unknown value, would predict without the transform the library applies.
    const auto path = (std::filesystem::temp_directory_path() / "lignum_test_io_transform.bin").string();
    lignum::load(write_temp("io_transform.json", kSectionsModel), "lightgbm").save(path);
    const std::vector<char> original = read_bytes(path);
    ASSERT_EQ(original[10], static_cast<char>(lignum::transform::Identity));  // FileHeader::transform
    for (uint8_t transform : {lignum::transform::Hinge, static_cast<uint8_t>(lignum::transform::Last + 1), uint8_t{255}}) {
        std::vector<char> bytes = original;
        bytes[10] = static_cast<char>(transform);
        expect_rejected(bytes, "transform " + std::to_string(transform));
    }
    std::vector<char> bytes = original;
    bytes[10] = static_cast<char>(lignum::transform::SignedSquare);
    const auto ok = (std::filesystem::temp_directory_path() / "lignum_test_io_transform_ok.bin").string();
    std::ofstream(ok, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    EXPECT_EQ(lignum::load(ok).transform, lignum::transform::SignedSquare);
}

namespace {

    // One tree whose leftmost path is four splits deep (two KNodes, two BNodes on that path), and a
    // second tree of one split.
    std::string deep_model() {
        auto split = [](int feature, double threshold, const std::string& left, const std::string& right) {
            return R"({"split_feature": )" + std::to_string(feature) + R"(, "threshold": )" + std::to_string(threshold) +
                   R"(, "decision_type": "<=", "default_left": true, "missing_type": "NaN", "left_child": )" + left +
                   R"(, "right_child": )" + right + "}";
        };
        auto leaf = [](double v) { return R"({"leaf_value": )" + std::to_string(v) + "}"; };
        const std::string chain = split(0, 0.5, split(1, 0.25, split(0, -0.5, split(1, -1.0, leaf(1), leaf(2)), leaf(3)), leaf(4)), leaf(5));
        return R"({"objective": "regression", "tree_info": [{"tree_structure": )" + chain + R"(}, {"tree_structure": )" +
               split(1, 0.0, leaf(6), leaf(7)) + "}]}";
    }

} // namespace

TEST(BinaryIO, NodesReferencedTwiceAreRejected) {
    // Writers lay out trees: every node has one parent. A shared node was unfolded once per path,
    // and KNode children {2, 2} (a node twice) were taken for a leaf, which made the binned walk
    // loop forever (found by tests/fuzz).
    const auto path = (std::filesystem::temp_directory_path() / "lignum_test_io_refs.bin").string();
    lignum::load(write_temp("io_refs.json", deep_model()), "lightgbm").save(path);
    const std::vector<char> original = read_bytes(path);
    ASSERT_NO_THROW(lignum::load(path, "auto"));
    const Layout layout(original);
    const uint64_t n_nodes = get<uint64_t>(original, Layout::kNodes);
    ASSERT_EQ(n_nodes, 3u);  // two on the chain, one for the second tree
    const size_t knodes = 128 + align64(get<uint64_t>(original, 64 + 24) + get<uint32_t>(original, 64 + 40));
    const size_t roots = knodes + align64(n_nodes * 64);
    auto knode_child = [&](size_t node, int c) { return knodes + node * 64 + 32 + 4 * static_cast<size_t>(c); };
    ASSERT_EQ(get<int32_t>(original, knode_child(0, 0)), 1);  // the chain continues in KNode 1

    std::vector<char> bytes = original;
    put<int32_t>(bytes, knode_child(0, 1), 1);  // KNode 1 for both grandchildren
    expect_rejected(bytes, "a KNode twice in one node");
    bytes = original;
    put<int32_t>(bytes, knode_child(0, 3), 1);  // KNode 1 under both children of the root
    expect_rejected(bytes, "a KNode in two slots");
    bytes = original;
    put<int32_t>(bytes, roots + 4, 1);  // the second tree's root is KNode 1 as well
    expect_rejected(bytes, "a KNode as a child and a root");

    // The binned nodes: BNode 0 holds three levels of the chain, BNode 1 the fourth.
    const size_t bnodes = layout.binned_header + 64;
    ASSERT_EQ(get<uint64_t>(original, layout.binned_header), 3u);
    size_t to_bnode1 = 8, leaf_slot = 8;
    for (size_t c = 0; c < 8; c++) {
        const int32_t child = get<int32_t>(original, bnodes + 4 * c);
        if (child == 1 && to_bnode1 == 8) to_bnode1 = c;
        if (child < 0) leaf_slot = c;
    }
    ASSERT_LT(to_bnode1, 8u);
    ASSERT_LT(leaf_slot, 8u);
    bytes = original;
    put<int32_t>(bytes, bnodes + 4 * leaf_slot, 1);  // BNode 1 in a second slot of BNode 0
    expect_rejected(bytes, "a BNode in two slots", true);

    // Flag bits lignum does not write.
    bytes = original;
    bytes[11] = static_cast<char>(static_cast<uint8_t>(bytes[11]) | 0x80);
    expect_rejected(bytes, "an unknown header flag");
}
