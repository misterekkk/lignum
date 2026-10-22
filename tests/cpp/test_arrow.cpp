#include "test_util.hpp"

#include <lignum/c_api.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using lignum::test::predict_raw;
using lignum::test::write_temp;

namespace {

    // Splits on feature 0 (numeric, NaN right) and feature 1 (numeric, NaN left).
    const char* kModel = R"({"objective": "regression", "tree_info": [
        {"tree_structure": {"split_feature": 0, "threshold": 0.5, "decision_type": "<=", "default_left": false,
                            "missing_type": "NaN", "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}},
        {"tree_structure": {"split_feature": 1, "threshold": 2.0, "decision_type": "<=", "default_left": true,
                            "missing_type": "NaN", "left_child": {"leaf_value": 10.0}, "right_child": {"leaf_value": 20.0}}}]})";

    const double kNan = std::numeric_limits<double>::quiet_NaN();

    void no_release(ArrowArray*) {}

    // A column of `format` over `data` with an optional validity bitmap.
    struct Col {
        ArrowSchema schema{};
        ArrowArray array{};
        const void* buffers[2] = {nullptr, nullptr};

        Col(const char* format, const void* data, int64_t length, int64_t offset = 0, const uint8_t* validity = nullptr) {
            schema.format = format;
            schema.name = "col";
            buffers[0] = validity;
            buffers[1] = data;
            array.length = length;
            array.offset = offset;
            array.null_count = validity ? -1 : 0;
            array.n_buffers = 2;
            array.buffers = buffers;
            array.release = no_release;
        }
    };

    // A record batch over the columns.
    struct Batch {
        ArrowSchema schema{};
        ArrowArray array{};
        std::vector<ArrowSchema*> schemas;
        std::vector<ArrowArray*> arrays;
        const void* buffers[1] = {nullptr};

        Batch(std::vector<Col*> cols, int64_t length, int64_t offset = 0, const uint8_t* validity = nullptr) {
            for (auto* c : cols) {
                schemas.push_back(&c->schema);
                arrays.push_back(&c->array);
            }
            schema.format = "+s";
            schema.n_children = static_cast<int64_t>(cols.size());
            schema.children = schemas.data();
            buffers[0] = validity;
            array.length = length;
            array.offset = offset;
            array.null_count = validity ? -1 : 0;
            array.n_buffers = 1;
            array.buffers = buffers;
            array.n_children = static_cast<int64_t>(cols.size());
            array.children = arrays.data();
            array.release = no_release;
        }
    };

    std::vector<double> predict(const lignum::Model& model, const Batch& batch) {
        std::vector<double> out(static_cast<size_t>(batch.array.length));
        lignum::predict_arrow(model, batch.schema, batch.array, out.data(), true, 1);
        return out;
    }

} // namespace

TEST(Arrow, OffsetsValidityAndTypesMatchTheMatrix) {
    auto model = lignum::load(write_temp("arrow.json", kModel), "lightgbm");
    // Column 0: float32 with its own offset 1; column 1: int16 with nulls (bit 2 cleared).
    const float f0[] = {99.0f, 0.25f, 0.5f, 0.75f, NAN, 3.0f};
    const int16_t f1[] = {1, 2, 3, 0, 5, -7};
    const uint8_t valid1 = 0b111011;
    Col c0("f", f0, 6, 1), c1("s", f1, 6, 0, &valid1);
    Batch batch({&c0, &c1}, 4, 1);  // rows 1..4 of the batch
    const std::vector<double> X = {
        0.5, 2.0,    // row 1: c0[2], c1[1]
        0.75, kNan,  // row 2: c0[3], c1[2] (null)
        kNan, 0.0,   // row 3: c0[4], c1[3]
        3.0, 5.0,    // row 4: c0[5], c1[4]
    };
    EXPECT_EQ(predict(model, batch), predict_raw(model, X, 2));
}

TEST(Arrow, NullRowsBooleansAndDictionaries) {
    auto model = lignum::load(write_temp("arrow2.json", kModel), "lightgbm");
    const uint8_t bools = 0b0101;               // 1, 0, 1, 0
    const int8_t indices[] = {0, 3, 1, 2};      // dictionary indices: the values are the codes
    const char values[] = "abcd";
    Col c0("b", &bools, 4), c1("c", indices, 4);
    ArrowSchema dict{};
    dict.format = "u";
    c1.schema.dictionary = &dict;
    (void)values;
    const uint8_t rows_valid = 0b1101;          // row 1 of the batch is null: all features NaN
    Batch batch({&c0, &c1}, 4, 0, &rows_valid);
    const std::vector<double> X = {1.0, 0.0, kNan, kNan, 1.0, 1.0, 0.0, 2.0};
    EXPECT_EQ(predict(model, batch), predict_raw(model, X, 2));
}

TEST(Arrow, RejectsUnsupportedInput) {
    auto model = lignum::load(write_temp("arrow3.json", kModel), "lightgbm");
    const double d[] = {1.0};
    Col c0("g", d, 1), c1("u", d, 1), c2("e", d, 1);
    std::vector<double> out(1);
    Batch strings({&c0, &c1}, 1);
    EXPECT_THROW(lignum::predict_arrow(model, strings.schema, strings.array, out.data()), std::invalid_argument);
    Batch half({&c0, &c2}, 1);
    EXPECT_THROW(lignum::predict_arrow(model, half.schema, half.array, out.data()), std::invalid_argument);
    Batch narrow({&c0}, 1);
    EXPECT_THROW(lignum::predict_arrow(model, narrow.schema, narrow.array, out.data()), std::invalid_argument);
    EXPECT_THROW(lignum::predict_arrow(model, c0.schema, c0.array, out.data()), std::invalid_argument);  // not a struct
}

TEST(Arrow, StreamAndCApi) {
    auto model = lignum::load(write_temp("arrow4.json", kModel), "lightgbm");
    const double f0[] = {0.25, 1.0, kNan, 0.5};
    const double f1[] = {3.0, kNan, 2.0, 1.0};
    Col c0("g", f0, 4), c1("g", f1, 4);
    Batch batch({&c0, &c1}, 4);
    const std::vector<double> X = {0.25, 3.0, 1.0, kNan, kNan, 2.0, 0.5, 1.0};
    const auto expected = predict_raw(model, X, 2);

    // A stream of two batches: rows 0..1, then 2..3 (offset 2).
    struct State {
        Batch* batch;
        int next = 0;
    } state{&batch};
    ArrowArrayStream stream{};
    stream.private_data = &state;
    stream.get_schema = [](ArrowArrayStream* s, ArrowSchema* out) {
        *out = static_cast<State*>(s->private_data)->batch->schema;
        out->release = nullptr;
        return 0;
    };
    stream.get_next = [](ArrowArrayStream* s, ArrowArray* out) {
        auto* st = static_cast<State*>(s->private_data);
        if (st->next == 2) {
            out->release = nullptr;
            return 0;
        }
        *out = st->batch->array;
        out->offset = st->next * 2;
        out->length = 2;
        st->next++;
        return 0;
    };
    size_t n_rows = 0;
    EXPECT_EQ(lignum::predict_arrow_stream(model, stream, true, 1, lignum::Strategy::Auto, &n_rows), expected);
    EXPECT_EQ(n_rows, 4u);

    LignumModelHandle handle = nullptr;
    ASSERT_EQ(LignumModelLoad(write_temp("arrow5.json", kModel).c_str(), "lightgbm", &handle), 0);
    std::vector<double> out(4);
    ASSERT_EQ(LignumModelPredictArrow(handle, &batch.schema, &batch.array, 1, 1, LIGNUM_STRATEGY_AUTO, out.data()), 0)
        << LignumGetLastError();
    EXPECT_EQ(out, expected);
    Col bad("u", f0, 4);
    Batch strings({&c0, &bad}, 4);
    EXPECT_EQ(LignumModelPredictArrow(handle, &strings.schema, &strings.array, 1, 1, LIGNUM_STRATEGY_AUTO, out.data()), -1);
    EXPECT_NE(std::string(LignumGetLastError()).find("Unsupported Arrow type"), std::string::npos);
    EXPECT_EQ(LignumModelFree(handle), 0);
}
