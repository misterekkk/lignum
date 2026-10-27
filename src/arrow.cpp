#include <lignum/lignum.hpp>
#include "catboost.hpp"
#include "columns.hpp"
#include "validate.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace lignum {

    namespace {

        // Rows a thread converts and predicts at once: the converted block is still in cache when
        // the trees read it. Large enough for the compiled mode's row chunks.
        constexpr int64_t kMaxBlockRows = 16384;
        constexpr int64_t kMinBlockRows = 256;
        // Bytes of converted rows written column by column before moving on (fits L1).
        constexpr size_t kTileBytes = 24 * 1024;

        enum class Kind { Null, Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float32, Float64 };

        // One feature column of a record batch. Element i of the batch is element `offset + i` of
        // the column's buffers (the batch's offset plus the column's own).
        struct Column {
            Kind kind = Kind::Null;
            const uint8_t* validity = nullptr;  // null: no nulls
            const void* data = nullptr;
            int64_t offset = 0;

            bool valid(int64_t k) const { return !validity || ((validity[k >> 3] >> (k & 7)) & 1); }

            // The value as numpy's astype(float64) gives it (integers round to nearest); null is NaN.
            double value(int64_t i) const {
                const int64_t k = offset + i;
                if (kind == Kind::Null || !valid(k)) return std::numeric_limits<double>::quiet_NaN();
                switch (kind) {
                    case Kind::Bool:    return (static_cast<const uint8_t*>(data)[k >> 3] >> (k & 7)) & 1;
                    case Kind::Int8:    return static_cast<double>(static_cast<const int8_t*>(data)[k]);
                    case Kind::UInt8:   return static_cast<double>(static_cast<const uint8_t*>(data)[k]);
                    case Kind::Int16:   return static_cast<double>(static_cast<const int16_t*>(data)[k]);
                    case Kind::UInt16:  return static_cast<double>(static_cast<const uint16_t*>(data)[k]);
                    case Kind::Int32:   return static_cast<double>(static_cast<const int32_t*>(data)[k]);
                    case Kind::UInt32:  return static_cast<double>(static_cast<const uint32_t*>(data)[k]);
                    case Kind::Int64:   return static_cast<double>(static_cast<const int64_t*>(data)[k]);
                    case Kind::UInt64:  return static_cast<double>(static_cast<const uint64_t*>(data)[k]);
                    case Kind::Float32: return static_cast<double>(static_cast<const float*>(data)[k]);
                    case Kind::Float64: return static_cast<const double*>(data)[k];
                    default:            return std::numeric_limits<double>::quiet_NaN();
                }
            }
        };

        template <typename T>
        void fill_typed(const Column& c, int64_t begin, int64_t count, double* dst, size_t stride) {
            const T* data = static_cast<const T*>(c.data) + c.offset + begin;
            if (!c.validity) {
                for (int64_t i = 0; i < count; i++) dst[static_cast<size_t>(i) * stride] = static_cast<double>(data[i]);
                return;
            }
            for (int64_t i = 0; i < count; i++) {
                const int64_t k = c.offset + begin + i;
                dst[static_cast<size_t>(i) * stride] = c.valid(k) ? static_cast<double>(data[i]) : std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Rows [begin, begin + count) of column c into dst[i * stride]: one type dispatch per
        // column and block instead of per value.
        void fill(const Column& c, int64_t begin, int64_t count, double* dst, size_t stride) {
            switch (c.kind) {
                case Kind::Int8:    return fill_typed<int8_t>(c, begin, count, dst, stride);
                case Kind::UInt8:   return fill_typed<uint8_t>(c, begin, count, dst, stride);
                case Kind::Int16:   return fill_typed<int16_t>(c, begin, count, dst, stride);
                case Kind::UInt16:  return fill_typed<uint16_t>(c, begin, count, dst, stride);
                case Kind::Int32:   return fill_typed<int32_t>(c, begin, count, dst, stride);
                case Kind::UInt32:  return fill_typed<uint32_t>(c, begin, count, dst, stride);
                case Kind::Int64:   return fill_typed<int64_t>(c, begin, count, dst, stride);
                case Kind::UInt64:  return fill_typed<uint64_t>(c, begin, count, dst, stride);
                case Kind::Float32: return fill_typed<float>(c, begin, count, dst, stride);
                case Kind::Float64: return fill_typed<double>(c, begin, count, dst, stride);
                default:
                    for (int64_t i = 0; i < count; i++) dst[static_cast<size_t>(i) * stride] = c.value(begin + i);
            }
        }

        std::string column_name(const ArrowSchema& schema, size_t j) {
            const char* name = schema.name;
            return "column " + std::to_string(j) + (name && *name ? " ('" + std::string(name) + "')" : std::string());
        }

        bool kind_of(const char* format, Kind& kind) {
            if (!format || !format[0] || format[1]) return false;
            switch (format[0]) {
                case 'n': kind = Kind::Null; return true;
                case 'b': kind = Kind::Bool; return true;
                case 'c': kind = Kind::Int8; return true;
                case 'C': kind = Kind::UInt8; return true;
                case 's': kind = Kind::Int16; return true;
                case 'S': kind = Kind::UInt16; return true;
                case 'i': kind = Kind::Int32; return true;
                case 'I': kind = Kind::UInt32; return true;
                case 'l': kind = Kind::Int64; return true;
                case 'L': kind = Kind::UInt64; return true;
                case 'f': kind = Kind::Float32; return true;
                case 'g': kind = Kind::Float64; return true;
                default:  return false;
            }
        }

        // Dictionary-encoded columns give their indices: the category codes LightGBM and XGBoost
        // see for pandas categoricals.
        Column column(const ArrowSchema& schema, const ArrowArray& array, int64_t batch_offset, size_t j) {
            Column c;
            if (!kind_of(schema.format, c.kind) ||
                (schema.dictionary && (c.kind == Kind::Null || c.kind == Kind::Bool || c.kind == Kind::Float32 || c.kind == Kind::Float64))) {
                throw std::invalid_argument("Unsupported Arrow type '" + std::string(schema.format ? schema.format : "") + "' in " +
                                            column_name(schema, j) + "; expected a numeric, boolean or dictionary column.");
            }
            c.offset = batch_offset + array.offset;
            if (c.kind == Kind::Null) return c;
            if (array.n_buffers < 2) throw std::invalid_argument("Malformed Arrow array in " + column_name(schema, j) + ".");
            c.validity = array.null_count == 0 ? nullptr : static_cast<const uint8_t*>(array.buffers[0]);
            c.data = array.buffers[1];
            if (!c.data && array.length > 0) throw std::invalid_argument("Missing data buffer in " + column_name(schema, j) + ".");
            return c;
        }

        // Strings of an Arrow utf8 ('u'), large_utf8 ('U') or utf8_view ('vu') array, or binary
        // ('z', 'Z', 'vz') bytes, read in place.
        struct Strings {
            char kind = 0;  // 'u': int32 offsets, 'U': int64 offsets, 'v': 16-byte views
            const uint8_t* validity = nullptr;
            const int32_t* offsets32 = nullptr;
            const int64_t* offsets64 = nullptr;
            const char* data = nullptr;
            const uint8_t* views = nullptr;
            const ArrowArray* array = nullptr;
            int64_t offset = 0;

            bool valid(int64_t i) const {
                const int64_t k = offset + i;
                return !validity || ((validity[k >> 3] >> (k & 7)) & 1);
            }

            std::string_view at(int64_t i) const {
                const int64_t k = offset + i;
                if (kind == 'u') return {data + offsets32[k], static_cast<size_t>(offsets32[k + 1] - offsets32[k])};
                if (kind == 'U') return {data + offsets64[k], static_cast<size_t>(offsets64[k + 1] - offsets64[k])};
                // View: int32 length; up to 12 bytes inline, else prefix, buffer index, offset.
                const uint8_t* view = views + 16 * k;
                int32_t length, buffer, start;
                std::memcpy(&length, view, 4);
                if (length <= 12) return {reinterpret_cast<const char*>(view + 4), static_cast<size_t>(length)};
                std::memcpy(&buffer, view + 8, 4);
                std::memcpy(&start, view + 12, 4);
                if (buffer < 0 || 2 + buffer >= array->n_buffers - 1) throw std::invalid_argument("Malformed Arrow string view.");
                return {static_cast<const char*>(array->buffers[2 + buffer]) + start, static_cast<size_t>(length)};
            }
        };

        bool strings_of(const char* format, const ArrowArray& array, int64_t batch_offset, Strings& s) {
            if (!format) return false;
            const std::string_view f(format);
            if (f == "u" || f == "z") s.kind = 'u';
            else if (f == "U" || f == "Z") s.kind = 'U';
            else if (f == "vu" || f == "vz") s.kind = 'v';
            else return false;
            if (array.n_buffers < (s.kind == 'v' ? 2 : 3)) throw std::invalid_argument("Malformed Arrow string array.");
            s.offset = batch_offset + array.offset;
            s.validity = array.null_count == 0 ? nullptr : static_cast<const uint8_t*>(array.buffers[0]);
            s.array = &array;
            if (s.kind == 'v') {
                s.views = static_cast<const uint8_t*>(array.buffers[1]);
            } else {
                if (s.kind == 'u') s.offsets32 = static_cast<const int32_t*>(array.buffers[1]);
                else               s.offsets64 = static_cast<const int64_t*>(array.buffers[1]);
                s.data = static_cast<const char*>(array.buffers[2]);
            }
            return true;
        }

        // CatBoost hashes integer categories as their decimal text.
        template <typename T>
        uint32_t integer_hash(T value) {
            char text[32];
            const auto [end, ec] = std::to_chars(text, text + sizeof(text), value);
            (void)ec;
            return catboost_hash(std::string_view(text, static_cast<size_t>(end - text)));
        }

        bool is_integer(Kind k) {
            return k == Kind::Int8 || k == Kind::UInt8 || k == Kind::Int16 || k == Kind::UInt16 || k == Kind::Int32 ||
                   k == Kind::UInt32 || k == Kind::Int64 || k == Kind::UInt64;
        }

        // Hash of element i of an integer column (exact for 64-bit values, unlike Column::value).
        uint32_t integer_hash_at(const Column& c, int64_t i) {
            const int64_t k = c.offset + i;
            switch (c.kind) {
                case Kind::Int8:   return integer_hash(static_cast<int>(static_cast<const int8_t*>(c.data)[k]));
                case Kind::UInt8:  return integer_hash(static_cast<unsigned>(static_cast<const uint8_t*>(c.data)[k]));
                case Kind::Int16:  return integer_hash(static_cast<int>(static_cast<const int16_t*>(c.data)[k]));
                case Kind::UInt16: return integer_hash(static_cast<unsigned>(static_cast<const uint16_t*>(c.data)[k]));
                case Kind::Int32:  return integer_hash(static_cast<const int32_t*>(c.data)[k]);
                case Kind::UInt32: return integer_hash(static_cast<const uint32_t*>(c.data)[k]);
                case Kind::Int64:  return integer_hash(static_cast<const int64_t*>(c.data)[k]);
                case Kind::UInt64: return integer_hash(static_cast<const uint64_t*>(c.data)[k]);
                default:           return 0;
            }
        }

        // A categorical column of a CatBoost model: strings, integers, or a dictionary of either
        // (hashed once per dictionary value). Values become CatBoost's category hashes.
        struct CatColumn {
            Strings strings;               // kind != 0: a string column
            Column integers;               // otherwise: integers, or the indices of a dictionary
            std::vector<double> dictionary;  // hashes of the dictionary values (dictionary columns)
            std::string name;

            [[noreturn]] void missing() const {
                throw std::invalid_argument("Categorical " + name + " has nulls; CatBoost needs a category in every row.");
            }

            double hash(int64_t i) const {
                if (strings.kind) {
                    if (!strings.valid(i)) missing();
                    return static_cast<double>(catboost_hash(strings.at(i)));
                }
                if (!integers.valid(integers.offset + i)) missing();
                if (dictionary.empty()) return static_cast<double>(integer_hash_at(integers, i));
                const double index = integers.value(i);
                if (!(index >= 0 && index < static_cast<double>(dictionary.size()))) {
                    throw std::invalid_argument("Dictionary index out of range in " + name + ".");
                }
                const double h = dictionary[static_cast<size_t>(index)];
                if (h < 0) missing();  // a null dictionary value
                return h;
            }
        };

        CatColumn cat_column(const ArrowSchema& schema, const ArrowArray& array, int64_t batch_offset, size_t j) {
            CatColumn c;
            c.name = column_name(schema, j);
            const auto unsupported = [&] {
                return std::invalid_argument("Categorical " + c.name + " must hold strings, integers or a dictionary of them (got Arrow type '" +
                                             std::string(schema.format ? schema.format : "") + "').");
            };
            if (schema.dictionary) {
                if (!array.dictionary) throw std::invalid_argument("Missing Arrow dictionary in " + c.name + ".");
                if (!kind_of(schema.format, c.integers.kind) || !is_integer(c.integers.kind)) throw unsupported();
                if (array.n_buffers < 2) throw std::invalid_argument("Malformed Arrow array in " + c.name + ".");
                c.integers.offset = batch_offset + array.offset;
                c.integers.validity = array.null_count == 0 ? nullptr : static_cast<const uint8_t*>(array.buffers[0]);
                c.integers.data = array.buffers[1];
                const ArrowArray& dict = *array.dictionary;
                Strings values;
                Column ints;
                c.dictionary.resize(static_cast<size_t>(std::max<int64_t>(dict.length, 0)));
                if (strings_of(schema.dictionary->format, dict, 0, values)) {
                    for (int64_t k = 0; k < dict.length; k++) {
                        c.dictionary[static_cast<size_t>(k)] = values.valid(k) ? static_cast<double>(catboost_hash(values.at(k))) : -1.0;
                    }
                } else if (kind_of(schema.dictionary->format, ints.kind) && is_integer(ints.kind) && dict.n_buffers >= 2) {
                    ints.offset = dict.offset;
                    ints.validity = dict.null_count == 0 ? nullptr : static_cast<const uint8_t*>(dict.buffers[0]);
                    ints.data = dict.buffers[1];
                    for (int64_t k = 0; k < dict.length; k++) c.dictionary[static_cast<size_t>(k)] = static_cast<double>(integer_hash_at(ints, k));
                } else {
                    throw unsupported();
                }
                if (c.dictionary.empty()) c.dictionary.push_back(-1.0);  // keeps the dictionary path for an empty dictionary
                return c;
            }
            if (strings_of(schema.format, array, batch_offset, c.strings)) return c;
            if (!kind_of(schema.format, c.integers.kind) || !is_integer(c.integers.kind)) throw unsupported();
            c.integers = column(schema, array, batch_offset, j);
            return c;
        }

        int threads_for(int n_jobs) {
        #ifdef _OPENMP
            return n_jobs > 0 ? n_jobs : omp_get_max_threads();
        #else
            (void)n_jobs;
            return 1;
        #endif
        }

        // The batch's feature columns as a ColumnSource for the binned kernel.
        class ArrowColumns final : public ColumnSource {
            public:
                ArrowColumns(const std::vector<Column>& columns, const Column& rows) : columns_(columns), rows_(rows) {}

                void read(size_t column, size_t begin, size_t count, double* dst) const override {
                    fill(columns_[column], static_cast<int64_t>(begin), static_cast<int64_t>(count), dst, 1);
                    if (!rows_.validity) return;
                    for (size_t i = 0; i < count; i++) {
                        if (!rows_.valid(rows_.offset + static_cast<int64_t>(begin + i))) dst[i] = std::numeric_limits<double>::quiet_NaN();
                    }
                }

            private:
                const std::vector<Column>& columns_;
                const Column& rows_;
        };

    } // namespace

    void predict_arrow(const Model& model, const ArrowSchema& schema, const ArrowArray& batch, double* out, bool raw_score,
                       int n_jobs, Strategy strategy, bool validate_features) {
        if (!schema.format || std::strcmp(schema.format, "+s") != 0) {
            throw std::invalid_argument("Arrow input must be a record batch (struct array), got format '" +
                                        std::string(schema.format ? schema.format : "") + "'.");
        }
        if (validate_features && !model.feature_names.empty()) {
            auto name_of = [&](size_t j) {
                const char* name = schema.children && schema.children[j] ? schema.children[j]->name : nullptr;
                return name ? name : "";
            };
            const auto& expected = model.feature_names;
            bool same = expected.size() <= static_cast<size_t>(std::max<int64_t>(schema.n_children, 0));
            for (size_t j = 0; same && j < expected.size(); j++) same = expected[j] == name_of(j);
            if (!same) {  // a mismatch, or LightGBM names with spaces: build the list to decide and explain
                std::vector<std::string> names;
                for (int64_t j = 0; j < schema.n_children; j++) names.emplace_back(name_of(static_cast<size_t>(j)));
                check_feature_names(model, names);
            }
        }
        if (schema.n_children != batch.n_children) throw std::invalid_argument("Arrow schema and array disagree on the number of columns.");
        const size_t nf = model.n_features;
        if (static_cast<size_t>(batch.n_children) < nf) {
            throw std::invalid_argument("Input has " + std::to_string(batch.n_children) + " columns, but the model requires at least " +
                                        std::to_string(nf) + ".");
        }
        const int64_t n = batch.length;
        if (n <= 0) return;

        // Features are the first n_features columns, by position (like the columns of a matrix).
        // Categorical columns of CatBoost models become category hashes.
        std::vector<char> is_cat(nf, 0);
        if (model.oblivious) for (uint32_t c : model.oblivious->cat_columns) is_cat[c] = 1;
        std::vector<Column> columns(nf);
        std::vector<CatColumn> cats(nf);
        for (size_t j = 0; j < nf; j++) {
            if (!schema.children[j] || !batch.children[j]) throw std::invalid_argument("Missing Arrow column " + std::to_string(j) + ".");
            if (batch.children[j]->length < batch.offset + n) {
                throw std::invalid_argument("Arrow " + column_name(*schema.children[j], j) + " is shorter than the batch.");
            }
            if (is_cat[j]) cats[j] = cat_column(*schema.children[j], *batch.children[j], batch.offset, j);
            else           columns[j] = column(*schema.children[j], *batch.children[j], batch.offset, j);
        }
        // A null row of the batch itself: all its features are missing.
        Column rows;
        rows.kind = Kind::Float64;
        rows.offset = batch.offset;
        rows.validity = (batch.null_count != 0 && batch.n_buffers > 0) ? static_cast<const uint8_t*>(batch.buffers[0]) : nullptr;

        // Binned kernel: bins straight from the columns, no row-major copy.
        if (model.binned) {
            const ArrowColumns source(columns, rows);
            predict_columns(model, source, static_cast<size_t>(n), nf, out, raw_score, n_jobs, strategy);
            return;
        }

        // Direct traversal: rows are converted to a row-major matrix block by block.
        const size_t width = model.output_width(raw_score);
        const int64_t tile_rows = std::max<int64_t>(8, static_cast<int64_t>(kTileBytes / (nf * sizeof(double))));
        // Converts rows [start, start + count) and predicts them with `jobs` threads.
        auto block = [&](int64_t start, int64_t count, std::vector<double>& X, int jobs, Strategy s) {
            X.resize(static_cast<size_t>(count) * nf);
            // Tiles of rows that fit in L1, so the strided column writes stay in cache.
            for (int64_t t = 0; t < count; t += tile_rows) {
                const int64_t rows_in_tile = std::min(tile_rows, count - t);
                double* tile = X.data() + static_cast<size_t>(t) * nf;
                for (size_t j = 0; j < nf; j++) {
                    if (!is_cat[j]) {
                        fill(columns[j], start + t, rows_in_tile, tile + j, nf);
                        continue;
                    }
                    for (int64_t i = 0; i < rows_in_tile; i++) tile[static_cast<size_t>(i) * nf + j] = cats[j].hash(start + t + i);
                }
            }
            if (rows.validity) {
                for (int64_t i = 0; i < count; i++) {
                    if (!rows.valid(rows.offset + start + i)) {
                        std::fill_n(X.data() + static_cast<size_t>(i) * nf, nf, std::numeric_limits<double>::quiet_NaN());
                    }
                }
            }
            double* dst = out + static_cast<size_t>(start) * width;
            if (raw_score) model.predict_raw(X.data(), static_cast<size_t>(count), nf, dst, jobs, s);
            else           model.predict(X.data(), static_cast<size_t>(count), nf, dst, jobs, s);
        };

        const int threads = threads_for(n_jobs);
        if (threads <= 1 || strategy == Strategy::Serial || n < 2 * kMinBlockRows) {
            // One thread converts; predict keeps the caller's threads and strategy.
            std::vector<double> X;
            for (int64_t start = 0; start < n; start += kMaxBlockRows) block(start, std::min(kMaxBlockRows, n - start), X, n_jobs, strategy);
            return;
        }
        // Each thread converts a block and predicts it right away on its own thread (strategies
        // give identical results, so this matches predict on the whole matrix).
        const int64_t per_thread = (n + 2 * threads - 1) / (2 * threads);
        const int64_t rows_per_block = std::clamp<int64_t>((per_thread + 3) / 4 * 4, kMinBlockRows, kMaxBlockRows);
        const int64_t n_blocks = (n + rows_per_block - 1) / rows_per_block;
        std::exception_ptr failure;
        #pragma omp parallel num_threads(threads)
        {
            std::vector<double> X;
            #pragma omp for schedule(dynamic)
            for (int64_t b = 0; b < n_blocks; b++) {
                try {
                    const int64_t start = b * rows_per_block;
                    block(start, std::min(rows_per_block, n - start), X, 1, Strategy::Serial);
                } catch (...) {
                    #pragma omp critical
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }

    std::vector<double> predict_arrow_stream(const Model& model, ArrowArrayStream& stream, bool raw_score, int n_jobs,
                                             Strategy strategy, size_t* n_rows, bool validate_features) {
        auto error = [&](const char* what) {
            const char* detail = stream.get_last_error ? stream.get_last_error(&stream) : nullptr;
            return std::runtime_error(std::string(what) + (detail ? std::string(": ") + detail : std::string()));
        };
        ArrowSchema schema{};
        if (stream.get_schema(&stream, &schema) != 0) throw error("Reading the Arrow stream's schema failed");
        struct SchemaGuard {
            ArrowSchema& s;
            ~SchemaGuard() { if (s.release) s.release(&s); }
        } schema_guard{schema};

        std::vector<double> out;
        size_t total = 0;
        const size_t width = model.output_width(raw_score);
        while (true) {
            ArrowArray batch{};
            if (stream.get_next(&stream, &batch) != 0) throw error("Reading the Arrow stream failed");
            if (!batch.release) break;  // end of stream
            struct ArrayGuard {
                ArrowArray& a;
                ~ArrayGuard() { if (a.release) a.release(&a); }
            } array_guard{batch};
            out.resize((total + static_cast<size_t>(batch.length)) * width);
            predict_arrow(model, schema, batch, out.data() + total * width, raw_score, n_jobs, strategy, validate_features);
            validate_features = false;  // one schema for the whole stream
            total += static_cast<size_t>(batch.length);
        }
        if (n_rows) *n_rows = total;
        return out;
    }

} // namespace lignum
