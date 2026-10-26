// CatBoost's C API (libcatboostmodel, loaded with dlopen) against lignum's C API on the same model
// and rows. Built and run by benchmarks/bench_catboost.py, which writes the data file:
//
//   "LCB1", u64 rows, u32 columns, u32 floats, u32 cats, u32 dim,
//   u32 float columns[floats], u32 cat columns[cats],
//   f64 values[rows x floats] (row-major), u64 offsets[rows x cats + 1], UTF-8 bytes,
//   f64 expected raw scores[rows x dim] (CatBoost's Python package).
//
// Usage: catboost_capi <model.cbm> <data> <libcatboostmodel> <threads> <repeats> [case]
// Prints one JSON object per line: {"case": ..., "best_s"/"median_us": ..., "mismatch": ...}.
// With a case, runs only that work once on one thread, for profilers (cachegrind): none (loading
// and setup only), catboost-batch, lignum-batch (all rows from strings), catboost-row, lignum-row
// (every row as its own call).
// POSIX only (dlopen); it is a benchmark tool, not part of the library.

#include <lignum/c_api.h>

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

    struct CatBoostApi {
        void* (*create)();
        void (*destroy)(void*);
        const char* (*error)();
        bool (*load)(void*, const char*);
        bool (*set_type)(void*, const char*);
        bool (*predict)(void*, size_t, const float**, size_t, const char***, size_t, double*, size_t);
        bool (*predict_hashed)(void*, size_t, const float**, size_t, const int**, size_t, double*, size_t);
        bool (*predict_single)(void*, const float*, size_t, const char**, size_t, double*, size_t);
        int (*hash)(const char*, size_t);
        size_t (*float_count)(void*);
        size_t (*cat_count)(void*);
        size_t (*dim)(void*);
    };

    [[noreturn]] void die(const std::string& what) {
        std::fprintf(stderr, "catboost_capi: %s\n", what.c_str());
        std::exit(1);
    }

    template <typename T>
    void bind(void* lib, const char* name, T& fn) {
        fn = reinterpret_cast<T>(dlsym(lib, name));
        if (!fn) die(std::string("missing symbol ") + name);
    }

    CatBoostApi open_catboost(const char* path) {
        void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!lib) die(dlerror());
        CatBoostApi api{};
        bind(lib, "ModelCalcerCreate", api.create);
        bind(lib, "ModelCalcerDelete", api.destroy);
        bind(lib, "GetErrorString", api.error);
        bind(lib, "LoadFullModelFromFile", api.load);
        bind(lib, "SetPredictionTypeString", api.set_type);
        bind(lib, "CalcModelPrediction", api.predict);
        bind(lib, "CalcModelPredictionWithHashedCatFeatures", api.predict_hashed);
        bind(lib, "CalcModelPredictionSingle", api.predict_single);
        bind(lib, "GetStringCatFeatureHash", api.hash);
        bind(lib, "GetFloatFeaturesCount", api.float_count);
        bind(lib, "GetCatFeaturesCount", api.cat_count);
        bind(lib, "GetDimensionsCount", api.dim);
        return api;
    }

    struct Data {
        size_t rows = 0;
        uint32_t columns = 0, n_float = 0, n_cat = 0, dim = 0;
        std::vector<uint32_t> float_columns, cat_columns;
        std::vector<double> floats;
        std::vector<uint64_t> offsets;
        std::string blob;
        std::vector<double> expected;

        const char* cat(size_t row, size_t k) const { return blob.data() + offsets[row * n_cat + k]; }
        size_t cat_size(size_t row, size_t k) const { return offsets[row * n_cat + k + 1] - offsets[row * n_cat + k]; }
    };

    template <typename T>
    void read(std::ifstream& in, T* out, size_t n) {
        in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n * sizeof(T)));
        if (!in) die("data file is truncated");
    }

    Data read_data(const char* path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) die(std::string("cannot open ") + path);
        char magic[4];
        read(in, magic, 4);
        if (std::memcmp(magic, "LCB1", 4) != 0) die("not a catboost_capi data file");
        Data d;
        uint64_t rows = 0;
        read(in, &rows, 1);
        d.rows = rows;
        read(in, &d.columns, 1);
        read(in, &d.n_float, 1);
        read(in, &d.n_cat, 1);
        read(in, &d.dim, 1);
        d.float_columns.resize(d.n_float);
        d.cat_columns.resize(d.n_cat);
        read(in, d.float_columns.data(), d.n_float);
        read(in, d.cat_columns.data(), d.n_cat);
        d.floats.resize(d.rows * d.n_float);
        read(in, d.floats.data(), d.floats.size());
        d.offsets.resize(d.rows * d.n_cat + 1);
        read(in, d.offsets.data(), d.offsets.size());
        d.blob.resize(d.offsets.back());
        read(in, d.blob.data(), d.blob.size());
        d.expected.resize(d.rows * d.dim);
        read(in, d.expected.data(), d.expected.size());
        return d;
    }

    double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Best of `repeats` runs, in seconds.
    double best_of(int repeats, const std::function<void()>& run) {
        run();
        double best = 1e300;
        for (int i = 0; i < repeats; i++) {
            const double t0 = now();
            run();
            best = std::min(best, now() - t0);
        }
        return best;
    }

    // Median per call in µs, calls cycle through `count` start rows.
    double median_us(size_t count, const std::function<void(size_t)>& run) {
        for (size_t i = 0; i < std::min<size_t>(count, 200); i++) run(i);
        std::vector<double> times;
        const double deadline = now() + 1.0;
        for (size_t i = 0; times.size() < 2000 || (now() < deadline && times.size() < 50000); i++) {
            const double t0 = now();
            run(i % count);
            times.push_back(now() - t0);
        }
        std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
        return times[times.size() / 2] * 1e6;
    }

    // Rows [0, rows) split into `threads` contiguous chunks, one std::thread each.
    void in_chunks(size_t rows, int threads, const std::function<void(size_t, size_t)>& run) {
        std::vector<std::thread> pool;
        const size_t chunk = (rows + threads - 1) / threads;
        for (size_t begin = 0; begin < rows; begin += chunk) pool.emplace_back(run, begin, std::min(rows, begin + chunk));
        for (auto& t : pool) t.join();
    }

    size_t mismatches(const std::vector<double>& got, const Data& d, size_t rows) {
        size_t n = 0;
        for (size_t r = 0; r < rows; r++) {
            for (size_t k = 0; k < d.dim; k++) {
                if (std::memcmp(&got[r * d.dim + k], &d.expected[r * d.dim + k], sizeof(double)) != 0) { n++; break; }
            }
        }
        return n;
    }

    void report_batch(const char* name, double best, size_t mismatch) {
        std::printf("{\"case\": \"%s\", \"best_s\": %.6f, \"mismatch\": %zu}\n", name, best, mismatch);
        std::fflush(stdout);
    }

    void report_latency(const char* name, double median, size_t mismatch) {
        std::printf("{\"case\": \"%s\", \"median_us\": %.3f, \"mismatch\": %zu}\n", name, median, mismatch);
        std::fflush(stdout);
    }

} // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) die("usage: catboost_capi <model.cbm> <data> <libcatboostmodel> <threads> <repeats> [case]");
    const Data d = read_data(argv[2]);
    const CatBoostApi cb = open_catboost(argv[3]);
    const int threads = std::max(1, std::atoi(argv[4]));
    const int repeats = std::max(1, std::atoi(argv[5]));
    const size_t rows = d.rows, dim = d.dim;

    void* calcer = cb.create();
    if (!cb.load(calcer, argv[1]) || !cb.set_type(calcer, "RawFormulaVal")) die(cb.error());
    if (cb.float_count(calcer) != d.n_float || cb.cat_count(calcer) != d.n_cat || cb.dim(calcer) != dim) {
        die("feature counts of the model and the data differ");
    }
    LignumModelHandle model = nullptr;
    if (LignumModelLoad(argv[1], "auto", &model) != 0) die(LignumGetLastError());

    // CatBoost's input: float32 rows and per-row arrays of category pointers (or hashes), built
    // once outside the timings, as an application holding its data in that shape would.
    std::vector<float> floats32(d.floats.begin(), d.floats.end());
    std::vector<const float*> float_rows(rows);
    std::vector<std::string> cat_text(rows * d.n_cat);
    std::vector<const char*> cat_ptrs(rows * d.n_cat);
    std::vector<const char**> cat_rows(rows);
    std::vector<int> hashes(rows * d.n_cat);
    std::vector<const int*> hash_rows(rows);
    for (size_t r = 0; r < rows; r++) {
        float_rows[r] = floats32.data() + r * d.n_float;
        for (size_t k = 0; k < d.n_cat; k++) {
            cat_text[r * d.n_cat + k].assign(d.cat(r, k), d.cat_size(r, k));
            cat_ptrs[r * d.n_cat + k] = cat_text[r * d.n_cat + k].c_str();
            hashes[r * d.n_cat + k] = cb.hash(d.cat(r, k), d.cat_size(r, k));
        }
        cat_rows[r] = cat_ptrs.data() + r * d.n_cat;
        hash_rows[r] = hashes.data() + r * d.n_cat;
    }
    // lignum's input: the float64 matrix with category hashes in the categorical columns.
    auto fill_floats = [&](std::vector<double>& X, size_t begin, size_t end) {
        for (size_t r = begin; r < end; r++) {
            for (size_t k = 0; k < d.n_float; k++) X[r * d.columns + d.float_columns[k]] = d.floats[r * d.n_float + k];
        }
    };
    auto hash_rows_into = [&](std::vector<double>& X, size_t begin, size_t end) {
        for (size_t r = begin; r < end; r++) {
            for (size_t k = 0; k < d.n_cat; k++) X[r * d.columns + d.cat_columns[k]] = LignumCatBoostHash(d.cat(r, k), d.cat_size(r, k));
        }
    };
    std::vector<double> X(rows * d.columns), X_hashed(rows * d.columns);
    fill_floats(X, 0, rows);
    fill_floats(X_hashed, 0, rows);
    hash_rows_into(X_hashed, 0, rows);

    std::vector<double> out(rows * dim);
    auto cb_strings = [&](size_t begin, size_t end) {
        if (!cb.predict(calcer, end - begin, float_rows.data() + begin, d.n_float, cat_rows.data() + begin, d.n_cat,
                        out.data() + begin * dim, (end - begin) * dim)) die(cb.error());
    };
    auto cb_hashed = [&](size_t begin, size_t end) {
        if (!cb.predict_hashed(calcer, end - begin, float_rows.data() + begin, d.n_float, hash_rows.data() + begin, d.n_cat,
                               out.data() + begin * dim, (end - begin) * dim)) die(cb.error());
    };
    // lignum from category strings: hashing is part of the call, as CatBoost's string path.
    auto lg_strings = [&](size_t begin, size_t end, int n_jobs) {
        hash_rows_into(X, begin, end);
        if (LignumModelPredict(model, X.data() + begin * d.columns, end - begin, d.columns, 1, n_jobs, LIGNUM_STRATEGY_AUTO,
                               out.data() + begin * dim) != 0) die(LignumGetLastError());
    };
    auto lg_hashed = [&](size_t begin, size_t end, int n_jobs) {
        if (LignumModelPredict(model, X_hashed.data() + begin * d.columns, end - begin, d.columns, 1, n_jobs, LIGNUM_STRATEGY_AUTO,
                               out.data() + begin * dim) != 0) die(LignumGetLastError());
    };

    if (argc == 7) {  // one case, once, one thread
        const std::string only = argv[6];
        std::vector<double> row(d.columns);
        if (only == "catboost-batch") {
            cb_strings(0, rows);
        } else if (only == "lignum-batch") {
            lg_strings(0, rows, 1);
        } else if (only == "catboost-row") {
            for (size_t r = 0; r < rows; r++) {
                if (!cb.predict_single(calcer, float_rows[r], d.n_float, cat_rows[r], d.n_cat, out.data() + r * dim, dim)) die(cb.error());
            }
        } else if (only == "lignum-row") {
            for (size_t r = 0; r < rows; r++) {
                for (size_t k = 0; k < d.n_float; k++) row[d.float_columns[k]] = d.floats[r * d.n_float + k];
                for (size_t k = 0; k < d.n_cat; k++) row[d.cat_columns[k]] = LignumCatBoostHash(d.cat(r, k), d.cat_size(r, k));
                if (LignumModelPredict(model, row.data(), 1, d.columns, 1, 1, LIGNUM_STRATEGY_AUTO, out.data() + r * dim) != 0) die(LignumGetLastError());
            }
        } else if (only != "none") {
            die("unknown case " + only);
        }
        std::printf("{\"case\": \"%s\", \"rows\": %zu, \"mismatch\": %zu}\n", only.c_str(), rows, only == "none" ? 0 : mismatches(out, d, rows));
        LignumModelFree(model);
        cb.destroy(calcer);
        return 0;
    }

    // Batches. CatBoost's C API has no thread setting: one call is one thread, so the multi-thread
    // cases split the rows over std::threads for both libraries (lignum hashed: its own threads).
    double t;
    t = best_of(repeats, [&] { cb_strings(0, rows); });                                   report_batch("catboost strings, 1 thread", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { cb_hashed(0, rows); });                                    report_batch("catboost hashed, 1 thread", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { lg_strings(0, rows, 1); });                                report_batch("lignum strings, 1 thread", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { lg_hashed(0, rows, 1); });                                 report_batch("lignum hashed, 1 thread", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { in_chunks(rows, threads, cb_strings); });                  report_batch("catboost strings, threads", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { in_chunks(rows, threads, cb_hashed); });                   report_batch("catboost hashed, threads", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { in_chunks(rows, threads, [&](size_t b, size_t e) { lg_strings(b, e, 1); }); });
    report_batch("lignum strings, threads", t, mismatches(out, d, rows));
    t = best_of(repeats, [&] { lg_hashed(0, rows, threads); });                           report_batch("lignum hashed, threads", t, mismatches(out, d, rows));

    // Single rows and 100 rows from strings, one thread.
    const size_t starts = std::min<size_t>(rows - 100, 1000);
    std::vector<double> row(d.columns);
    std::fill(out.begin(), out.end(), 0.0);
    double m = median_us(starts, [&](size_t r) {
        if (!cb.predict_single(calcer, float_rows[r], d.n_float, cat_rows[r], d.n_cat, out.data() + r * dim, dim)) die(cb.error());
    });
    report_latency("catboost 1 row (CalcModelPredictionSingle)", m, mismatches(out, d, starts));
    std::fill(out.begin(), out.end(), 0.0);
    m = median_us(starts, [&](size_t r) {
        for (size_t k = 0; k < d.n_float; k++) row[d.float_columns[k]] = d.floats[r * d.n_float + k];
        for (size_t k = 0; k < d.n_cat; k++) row[d.cat_columns[k]] = LignumCatBoostHash(d.cat(r, k), d.cat_size(r, k));
        if (LignumModelPredict(model, row.data(), 1, d.columns, 1, 1, LIGNUM_STRATEGY_AUTO, out.data() + r * dim) != 0) die(LignumGetLastError());
    });
    report_latency("lignum 1 row", m, mismatches(out, d, starts));
    m = median_us(starts, [&](size_t r) { cb_strings(r, r + 100); });
    report_latency("catboost 100 rows", m, mismatches(out, d, starts));
    m = median_us(starts, [&](size_t r) { lg_strings(r, r + 100, 1); });
    report_latency("lignum 100 rows", m, mismatches(out, d, starts));

    LignumModelFree(model);
    cb.destroy(calcer);
    return 0;
}
