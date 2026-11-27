#include "parsers.hpp"
#include "simdjson.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lignum {

    namespace parsers {

        namespace {

            // UBJSON (https://ubjson.org, the format of XGBoost's .ubj models) rewritten as JSON text,
            // so the XGBoost JSON parser reads both. Numbers are big-endian; floats are written with
            // the shortest decimal that parses back to the same double, so values stay exact.
            class Converter {
                public:
                    explicit Converter(std::string_view in)
                        : in_(in), zero_byte_budget_(std::max<size_t>(in.size(), 65536)) {}

                    std::string run() {
                        out_.reserve(in_.size() * 2);
                        value(next_byte(), 0);
                        return std::move(out_);
                    }

                private:
                    static constexpr int kMaxDepth = 256;
                    std::string_view in_;
                    size_t pos_ = 0;
                    size_t zero_byte_budget_ = 0;
                    std::string out_;
                    bool skipping_ = false;  // inside a value that is dropped: no number formatting

                    // Tree fields the XGBoost parser never reads (training statistics): written as null.
                    static bool unused(std::string_view key) {
                        return key == "base_weights" || key == "loss_changes" || key == "sum_hessian" || key == "parents";
                    }

                    [[noreturn]] static void fail(const std::string& what) {
                        throw std::runtime_error("Malformed UBJSON model: " + what + ".");
                    }

                    uint8_t next_byte() {
                        if (pos_ >= in_.size()) fail("unexpected end of file");
                        return static_cast<uint8_t>(in_[pos_++]);
                    }

                    uint8_t peek() const {
                        if (pos_ >= in_.size()) fail("unexpected end of file");
                        return static_cast<uint8_t>(in_[pos_]);
                    }

                    template <typename T>
                    T big_endian() {
                        if (in_.size() - pos_ < sizeof(T)) fail("truncated number");
                        const auto* p = reinterpret_cast<const unsigned char*>(in_.data() + pos_);
                        pos_ += sizeof(T);
                        if constexpr (sizeof(T) == 1) {
                            return static_cast<T>(p[0]);
                        } else if constexpr (sizeof(T) == 2) {
                            const uint16_t u = static_cast<uint16_t>((uint16_t{p[0]} << 8) | uint16_t{p[1]});
                            return std::bit_cast<T>(u);
                        } else if constexpr (sizeof(T) == 4) {
                            const uint32_t u = (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | uint32_t{p[3]};
                            return std::bit_cast<T>(u);
                        } else {
                            static_assert(sizeof(T) == 8);
                            const uint64_t u = (uint64_t{p[0]} << 56) | (uint64_t{p[1]} << 48) | (uint64_t{p[2]} << 40) | (uint64_t{p[3]} << 32) |
                                               (uint64_t{p[4]} << 24) | (uint64_t{p[5]} << 16) | (uint64_t{p[6]} << 8) | uint64_t{p[7]};
                            return std::bit_cast<T>(u);
                        }
                    }

                    int64_t integer(uint8_t marker) {
                        switch (marker) {
                            case 'i': return big_endian<int8_t>();
                            case 'U': return big_endian<uint8_t>();
                            case 'I': return big_endian<int16_t>();
                            case 'l': return big_endian<int32_t>();
                            case 'L': return big_endian<int64_t>();
                            default:  fail("expected an integer length");
                        }
                    }

                    size_t length() {
                        const int64_t n = integer(next_byte());
                        if (n < 0 || static_cast<uint64_t>(n) > in_.size()) fail("invalid length");
                        return static_cast<size_t>(n);
                    }

                    void number(double v) {
                        if (skipping_) return;
                        if (!std::isfinite(v)) {
                            out_ += "null";  // JSON has no NaN/inf; the XGBoost parser rejects them where it needs numbers
                            return;
                        }
                        char text[32];
                        char* end = simdjson::internal::to_chars(text, text + sizeof(text), v);
                        out_.append(text, end);
                    }

                    void string(size_t n) {
                        if (in_.size() - pos_ < n) fail("truncated string");
                        out_ += '"';
                        for (size_t i = 0; i < n; i++) {
                            const char c = in_[pos_ + i];
                            if (c == '"' || c == '\\') {
                                out_ += '\\';
                                out_ += c;
                            } else if (static_cast<unsigned char>(c) < 0x20) {
                                char esc[8];
                                std::snprintf(esc, sizeof(esc), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                                out_ += esc;
                            } else {
                                out_ += c;
                            }
                        }
                        out_ += '"';
                        pos_ += n;
                    }

                    // Fixed size of a value of `type` in an optimized container (0: variable).
                    static size_t fixed_size(uint8_t type) {
                        switch (type) {
                            case 'i': case 'U': case 'C': return 1;
                            case 'I': return 2;
                            case 'l': case 'd': return 4;
                            case 'L': case 'D': return 8;
                            case 'Z': case 'T': case 'F': return 0;
                            default:  return 0;
                        }
                    }

                    // Optimized container header: [$type][#count]. Returns the element type (0 if
                    // none) and the count (SIZE_MAX if none: the container ends with ']' or '}').
                    std::pair<uint8_t, size_t> header() {
                        uint8_t type = 0;
                        size_t count = SIZE_MAX;
                        if (peek() == '$') {
                            pos_++;
                            type = next_byte();
                            if (type == 'N') fail("no-op cannot be a container element type");
                            if (peek() != '#') fail("typed container without a count");
                        }
                        if (peek() == '#') {
                            pos_++;
                            count = length();
                            const size_t size = type ? fixed_size(type) : 0;
                            if (size && count > (in_.size() - pos_) / size) fail("container larger than the file");
                            if ((type == 'Z' || type == 'T' || type == 'F') && count > zero_byte_budget_) {
                                fail("container larger than the file");
                            }
                            if (type == 'Z' || type == 'T' || type == 'F') zero_byte_budget_ -= count;
                        }
                        return {type, count};
                    }

                    void value(uint8_t marker, int depth) {
                        if (depth > kMaxDepth) fail("nested too deeply");
                        while (marker == 'N') marker = next_byte();  // no-op marker (bounded by file size, no stack growth)
                        switch (marker) {
                            case 'Z': out_ += "null"; return;
                            case 'T': out_ += "true"; return;
                            case 'F': out_ += "false"; return;
                            case 'i': case 'U': case 'I': case 'l': case 'L': out_ += std::to_string(integer(marker)); return;
                            case 'd': number(static_cast<double>(big_endian<float>())); return;
                            case 'D': number(big_endian<double>()); return;
                            case 'C': string(1); return;
                            case 'S': string(length()); return;
                            case 'H': {  // high-precision number, stored as its decimal text
                                const size_t n = length();
                                if (in_.size() - pos_ < n) fail("truncated number");
                                out_.append(in_.substr(pos_, n));
                                pos_ += n;
                                return;
                            }
                            case '[': array(depth); return;
                            case '{': object(depth); return;
                            default:  fail("unknown marker '" + std::string(1, static_cast<char>(marker)) + "'");
                        }
                    }

                    void array(int depth) {
                        const auto [type, count] = header();
                        out_ += '[';
                        for (size_t i = 0; count == SIZE_MAX || i < count; i++) {
                            if (count == SIZE_MAX && peek() == ']') {
                                pos_++;
                                break;
                            }
                            if (i) out_ += ',';
                            value(type ? type : next_byte(), depth + 1);
                        }
                        out_ += ']';
                    }

                    void object(int depth) {
                        const auto [type, count] = header();
                        out_ += '{';
                        for (size_t i = 0; count == SIZE_MAX || i < count; i++) {
                            if (count == SIZE_MAX && peek() == '}') {
                                pos_++;
                                break;
                            }
                            if (i) out_ += ',';
                            const size_t key_start = out_.size() + 1;
                            string(length());  // keys are strings without the 'S' marker
                            // Decided before appending ':', which may move out_ (the view would
                            // then point into freed memory; found by tests/fuzz).
                            const bool drop = !skipping_ && unused(std::string_view(out_.data() + key_start, out_.size() - key_start - 1));
                            out_ += ':';
                            if (drop) {
                                const size_t mark = out_.size();
                                skipping_ = true;
                                value(type ? type : next_byte(), depth + 1);
                                skipping_ = false;
                                out_.resize(mark);
                                out_ += "null";
                            } else {
                                value(type ? type : next_byte(), depth + 1);
                            }
                        }
                        out_ += '}';
                    }
            };

        } // namespace

        std::string ubjson_to_json(std::string_view data) {
            return Converter(data).run();
        }

    } // namespace parsers

} // namespace lignum
