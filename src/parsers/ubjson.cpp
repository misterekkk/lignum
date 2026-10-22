#include "parsers.hpp"
#include "simdjson.h"

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
                    explicit Converter(std::string_view in) : in_(in) {}

                    std::string run() {
                        out_.reserve(in_.size() * 2);
                        value(next_byte(), 0);
                        return std::move(out_);
                    }

                private:
                    static constexpr int kMaxDepth = 256;
                    std::string_view in_;
                    size_t pos_ = 0;
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
                        unsigned char bytes[sizeof(T)];
                        for (size_t i = 0; i < sizeof(T); i++) bytes[i] = static_cast<unsigned char>(in_[pos_ + sizeof(T) - 1 - i]);
                        pos_ += sizeof(T);
                        T v;
                        std::memcpy(&v, bytes, sizeof(T));
                        return v;
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
                            if (peek() != '#') fail("typed container without a count");
                        }
                        if (peek() == '#') {
                            pos_++;
                            count = length();
                            const size_t size = type ? fixed_size(type) : 0;
                            if (size && count > (in_.size() - pos_) / size) fail("container larger than the file");
                        }
                        return {type, count};
                    }

                    void value(uint8_t marker, int depth) {
                        if (depth > kMaxDepth) fail("nested too deeply");
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
                            case 'N': value(next_byte(), depth); return;  // no-op marker
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
                            const std::string_view key(out_.data() + key_start, out_.size() - key_start - 1);
                            out_ += ':';
                            if (!skipping_ && unused(key)) {
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
