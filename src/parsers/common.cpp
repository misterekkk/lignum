#include "parsers.hpp"
#include "../paths.hpp"
#include "simdjson.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lignum {

    namespace parsers {

        std::string read_file(const std::string& filepath) {
            std::ifstream file(utf8_path(filepath), std::ios::binary | std::ios::ate);
            if (!file) throw std::runtime_error("Cannot open " + filepath + ".");
            const std::streamoff size = file.tellg();
            if (size < 0) throw std::runtime_error("Cannot read " + filepath + ".");
            std::string content(static_cast<size_t>(size), '\0');
            file.seekg(0);
            if (!file.read(content.data(), size)) throw std::runtime_error("Cannot read " + filepath + ".");
            return content;
        }

        double parse_double(std::string_view token, const char* format_name) {
            auto bad = [&](const std::string& msg) -> void {
                if (format_name) throw std::runtime_error(std::string("Malformed ") + format_name + " model file: " + msg + ".");
                throw std::runtime_error("Malformed model file: " + msg + ".");
            };
            if (token.empty()) bad("empty number");
            std::string_view body = token;
            const bool negative = body[0] == '-';
            if (body[0] == '-' || body[0] == '+') body.remove_prefix(1);
            if (body == "inf" || body == "Infinity" || body == "infinity") {
                return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
            }
            if (body == "nan" || body == "NaN") return std::numeric_limits<double>::quiet_NaN();
            // Strict syntax first: from_chars below assumes a well-formed number.
            size_t i = 0, digits = 0;
            while (i < body.size() && body[i] >= '0' && body[i] <= '9') { i++; digits++; }
            if (i < body.size() && body[i] == '.') {
                i++;
                while (i < body.size() && body[i] >= '0' && body[i] <= '9') { i++; digits++; }
            }
            if (digits == 0) bad("invalid number '" + std::string(token) + "'");
            if (i < body.size() && (body[i] == 'e' || body[i] == 'E')) {
                i++;
                if (i < body.size() && (body[i] == '+' || body[i] == '-')) i++;
                const size_t start = i;
                while (i < body.size() && body[i] >= '0' && body[i] <= '9') i++;
                if (i == start) bad("invalid number '" + std::string(token) + "'");
            }
            if (i != body.size()) bad("invalid number '" + std::string(token) + "'");
            const double v = simdjson::internal::from_chars(body.data(), body.data() + body.size());
            return negative ? -v : v;
        }

    } // namespace parsers

} // namespace lignum
