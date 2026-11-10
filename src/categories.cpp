#include "categories.hpp"

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace lignum {

    namespace {

        // One list's categories by kind. Integers are also found by value among numbers and the
        // other way round (pandas: 1 == 1.0); the first position wins for repeated values.
        struct ListIndex {
            std::unordered_map<std::string, int32_t> text;
            std::unordered_map<int64_t, int32_t> integer;
            std::unordered_map<double, int32_t> number;
            int32_t boolean[2] = {-1, -1};
        };

        struct Index {
            std::vector<ListIndex> lists;
        };

        // An integral double that an int64 holds exactly.
        bool as_integer(double d, int64_t& out) {
            if (!(d >= -9223372036854775808.0 && d < 9223372036854775808.0) || d != std::floor(d)) return false;
            out = static_cast<int64_t>(d);
            return true;
        }

    } // namespace

    void index_categories(TrainingCategories& c) {
        auto index = std::make_shared<Index>();
        index->lists.resize(c.lists.size());
        for (size_t k = 0; k < c.lists.size(); k++) {
            ListIndex& li = index->lists[k];
            for (size_t p = 0; p < c.lists[k].size(); p++) {
                const CategoryValue& v = c.lists[k][p];
                const auto pos = static_cast<int32_t>(p);
                switch (v.kind) {
                    case CategoryValue::Kind::Text: li.text.try_emplace(v.text, pos); break;
                    case CategoryValue::Kind::Integer: li.integer.try_emplace(v.integer, pos); break;
                    case CategoryValue::Kind::Number:
                        if (!std::isnan(v.number)) li.number.try_emplace(v.number, pos);
                        break;
                    case CategoryValue::Kind::Boolean:
                        if (li.boolean[v.integer != 0] < 0) li.boolean[v.integer != 0] = pos;
                        break;
                }
            }
        }
        c.index = index;
    }

    int32_t lookup(const TrainingCategories& c, size_t list, const CategoryValue& value) {
        const ListIndex& li = static_cast<const Index*>(c.index.get())->lists[list];
        switch (value.kind) {
            case CategoryValue::Kind::Text: {
                const auto it = li.text.find(value.text);
                return it == li.text.end() ? -1 : it->second;
            }
            case CategoryValue::Kind::Integer: {
                if (const auto it = li.integer.find(value.integer); it != li.integer.end()) return it->second;
                const double d = static_cast<double>(value.integer);
                int64_t back = 0;
                if (!as_integer(d, back) || back != value.integer) return -1;  // not exact as a double
                const auto it = li.number.find(d);
                return it == li.number.end() ? -1 : it->second;
            }
            case CategoryValue::Kind::Number: {
                if (const auto it = li.number.find(value.number); it != li.number.end()) return it->second;
                int64_t i = 0;
                if (!as_integer(value.number, i)) return -1;
                const auto it = li.integer.find(i);
                return it == li.integer.end() ? -1 : it->second;
            }
            case CategoryValue::Kind::Boolean:
                return li.boolean[value.integer != 0];
        }
        return -1;
    }

} // namespace lignum
