#include <lignum/model.hpp>
#include "io.hpp"
#include "binned.hpp"
#include "validate.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <string_view>
#include <cstring>

#if defined(_WIN32)
    #include <windows.h>

    std::wstring utf8_to_utf16(const std::string& str) {
        if (str.empty()) return std::wstring();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
        return wstrTo;
    }

#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

#if defined(_WIN32)
    #define LIGNUM_OS_ID 1
#elif defined(__unix__) || defined(__unix) || defined(__APPLE__) || defined(__linux__)
    #define LIGNUM_OS_ID 2
#else
    #define LIGNUM_OS_ID 0
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define LIGNUM_ARCH_ID 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON__)
    #define LIGNUM_ARCH_ID 2
#else
    #define LIGNUM_ARCH_ID 0
#endif

namespace lignum {

    namespace {

        struct alignas(64) FileHeader {
            char magic[6] = {'L', 'I', 'G', 'N', 'U', 'M'}; 
            uint8_t version_major = 0;
            uint8_t version_minor = 6;

            uint8_t os_id = LIGNUM_OS_ID;
            uint8_t arch_id = LIGNUM_ARCH_ID;
            uint8_t transform = 2;
            uint8_t flags = 0;
            uint32_t endian_check = 0x12345678;

            uint64_t n_cat_words = 0;
            uint64_t n_nodes = 0;
            uint64_t n_offsets = 0;
            uint64_t n_leaves = 0;

            double sigmoid = 1.0;
            uint32_t n_features = 0;
            uint16_t n_outputs = 1;
            uint8_t padding[2] = {0};
        };
        static_assert(sizeof(FileHeader) == 64, "FileHeader must stay 64 bytes");

        namespace header_flags {
            constexpr uint8_t Float32 = 1 << 0;
            constexpr uint8_t SpecialTrees = 1 << 1;
            constexpr uint8_t Binned = 1 << 2;  // a BinnedFileHeader and its sections follow
        } // namespace header_flags

        // Binned sections, in file order: (pointer, element size, element count).
        struct Section {
            const void* data;
            size_t element_size;
            size_t count;
        };

        std::vector<Section> binned_sections(const BinnedForest& f) {
            return {
                {f.nodes, sizeof(BNode), f.n_nodes},
                {f.roots, sizeof(int32_t), f.n_trees},
                {f.tree_kind, sizeof(uint8_t), f.n_trees},
                {f.feature_of, sizeof(uint32_t), f.n_features},
                {f.feature_kind, sizeof(uint8_t), f.n_features},
                {f.zero_bin, sizeof(uint16_t), f.n_features},
                {f.cut_offsets, sizeof(uint32_t), f.n_features + 1},
                {f.cuts, sizeof(double), f.n_cuts},
                {f.cat_features, sizeof(BinnedCatFeature), f.n_cat_features},
                {f.cat_tables, sizeof(uint64_t), f.n_table_words},
            };
        }

        size_t align(size_t size) {
            return (size + 63) & ~size_t(63); // aligns data to 64B
        }

        void padded_write(std::ofstream& out, const char* data, size_t size_bytes) {
            if (size_bytes == 0) return;
            
            out.write(data, size_bytes);
            
            size_t padded_size = align(size_bytes);
            if (padded_size > size_bytes) {
                std::vector<char> padding(padded_size - size_bytes, 0);
                out.write(padding.data(), padding.size());
            }
        }

    } // namespace

    class MmapData : public Data {
        public:
            const char* data = nullptr;
            size_t size = 0;
        
        #if defined(_WIN32)
            HANDLE hFile = INVALID_HANDLE_VALUE;
            HANDLE hMap = NULL;
        #else
            int fd = -1;
        #endif

        MmapData(const std::string& filepath) {
            #if defined(_WIN32)
                std::wstring wfilepath = utf8_to_utf16(filepath);
                hFile = CreateFileW(wfilepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Error opening file.");

                LARGE_INTEGER file_size;
                if (!GetFileSizeEx(hFile, &file_size)) {
                    CloseHandle(hFile);
                    throw std::runtime_error("Error getting file size.");
                }
                size = file_size.QuadPart;

                if (size == 0) { CloseHandle(hFile); throw std::runtime_error("File is empty."); }
                
                hMap = CreateFileMappingW(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
                if (!hMap) { CloseHandle(hFile); throw std::runtime_error("Error running mmap."); }

                data = static_cast<const char*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
                if (!data) throw std::runtime_error("Error mapping view of file.");

            #else
                fd = open(filepath.c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("Error opening file: " + filepath);

                struct stat sb;
                if (fstat(fd, &sb) < 0) { close(fd); throw std::runtime_error("Error getting file size."); }
                size = sb.st_size;

                if (size == 0) { close(fd); throw std::runtime_error("File is empty."); }

                data = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
                if (data == MAP_FAILED) { close(fd); throw std::runtime_error("Error running mmap."); }
            #endif
        }

        ~MmapData() override {
            #if defined(_WIN32)
                if (data) UnmapViewOfFile(data);
                if (hMap) CloseHandle(hMap);
                if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
            #else
                if (data && data != MAP_FAILED) munmap(const_cast<char*>(data), size);
                if (fd >= 0) close(fd);
            #endif
        }
    };

    void Model::save(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) throw std::runtime_error("Can't save the file.");

        FileHeader header;
        header.n_outputs = static_cast<uint16_t>(n_outputs);
        header.n_cat_words = n_cat_words;
        header.sigmoid = sigmoid;
        header.transform = transform;
        header.flags = static_cast<uint8_t>((float32 ? header_flags::Float32 : 0) | (has_special_trees ? header_flags::SpecialTrees : 0) |
                                            (binned ? header_flags::Binned : 0));
        header.n_features = n_features;
        header.n_nodes = n_nodes;
        header.n_offsets = n_offsets;
        header.n_leaves = n_leaves;

        out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));

        padded_write(out, reinterpret_cast<const char*>(k_nodes), n_nodes * sizeof(KNode));
        padded_write(out, reinterpret_cast<const char*>(tree_offsets), n_offsets * sizeof(int32_t));
        padded_write(out, reinterpret_cast<const char*>(leaf_values), n_leaves * sizeof(double));
        padded_write(out, reinterpret_cast<const char*>(tree_kinds), n_offsets * sizeof(uint8_t));
        padded_write(out, reinterpret_cast<const char*>(base_scores), n_outputs * sizeof(double));
        if (n_outputs > 1) {
            padded_write(out, reinterpret_cast<const char*>(tree_outputs), n_offsets * sizeof(uint16_t));
        }
        padded_write(out, reinterpret_cast<const char*>(cat_bitsets), n_cat_words * sizeof(uint32_t));

        // The binned kernel is stored too, so a mapped model needs no preparation at load time.
        if (binned) {
            BinnedFileHeader bh;
            bh.n_nodes = binned->n_nodes;
            bh.n_trees = binned->n_trees;
            bh.n_features = binned->n_features;
            bh.n_cuts = binned->n_cuts;
            bh.n_cat_features = binned->n_cat_features;
            bh.n_table_words = binned->n_table_words;
            bh.decision_words = binned->decision_words;
            out.write(reinterpret_cast<const char*>(&bh), sizeof(bh));
            for (const Section& sec : binned_sections(*binned)) {
                padded_write(out, static_cast<const char*>(sec.data), sec.element_size * sec.count);
            }
        }
        if (!out) throw std::runtime_error("Failed to write the model file.");
    }

    Model load_binary(const std::string& filepath, bool binned) {
        auto binary_data = std::make_shared<MmapData>(filepath);

        if (binary_data->size < sizeof(FileHeader)) {
            throw std::runtime_error("File is too small to contain valid header.");
        }

        FileHeader header;
        std::memcpy(&header, binary_data->data, sizeof(FileHeader));

        if (std::memcmp(header.magic, "LIGNUM", 6) != 0) {
            throw std::runtime_error("Binary file does not contain LIGNUM signature.");
        }

        if (header.endian_check != 0x12345678) {
            throw std::runtime_error("Endianness mismatch.");
        }

        if (header.version_major != 0 || header.version_minor != 6) {
            throw std::runtime_error("Unsupported lignum binary format version " + std::to_string(header.version_major) + "." +
                                     std::to_string(header.version_minor) + "; re-export the model from JSON.");
        }

        const size_t expected_size = sizeof(FileHeader) + align(header.n_nodes * sizeof(KNode)) +
                                     align(header.n_offsets * sizeof(int32_t)) + align(header.n_leaves * sizeof(double)) +
                                     align(header.n_offsets * sizeof(uint8_t)) + align(header.n_outputs * sizeof(double)) +
                                     (header.n_outputs > 1 ? align(header.n_offsets * sizeof(uint16_t)) : 0) +
                                     header.n_cat_words * sizeof(uint32_t);
        if (header.n_outputs == 0) {
            throw std::runtime_error("Binary file declares zero outputs.");
        }
        if (binary_data->size < expected_size) {
            throw std::runtime_error("Binary file is truncated or corrupted.");
        }

        Model model;
        model.n_outputs = header.n_outputs;
        model.sigmoid = header.sigmoid;
        model.transform = header.transform;
        model.float32 = (header.flags & header_flags::Float32) != 0;
        model.has_special_trees = (header.flags & header_flags::SpecialTrees) != 0;
        model.n_cat_words = header.n_cat_words;
        model.n_features = header.n_features;
        model.n_nodes = header.n_nodes;
        model.n_offsets = header.n_offsets;
        model.n_leaves = header.n_leaves;
        model.data = binary_data;

        size_t offset = sizeof(FileHeader);

        if (model.n_nodes > 0) {
            model.k_nodes = reinterpret_cast<const KNode*>(binary_data->data + offset);
            offset += align(model.n_nodes * sizeof(KNode));
        }
        if (model.n_offsets > 0) {
            model.tree_offsets = reinterpret_cast<const int32_t*>(binary_data->data + offset);
            offset += align(model.n_offsets * sizeof(int32_t));
        }
        if (model.n_leaves > 0) {
            model.leaf_values = reinterpret_cast<const double*>(binary_data->data + offset);
            offset += align(model.n_leaves * sizeof(double));
        }
        if (model.n_offsets > 0) {
            model.tree_kinds = reinterpret_cast<const uint8_t*>(binary_data->data + offset);
            offset += align(model.n_offsets * sizeof(uint8_t));
        }
        model.base_scores = reinterpret_cast<const double*>(binary_data->data + offset);
        offset += align(model.n_outputs * sizeof(double));
        if (model.n_outputs > 1 && model.n_offsets > 0) {
            model.tree_outputs = reinterpret_cast<const uint16_t*>(binary_data->data + offset);
            offset += align(model.n_offsets * sizeof(uint16_t));
        }
        if (model.n_cat_words > 0) {
            model.cat_bitsets = reinterpret_cast<const uint32_t*>(binary_data->data + offset);
        }
        offset += align(model.n_cat_words * sizeof(uint32_t));

        if (binned && (header.flags & header_flags::Binned)) {
            if (binary_data->size < offset + sizeof(BinnedFileHeader)) {
                throw std::runtime_error("Binary file is truncated or corrupted.");
            }
            BinnedFileHeader bh;
            std::memcpy(&bh, binary_data->data + offset, sizeof(bh));
            offset += sizeof(bh);

            auto f = std::make_shared<BinnedForest>();
            f->n_nodes = bh.n_nodes;
            f->n_trees = bh.n_trees;
            f->n_features = bh.n_features;
            f->n_cuts = bh.n_cuts;
            f->n_cat_features = bh.n_cat_features;
            f->n_table_words = bh.n_table_words;
            f->decision_words = bh.decision_words;
            if (f->n_trees != model.n_offsets) throw std::runtime_error("Binary file is truncated or corrupted.");

            size_t needed = offset;
            for (const Section& sec : binned_sections(*f)) needed += align(sec.element_size * sec.count);
            if (binary_data->size < needed) throw std::runtime_error("Binary file is truncated or corrupted.");

            auto at = [&](size_t bytes) {
                const char* p = binary_data->data + offset;
                offset += align(bytes);
                return p;
            };
            f->nodes = reinterpret_cast<const BNode*>(at(f->n_nodes * sizeof(BNode)));
            f->roots = reinterpret_cast<const int32_t*>(at(f->n_trees * sizeof(int32_t)));
            f->tree_kind = reinterpret_cast<const uint8_t*>(at(f->n_trees * sizeof(uint8_t)));
            f->feature_of = reinterpret_cast<const uint32_t*>(at(f->n_features * sizeof(uint32_t)));
            f->feature_kind = reinterpret_cast<const uint8_t*>(at(f->n_features * sizeof(uint8_t)));
            f->zero_bin = reinterpret_cast<const uint16_t*>(at(f->n_features * sizeof(uint16_t)));
            f->cut_offsets = reinterpret_cast<const uint32_t*>(at((f->n_features + 1) * sizeof(uint32_t)));
            f->cuts = reinterpret_cast<const double*>(at(f->n_cuts * sizeof(double)));
            f->cat_features = reinterpret_cast<const BinnedCatFeature*>(at(f->n_cat_features * sizeof(BinnedCatFeature)));
            f->cat_tables = reinterpret_cast<const uint64_t*>(at(f->n_table_words * sizeof(uint64_t)));
            f->owner = binary_data;
            model.binned = f;
        }

        validate_model(model);
        return model;
    }

} // namespace lignum