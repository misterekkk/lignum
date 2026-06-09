#include "model.hpp"
#include "io.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>
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
            uint8_t version_minor = 1;

            uint8_t os_id = LIGNUM_OS_ID;
            uint8_t arch_id = LIGNUM_ARCH_ID;
            uint8_t transform = 2;
            uint8_t reserved = 0;
            uint32_t endian_check = 0x12345678;

            double base_score = 0.0;
            uint64_t n_nodes = 0;
            uint64_t n_offsets = 0;
            uint64_t n_leaves = 0;
            
            uint8_t padding[16] = {0};
        };

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
        header.base_score = base_score;
        header.transform = transform;
        header.n_nodes = n_nodes;
        header.n_offsets = n_offsets;
        header.n_leaves = n_leaves;

        out.write(reinterpret_cast<const char*>(&header), sizeof(FileHeader));

        padded_write(out, reinterpret_cast<const char*>(k_nodes), n_nodes * sizeof(KNode));
        padded_write(out, reinterpret_cast<const char*>(tree_offsets), n_offsets * sizeof(int));
        padded_write(out, reinterpret_cast<const char*>(leaf_values), n_leaves * sizeof(double));
    }

    Model load_binary(const std::string& filepath) {
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

        if (header.version_major != 0) {
            throw std::runtime_error("Not compatible binary file format.");
        }

        Model model;
        model.base_score = header.base_score;
        model.transform = header.transform;
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
            model.tree_offsets = reinterpret_cast<const int*>(binary_data->data + offset);
            offset += align(model.n_offsets * sizeof(int));
        }
        if (model.n_leaves > 0) {
            model.leaf_values = reinterpret_cast<const double*>(binary_data->data + offset);
        }

        return model;
    }

} // namespace lignum