#include "model.hpp"
#include "io.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>
#include <string_view>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace lignum {

    namespace {

        struct alignas(64) FileHeader {
            char magic[8] = {'L', 'I', 'G', 'N', 'U', 'M', '\0', '\1'}; // name + version
            double base_score = 0.0;
            uint64_t n_nodes = 0;
            uint64_t n_offsets = 0;
            uint64_t n_leaves = 0;
            uint8_t padding[24] = {0};
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
                hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
                if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Error opening file.");
                
                hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
                if (!hMap) { CloseHandle(hFile); throw std::runtime_error("Error running mmap."); }

                data = static_cast<const char*>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
            #else
                fd = open(filepath.c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("Error opening file.");

                struct stat sb;
                fstat(fd, &sb);
                size = sb.st_size;

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

        const FileHeader* header = reinterpret_cast<const FileHeader*>(binary_data->data);

        Model model;
        model.base_score = header->base_score;
        model.n_nodes = header->n_nodes;
        model.n_offsets = header->n_offsets;
        model.n_leaves = header->n_leaves;
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