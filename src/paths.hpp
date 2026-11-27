#pragma once

#include <filesystem>
#include <string>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace lignum {

    // Converts a UTF-8 file path to std::filesystem::path so MSVC opens non-ASCII paths properly
    // instead of interpreting narrow std::string in the ANSI code page.
    inline std::filesystem::path utf8_path(const std::string& path) {
    #if defined(_WIN32)
        if (path.empty()) return {};
        const int needed = MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), nullptr, 0);
        if (needed <= 0) return std::filesystem::path(path);
        std::wstring wide(static_cast<size_t>(needed), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), wide.data(), needed);
        return std::filesystem::path(wide);
    #else
        return std::filesystem::path(path);
    #endif
    }

} // namespace lignum
