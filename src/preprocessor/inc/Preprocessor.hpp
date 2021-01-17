#pragma once

#include <experimental/filesystem>
#include <map>
#include <string>
#include <vector>

namespace Preprocessor {
    class ProgramFileNode;

    class SourceFileLoader {
    public:
        SourceFileLoader(std::string rootFile) : rootFile(std::move(rootFile)) {}

        SourceFileLoader(std::string rootFile, std::experimental::filesystem::path path);

        std::vector<std::vector<std::string>> load();

        static std::unique_ptr<std::vector<std::string>> load(const std::string &file);

    private:
        std::string rootFile;
        std::map<std::string, std::experimental::filesystem::path> lookupPath;
    };
}  // namespace Preprocessor
