#include <string>
#include <vector>
#include <map>
#include <experimental/filesystem>

#ifndef PREPROCESSOR_PREPROCESOR_H
#define PREPROCESSOR_PREPROCESOR_H

namespace Preprocessor {
    class SourceFileLoader {
    public:
        SourceFileLoader(std::string rootFile) : rootFile(rootFile) {}
        SourceFileLoader(std::string rootFile, std::experimental::filesystem::path path);
        std::vector<std::string> load();
        static std::unique_ptr<std::vector<std::string>> load(const std::string &file);
    private:
        std::string rootFile;
        std::map<std::string, std::experimental::filesystem::path> lookupPath;
    };
}


#endif
