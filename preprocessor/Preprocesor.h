#include <string>
#include <vector>
#include <map>
#include <filesystem>

#ifndef PREPROCESSOR_PREPROCESOR_H
#define PREPROCESSOR_PREPROCESOR_H

namespace Preprocessor {
    class SourceFileLoader {
    public:
        SourceFileLoader(std::string rootFile) : rootFile(rootFile) {}
        SourceFileLoader(std::string rootFile, std::filesystem::path path);
        std::vector<std::string> load();
    private:
        std::string rootFile;
        std::map<std::string, std::filesystem::path> lookupPath;
    };
}


#endif
