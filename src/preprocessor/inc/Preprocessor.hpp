#include <filesystem>
#include <map>
#include <string>
#include <vector>

#ifndef PREPROCESSOR_PREPROCESOR_H
#define PREPROCESSOR_PREPROCESOR_H

namespace Preprocessor {
class SourceFileLoader {
   public:
    SourceFileLoader(std::string rootFile) : rootFile(rootFile) {}
    SourceFileLoader(std::string rootFile, std::filesystem::path path);
    std::vector<std::string> load();
    static std::unique_ptr<std::vector<std::string>> load(std::string file);

   private:
    std::string rootFile;
    std::map<std::string, std::filesystem::path> lookupPath;
};
}  // namespace Preprocessor

#endif
