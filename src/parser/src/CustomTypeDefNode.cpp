#include <tuple>

#include "CustomTypeDefNode.hpp"
#include "Type.hpp"
#include "TypeCheckingUtils.hpp"

void AST::CustomTypeDefNode::semanticPass(Utils::IRContext* context){
    std::vector<std::pair<std::string, std::shared_ptr<Typing::Type>>> attrs;
    for (auto& attr : attributes) {
        attr->semanticPass(context);
        std::pair<std::string, std::shared_ptr<Typing::Type>> ty = std::pair<std::string, std::shared_ptr<Typing::Type>>(attr->name, attr->attrType);
        attrs.push_back(ty);
    }
    std::shared_ptr<Typing::Type> type = TypeCheckUtils::makeCustomType(this->name, attrs);
//    TODO: When semantic symbol table is implemented, add the type to that under the correct name
}
