const traverse = require('@babel/traverse').default;

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]

Transformations are applied via a flatMap
 */

function extract_methods(ast, {}) {
    const function_decls = [];
    traverse(ast, {
        FunctionDeclaration(path) {
            function_decls.push(path.node);
        }
    });
    return function_decls;
}

module.exports = extract_methods;