const traverse = require('@babel/traverse').default;

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function rename_variable(ast, {}) {
    const function_decls = [];
    traverse(ast, {
        FunctionDeclaration(path) {
            function_decls.push(path.node);
        }
    });
    return function_decls;
}

module.exports = rename_variable;
