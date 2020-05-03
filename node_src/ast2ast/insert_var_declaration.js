const traverse = require('@babel/traverse').default;
const t = require("@babel/types");

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

module.exports = function (ast, {prob}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                const id = path.scope.generateUidIdentifierBasedOnNode(path.node.id);
                path.get('body').unshiftContainer('body', t.variableDeclaration("var", [t.variableDeclarator(id)]));
            }
        }
    });
    return ast;
};
