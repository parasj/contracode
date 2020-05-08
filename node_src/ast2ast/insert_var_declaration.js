const traverse = require('@babel/traverse').default;
const t = require("@babel/types");
/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function randomString(length) {
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    var result = '';
    for (var i = length; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
    return result;
}

function insert_var_declaration(ast, {prob = 0.5}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                //const id = path.scope.generateUidIdentifierBasedOnNode(path.node.id);
                let len = Math.random() * 10 + 1;
                const id = { type: 'Identifier', name: randomString(len) };
                path.get('body').unshiftContainer('body', t.variableDeclaration("var", [t.variableDeclarator(id)]));
            }
        }
    });
    return ast;
}

module.exports = insert_var_declaration;
