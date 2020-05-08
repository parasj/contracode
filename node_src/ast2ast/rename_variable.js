const traverse = require('@babel/traverse').default;
const t = require("@babel/types");

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
function(ASTNode, {prob}) -> ASTNode
prob is the probability to replace the name with fixed name
*/

function randomString(length) {
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
    var result = '';
    for (var i = length; i > 0; --i) result += chars[Math.floor(Math.random() * chars.length)];
    return result;
}

function rename_variable(ast, {prob = 0.5}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                var exampleState = path.node.params[0].name;
                let len = Math.random() * 10 + 1;
                const id = randomString(len); 
                path.scope.rename(exampleState, id);
            }
        }
    });
    return ast;
}

module.exports = rename_variable;
