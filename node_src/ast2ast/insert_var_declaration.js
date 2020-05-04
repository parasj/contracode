const parser = require('@babel/parser').parse;
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require("@babel/types");
const prettier = require('prettier'); 

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function insert_var_declaration(ast, {prob = 0.5}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                const id = path.scope.generateUidIdentifierBasedOnNode(path.node.id);
                path.get('body').unshiftContainer('body', t.variableDeclaration("var", [t.variableDeclarator(id)]));
            }
        }
    });
    return ast;
}

module.exports = insert_var_declaration;
