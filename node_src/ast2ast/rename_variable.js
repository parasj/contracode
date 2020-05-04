const parser = require('@babel/parser').parse;
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require("@babel/types");
const prettier = require('prettier');

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
function(ASTNode, {prob}) -> ASTNode
prob is the probability to replace the name with fixed name
*/

function rename_variable(ast, {prob = 0.5}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                var exampleState = path.node.params[0].name;
                path.scope.rename(exampleState);
            }
        }
    });
    return ast;
}

module.exports = rename_variable;
