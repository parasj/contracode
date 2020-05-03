const traverse = require('@babel/traverse').default;

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
function(ASTNode, {prob}) -> ASTNode
prob is the probability to replace the name with fixed name
*/

module.exports = (ast, {prob}) => {
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

