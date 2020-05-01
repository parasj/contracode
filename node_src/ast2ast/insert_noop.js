const traverse = require('@babel/traverse').default;

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function insert_noop(ast, {}) {
    // TODO: Implement
    return ast;
}

module.exports = insert_noop;
