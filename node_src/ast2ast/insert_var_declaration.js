const parser = require('@babel/parser').parse;
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require("@babel/types");
const prettier = require('prettier'); 
const fs = require('fs');

const js_src = fs.readFileSync(process.argv[2]).toString(); // STDIN_FILENO = 0
const ast = parser(js_src, {sourceType: 'module'});
// console.log(js_src);

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function insert_var_declaration(ast, prob) {
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

// console.log(generate(insert_var_declaration(ast, 1)).code);
module.exports = insert_var_declaration;
