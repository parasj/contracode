const traverse = require('@babel/traverse').default;
const t = require("@babel/types");
var randomWords = require('random-words');
/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
*/

function insert_var_declaration(ast, {prob = 0.25}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                let len = Math.floor(Math.random() * 2) + 1;
                let name;
                if (Math.random() < 0.5) {
                    name = randomWords({exactly:1, wordsPerString:len, separator:'_'})[0];
                } else {
                    name = randomWords({exactly:1, wordsPerString:len, formatter: (word, index)=> {
                        return index > 0 ? word.slice(0,1).toUpperCase().concat(word.slice(1)) : word;
                    }, separator: ""})[0];
                }
                const id = { type: 'Identifier', name: name };
                path.get('body').unshiftContainer('body', t.variableDeclaration("var", [t.variableDeclarator(id)]));
            }
        }
    });
    return ast;
}

module.exports = insert_var_declaration;
