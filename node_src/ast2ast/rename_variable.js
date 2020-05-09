const traverse = require('@babel/traverse').default;
const t = require("@babel/types");
var randomWords = require('random-words');

/*
Function signature for transformations:
function(ASTNode, {optional_arguments}) -> List[AST]
Transformations are applied via a flatMap
function(ASTNode, {prob}) -> ASTNode
prob is the probability to replace the name with fixed name
*/

function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

function rename_variable(ast, {prob = 0.25}) {
    traverse(ast, {
        FunctionDeclaration(path) {
            if (Math.random() < prob) {
                if (path.node.params.length > 0) {
                    let idx = getRandomInt(path.node.params.length);
                    var exampleState = path.node.params[idx].name;
                    let len = Math.floor(Math.random() * 2) + 1;
                    let id;
                    if (Math.random() < 0.5) {
                        id = randomWords({exactly:1, wordsPerString:len, separator:'_'})[0];
                    } else {
                        id = randomWords({exactly:1, wordsPerString:len, formatter: (word, index)=> {
                            return index > 0 ? word.slice(0,1).toUpperCase().concat(word.slice(1)) : word;
                        }, separator: ""})[0];
                    }
                    path.scope.rename(exampleState, id);
                }
            }
        }
    });
    return ast;
}

module.exports = rename_variable;
