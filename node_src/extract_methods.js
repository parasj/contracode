const fs = require('fs');
const parser = require('@babel/parser').parse;
const traverse = require('@babel/traverse').default;
const generate = require('@babel/generator').default;
const t = require("@babel/types");
const prettier = require('prettier');

const js_src = fs.readFileSync(0).toString(); // STDIN_FILENO = 0
const ast = parser(js_src, {sourceType: 'module'});

const function_decls = [];
traverse(ast, {
    FunctionDeclaration(path) {
        function_decls.push(generate(path.node).code);
    }
});

console.log(JSON.stringify(function_decls));
