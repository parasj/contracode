const assert = require('assert').strict;
const fs = require('fs');

const lineReader = require('line-reader');
const parser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
// var acornLoose = require("acorn-loose");

const transform_jsonl = require('./transform_jsonl');


var inFilepath = process.argv[2];
var outFilepath = process.argv[3];
var writeStream = fs.createWriteStream(outFilepath)
var numProcessed = 0;
var startTime = Date.now();


// function getPaths(fn) {
//     // Given fn as AST node, get paths.
//     console.log("fn", fn);
//     traverse(fn, {
//         enter(path) {
//             console.log(path);
//         }}
//     )
//     return daf;
//     return [];
// }


function hashCode(str) {
    var hash = 0;
    for (let i = 0; i < str.length; i++) {
        let chr = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + chr;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
};



// var props = ["params", "body", "argument", "property", "key", "value", "object", "property", "callee", "arguments", "params", "properties"];
var terminalTypes = ["Identifier", "Literal"]
var terminalProps = ["name", "raw"]
var numPathSamples = 300;

function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
}  

function get(value) {
    if (terminalTypes.indexOf(value.type) >= 0) {
        // This is a leaf
        var terminalKey = terminalProps[terminalTypes.indexOf(value.type)];
        return [value[terminalKey], true];
        // children.push(value[terminalKey]);  // push a string
        // childIsLeaf.push(true);
    } else {
        return [value, false];
        // children.push(value);  // push a node
        // childIsLeaf.push(false);
    }
}

function getPath(ast, haveSplit=false, isLeft=true) {
    // console.log("ast", ast);
    // console.log("entries", Object.entries(ast));
    // console.log("\n\n\n");
    var children = [];
    var childIsLeaf = [];
    for (const [key, value] of Object.entries(ast)) {
        if (value != null) {
            if (value.hasOwnProperty("type")) {  // TODO: use instanceOf instead
                var [child, isLeaf] = get(value);
                children.push(child);
                childIsLeaf.push(isLeaf);
            } else if (Array.isArray(value)) {
                for (var i = 0; i < value.length; i++) {
                    if (value[i] != null) {
                        var [child, isLeaf] = get(value[i]);
                        children.push(child);
                        childIsLeaf.push(isLeaf);
                    }
                }
            }
        }
    }

    // console.log("children", children);
    if (haveSplit) {
        if (children.length == 0) {
            return null;
        }
        var c = getRndInteger(0, children.length);
        var child = children[c];
        if (childIsLeaf[c]) {
            var path = [child];
        } else {
            var path = getPath(child, true, isLeft);
        }

        if (path != null) {
            if (isLeft) {
                path.push(ast.type);
            } else {
                path.unshift(ast.type);
            }
        }
        return path;
    } else if (children.length >= 2 && Math.random() <= 0.5) {
        // Split
        var i = getRndInteger(0, children.length - 1);
        var j = getRndInteger(i+1, children.length);
        // console.log("Split at", ast.type, i, j);
        var leftPath = getPath(children[i], true, true);
        // console.log("left", leftPath);
        if (leftPath) {
            var rightPath = getPath(children[j], true, false);
            if (rightPath) {
                leftPath.push(ast.type);
                leftPath.push(...rightPath);
                return leftPath;
            }
        }
        return null;
    } else if (children.length >= 1) {
        // Traverse deeper
        // console.log("Going deeper from", ast.type)
        var c = getRndInteger(0, children.length);
        return getPath(children[c]);
    } else {
        return null;
    }

    return [];
}

lineReader.eachLine(inFilepath, function(line, last) {
    var data = JSON.parse(line);
    transform_jsonl._fix_json_dict(data, 'code', 'func_name');
    const identifier = data['func_name'];
    const function_string = data['code'];

    if (identifier) {
        // var ast = acornLoose.parse(function_string);
        // console.log(ast);
        // return false;
        try {
            var ast = parser.parse(function_string, {sourceType: "script", plugins: ["jsx", "es2015", "es6", "v8intrinsic", "typescript", "classProperties"], errorRecovery: false}, {strictMode: false});
        } catch (e) {
            try {
                var ast = parser.parse(function_string, {sourceType: "script", plugins: ["jsx", "es2015", "es6", "v8intrinsic", "flow", "classProperties"], errorRecovery: false}, {strictMode: false});
            } catch (e) {
                console.log("parsing error", e);
                console.log(function_string);
                writeStream.write(identifier + ` ERROR_PARSE,${hashCode("ERROR_PARSE")},ERROR_PARSE\n`)
                return;
            }
        }

        // Get function declaration
        assert.equal(ast.program.body.length, 1);
        fn = ast.program.body[0];
        if (fn.type != 'FunctionDeclaration' && fn.type != 'TSDeclareFunction') {
            throw Error("Invalid function type:" + fn.type)
        }

        // Check name is x
        // console.log(function_string);
        assert.equal(fn.id.type, 'Identifier')
        if (fn.id.name != 'x') {
            console.log(identifier);
            console.log(function_string);
        }
        assert.equal(fn.id.name, 'x')

        // Recur, get path
        var paths = [];
        for (var i = 0; i < numPathSamples; i++) {
            var path = getPath(fn);
            if (path) {
                var nodeStr = path.slice(1,path.length-1).join("|");
                pathStr = `${path[0]},${hashCode(nodeStr)},${path[path.length-1]}`;
                // pathStr = `${path[0]},${path.slice(1,path.length-1)},${path[path.length-1]}`;
                paths.push(pathStr);
            }
        }
        paths = [...new Set(paths)];
        writeStream.write(identifier);
        if (paths.length > 0) {
            paths.forEach(function (path) {
                writeStream.write(" " + path);
            })
        } else {
            writeStream.write(` ERROR_NOPATHS,${hashCode("ERROR_NOPATHS")},ERROR_NOPATHS`);
        }
        writeStream.write("\n");

        // console.log(function_string);
        // console.log(fn.params);
        // console.log(fn.body);
        // return false;

        numProcessed++;
        if (numProcessed % 100 == 0) {
            var elapsed = Date.now() - startTime;
            var linesPerSecond = numProcessed / (elapsed / 1000);
            console.log(`[${inFilepath}] Processed ${numProcessed} lines (${linesPerSecond} per sec)`)
        }
    }

    if (last) {
        console.log(`[${inFilepath}] Done reading! Processed ${numProcessed} lines.`);
        return false;
    }
});