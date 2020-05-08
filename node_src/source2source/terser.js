var Terser = require("terser");

const fs = require('fs');
const js_src = fs.readFileSync(process.argv[2]).toString(); // STDIN_FILENO = 0
console.log(js_src);
console.log(Terser.minify(js_src).code);

module.exports = (js_src) => {
    return Terser.minify(js_src).code;
}
