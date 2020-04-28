var acornLoose = require("acorn-loose");
var fs = require('fs');

if (process.argv.length !== 3 && process.argv.length !== 4) {
    console.error("USAGE: node parse_javascript.js <INPUT_FILE>");
    process.exit(1);
}

try {
    var data = fs.readFileSync(process.argv[2], 'utf8');
} catch (e) {
    console.error('FSError: ', e.stack);
    process.exit(1);
}

var acorn_options = {'ranges': true, 'locations': true};
console.log(JSON.stringify(acornLoose.parse(data, acorn_options)));