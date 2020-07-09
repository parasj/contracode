const fs = require('fs');
const lineReader = require('line-reader');

function escapeRegExp(string) {
    return string.replace(/[.*+\-?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}


function _fix_json_dict(json_dict, src_function_key, src_method_name_key) {
    // Fix cropped "function" token at the begging of the function string
    // for (var regex of _fix_function_crop_regexes) {
    //     json_dict[src_function_key] = regex.sub(r'function\1', json_dict[src_function_key], count=1);
    // }
    const method_name = json_dict[src_method_name_key].replace("(", "");
    if (method_name) {
        // Remove function name from declaration, but leave it in the function body
        var _function_name_regex = new RegExp(`(function\\s*)${escapeRegExp(method_name)}(\\s*\\()`, '');
        var new_fn = json_dict[src_function_key].replace(_function_name_regex, '$1x$2');
        if (new_fn === json_dict[src_function_key]) {
            _function_name_regex = new RegExp(`(function\\*?\\s*)${escapeRegExp(method_name)}(.*\\()`, '');
            new_fn = json_dict[src_function_key].replace(_function_name_regex, '$1x$2');
            if (new_fn === json_dict[src_function_key]) {
                _function_name_regex = new RegExp(`(function.*)${escapeRegExp(method_name)}(\\s*\\()`, '');
                new_fn = json_dict[src_function_key].replace(_function_name_regex, '$1x$2');
            }
        }
        json_dict[src_function_key] = new_fn;
    } else {
        json_dict[src_function_key] = "const x = " + json_dict[src_function_key];
    }
}


if (require.main === module) {
    var JavascriptAugmentations = require('./javascript_augmentations')
    const javascriptAugmenter = new JavascriptAugmentations();

    const augmentations = [	
        {"fn": "rename_variable"},
        {"fn": "insert_var_declaration"},
        {"fn": "terser"},
        {"fn": "sample_lines"}
    ]

    const numAlternatives = 20;

    var inFilepath = process.argv[2];
    var outFilepath = process.argv[3];
    var writeStream = fs.createWriteStream(outFilepath)
    var numProcessed = 0;
    var startTime = Date.now();

    lineReader.eachLine(inFilepath, function(line, last) {
        var data = JSON.parse(line);
        _fix_json_dict(data, 'function', 'identifier');
        const identifier = data['identifier'];
        const fn = data['function'];

        let alternatives = [fn];
        for (let i = 0; i < numAlternatives; i++) {
            const transformed = javascriptAugmenter.transform(fn, augmentations);
            alternatives.push(transformed);
        }
        let alternativesStr = JSON.stringify(alternatives);
        writeStream.write(alternativesStr + "\n")

        numProcessed++;

        if (numProcessed % 1000 == 0) {
            var elapsed = Date.now() - startTime;
            var linesPerSecond = numProcessed / (elapsed / 1000);
            var remaining = (1843099 - numProcessed) / linesPerSecond / 3600;
            console.log(`[${inFilepath}] Processed ${numProcessed} lines (${linesPerSecond} per sec, ${remaining} hours remaining)`)
        }

        if (last) {
            console.log(`[${inFilepath}] Done reading!`);

            return false;
        }
    });
}

module.exports = {
    _fix_json_dict: _fix_json_dict
}