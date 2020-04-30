const traverse = require('@babel/traverse').default;
const parser = require('@babel/parser');
const generator = require('@babel/generator').default;
const glob = require('glob');
const fs = require('fs');
const async = require('async');
const cliProgress = require('cli-progress');

console.log("loading file index");
const file_list = glob.sync(__dirname + '/../data/data/**/*.js').map(x => { return {'fname': x}});

const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
bar.start(file_list.length, 0);
const pending_map = async.mapLimit(file_list, 32, async (task) => {
    let js;
    try {
        js = fs.readFileSync(task['fname'], encoding = "utf-8");
        const ast = parser.parse(js);
        const function_decls = [];
        traverse(ast, {
            FunctionDeclaration(path) {
                function_decls.push(generator(path.node).code);
            }
        });
        task['status_err'] = false;
        task['function_decls'] = function_decls;
    } catch (err) {
        console.error(task['fname'], err);
        task['status_err'] = true;
        task['err'] = err;
    }
    bar.increment();
    return task;
}, (err, results) => {
    bar.stop();
    if (err) {
        console.error(err);
    } else {
        try {
            fs.writeFileSync(__dirname + "../data/methods.json", JSON.stringify(data));
        } catch (err) {
            console.error(err)
        }
    }
});

