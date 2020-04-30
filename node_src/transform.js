const fs = require('fs');
const parser = require('@babel/parser');
const generator = require('@babel/generator').default;

class JavascriptAugmentations {
    constructor() {  // List[string]
        // register transformations
        this.fnAstToAst = {
            'extract_methods': require('./ast2ast/extract_methods.js');
            'rename_variable': require('./ast2ast/rename_variable.js');
            'insert_noop': require('./ast2ast/insert_noop.js');
        };
        this.fnSrcToSrc = {};
    }

    srcToAst(jsSrc) {
        return parser.parse(jsSrc, {sourceType: 'module'})
    }

    astToSrc(ast) {
        return generator(ast).code
    }

    transform(jsSrc, transformationList) {
        let is_ast = false;
        let data = [jsSrc];
        for (const transformationObj of transformationList) {
            const transformation = transformationObj['fn'];
            const options = transformationObj['options'] || {};
            let fnExpectsAst, fnProducesAst, fnTransform;
            if (transformation in this.fnAstToAst) {
                fnExpectsAst = true;
                fnProducesAst = true;
                fnTransform = this.fnAstToAst[transformation];
            } else if (transformation in this.fnSrcToSrc) {
                fnExpectsAst = false;
                fnProducesAst = false;
                fnTransform = this.fnSrcToSrc[transformation];
            } else {
                const validTransforms = new Set(Object.keys(this.fnSrcToSrc).concat(Object.keys(this.fnAstToAst)))
                console.error(`Transformation ${transformation} not in accepted list of transformations ${validTransforms}`);
                process.exit(1);
            }

            if (fnExpectsAst && !is_ast) {
                data = data.map(this.srcToAst);
                is_ast = true;
            } else if (!fnExpectsAst && is_ast) {
                data = data.map(this.astToSrc);
                is_ast = false;
            }

            data = data.flatMap(x => fnTransform(x, options));
            is_ast = fnProducesAst;
        }

        if (is_ast) {
            return data.map(this.astToSrc);
        } else {
            return data;
        }
    }
}

var data = JSON.parse(fs.readFileSync(0, 'utf-8'));
const javascriptAugmenter = new JavascriptAugmentations();
console.log(JSON.stringify(data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations']))));
