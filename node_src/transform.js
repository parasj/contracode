const fs = require('fs');
const parser = require('@babel/parser');
const generator = require('@babel/generator').default;

class JavascriptAugmentations {
    constructor() {  // List[string]
        // register transformations
        this.fnAstToAst = {
            'rename_variable': require('./ast2ast/rename_variable.js'),
            'insert_noop': require('./ast2ast/insert_noop.js'),
            'extract_methods': require('./preprocess_extract_methods.js')
        };
        this.fnSrcToSrc = {};
    }

    srcToAst(jsSrc) { return parser.parse(jsSrc, {sourceType: 'module'}); }

    astToSrc(ast) { return generator(ast).code; }

    transform_match_input(data, is_ast, transformation) {
        // transform the data to AST or source depending on class of transformation
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
            return new Error(`Transformation ${transformation} not in accepted list of transformations ${validTransforms}`)
        }

        if (fnExpectsAst && !is_ast) {
            data = data.map(this.srcToAst);
            is_ast = true;
        } else if (!fnExpectsAst && is_ast) {
            data = data.map(this.astToSrc);
            is_ast = false;
        }
        return {'data': data, 'expects_ast': is_ast, 'produces_ast': fnProducesAst, 'fn': fnTransform}
    }

    transform(jsSrc, transformationList) {
        let is_ast = false;
        let data = [jsSrc];

        for (const transformationObj of transformationList) {
            const transformation = transformationObj['fn'];
            const options = transformationObj['options'] || {};
            const transformed_obj = this.transform_match_input(data, is_ast, transformation);
            if (transformed_obj instanceof Error) {
                return transformed_obj;
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
