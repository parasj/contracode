const parser = require('@babel/parser');
const generator = require('@babel/generator').default;

class JavascriptAugmentations {
    constructor() {  // List[string]
        // register transformations
        this.fnAstToAst = {
            'identity_ast2ast': require('./ast2ast/identity_ast2ast.js'),
            'rename_variable': require('./ast2ast/rename_variable.js'),
            'insert_var_declaration': require('./ast2ast/insert_var_declaration.js'),
        };
        this.fnSrcToSrc = {
            'sample_lines': require('./source2source/sample_lines.js'),
            // 'compress': require('./source2source/compress.js'),
            // 'mangle': require('./source2source/mangle.js'),
            // 'compress_mangle': require('./source2source/compress_mangle.js'),
            // 'remove_comments': require('./source2source/remove_comments.js'),
            'terser': require('./source2source/terser.js'),
        };
    }

    srcToAst(jsSrc) {
        try {
            return parser.parse(jsSrc, {sourceType: "module", plugins: ["jsx", "es2015", "es6", "flow"], errorRecovery: true});
        } catch (e) {
            return parser.parse(jsSrc, {sourceType: "module", plugins: ["jsx", "es2015", "es6", "typescript"], errorRecovery: true});
        }
    }

    astToSrc(ast) {
        return generator(ast).code;
    }

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
            data = this.srcToAst(data);
            is_ast = true;
        } else if (!fnExpectsAst && is_ast) {
            data = this.astToSrc(data);
            is_ast = false;
        }
        return {'data': data, 'expects_ast': is_ast, 'produces_ast': fnProducesAst, 'fnTransform': fnTransform}
    }

    transform(jsSrc, transformationList) {
        let is_ast = false;
        let data = jsSrc;

        for (const transformationObj of transformationList) {
            // console.log("javascript_augmentations.transform", transformationObj)
            const transformation = transformationObj['fn'];
            const options = transformationObj['options'] || {};
            try {
                const transformed_obj = this.transform_match_input(data, is_ast, transformation);
                let data_new = transformed_obj['fnTransform'](transformed_obj['data'], options);
                const is_ast_new = transformed_obj['produces_ast'];

                if (data_new != null && data_new != undefined) {
                    // no exception thrown
                    data = data_new;
                    is_ast = is_ast_new;
                }
            } catch (e) {
                console.error("Could not transform object!", transformationObj, ", got error ", e);
                console.log(jsSrc)
            }
        }

        if (is_ast) {
            data = this.astToSrc(data);
        }

        return data;
    }
}

module.exports = JavascriptAugmentations;

// var data = JSON.parse(fs.readFileSync(0, 'utf-8'));
// const javascriptAugmenter = new JavascriptAugmentations();
// console.log(JSON.stringify(data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations']))));
