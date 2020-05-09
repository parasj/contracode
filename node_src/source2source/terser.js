var Terser = require("terser");

var options = {
    compress: false,
    mangle: false,
    output: {
        beautify: true,
        comments: true,
    }
};

module.exports = (js_src, {
    prob = 0.5, prob_compress = 0.5, prob_mangle = 0.5, prob_remove_comments = 0.5,
    prob_keep_fnames = 0.5, prob_keep_classnames = 0.5, prob_compress_unsafe = 0.5, prob_compress_arguments = 0.5,
    prob_compress_booleans_as_integers = 0.5, prob_compress_drop_console = 0.5, prob_compress_hoist_vars = 0.5,
    prob_compress_dead_code = 0.5
}) => {
    if (Math.random() < prob) {
        if (Math.random() < prob_compress) {
            options["compress"] = {
                "unsafe": (Math.random() < prob_compress_unsafe),
                "arguments": (Math.random() < prob_compress_arguments),
                "booleans_as_integers": (Math.random() < prob_compress_booleans_as_integers),
                "drop_console": (Math.random() < prob_compress_drop_console),
                "hoist_vars": (Math.random() < prob_compress_hoist_vars),
                "dead_code": (Math.random() < prob_compress_dead_code)
            }
        }
        if (Math.random() < prob_mangle) {
            options["mangle"] = true;
        }
        if (Math.random() < prob_remove_comments) {
            options["output"]["comments"] = false;
        }
        if (Math.random() < prob_keep_fnames) {
            options["keep_fnames"] = true;
        }
        if (Math.random() < prob_keep_classnames) {
            options["keep_classnames"] = true;
        }
        // Output options
        options["beautify"] = {
            "braces": (Math.random() < 0.5),
            "indent_level": (Math.random() < 0.5 ? 4 : 2),
            "keep_quoted_props": (Math.random() < 0.5),
            "quote_keys": (Math.random() < 0.5),
            "quote_style": (Math.random() < 0.5 ? 0 : (Math.random() < 0.5 ? 1 : (Math.random() < 0.5 ? 2 : 3)))
        }
        return Terser.minify(js_src, options).code;
    }

    return js_src;
}
