var Terser = require("terser");

var options = {
    compress: false,
    mangle: false,
    output: {
        beautify: true,
        comments: true,
    }
};

module.exports = (js_src, {prob_compress = 0.5, prob_mangle = 0.5, prob_remove_comments = 0.5}) => {
    if (Math.random() < prob_compress) {
        options["compress"] = true;
    }
    if (Math.random() < prob_mangle) {
        options["mangle"] = true;
    }
    if (Math.random() < prob_compress) {
        options["output"]["comments"] = false;
    }
    return Terser.minify(js_src, options).code;
}
