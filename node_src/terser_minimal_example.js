var Terser = require("terser");

var options = {
    compress: true,
};

var js_src = `declare let global;`

var output = Terser.minify(js_src, options);
console.log(output);

/*
{
  error: Q [SyntaxError]: Unexpected token: keyword (let)
      at ee (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:19541)
      at c (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:28244)
      at l (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:28335)
      at f (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:28388)
      at g (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:28869)
      at T (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:34147)
      at /home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:30509
      at /home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:28976
      at /home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:50966
      at ce (/home/ajay/contracode/node_modules/terser/dist/bundle.min.js:1:51103) {
    filename: '0',
    line: 1,
    col: 8,
    pos: 8
  }
}
*/