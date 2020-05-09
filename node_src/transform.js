const fs = require('fs');
var JavascriptAugmentations = require('./javascript_augmentations')

var data = JSON.parse(fs.readFileSync(0, 'utf-8'));
const javascriptAugmenter = new JavascriptAugmentations();
console.log(JSON.stringify(data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations']))));
