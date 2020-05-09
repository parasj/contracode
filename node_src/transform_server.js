const http = require('http');
var JavascriptAugmentations = require('./javascript_augmentations')

const hostname = '127.0.0.1';
const port = 3000;

const javascriptAugmenter = new JavascriptAugmentations();

const server = http.createServer((req, res) => {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/json');

    let body = [];
    req.on('data', (chunk) => {
        body.push(chunk);
    }).on('end', () => {
        body = Buffer.concat(body).toString();
        const data = JSON.parse(body);
        console.log("got data", data);
        const replyString = JSON.stringify(
            data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations'])));
        res.end(replyString);
    });
});

server.listen(port, hostname, () => {
    console.log(`Server running at http://${hostname}:${port}/`);
});

