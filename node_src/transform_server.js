// const cluster = require('cluster');
// const http = require('http');
// const numCPUs = 1; //Math.floor(require('os').cpus().length * 0.5);

// const hostname = '127.0.0.1';
// const port = 3000;

// if (cluster.isMaster) {
//     console.log(`Master ${process.pid} is running`);
//     for (let i = 0; i < numCPUs; i++) {
//         cluster.fork();
//     }

//     cluster.on('exit', (worker, code, signal) => {
//         console.error(`worker ${worker.process.pid} died`);
//         console.log("Spawning new worker to replace");
//         cluster.fork();
//     });
// } else {
//     console.log(`Worker ${process.pid} started`);
//     var JavascriptAugmentations = require('./javascript_augmentations');
//     const javascriptAugmenter = new JavascriptAugmentations();
//     const server = http.createServer((req, res) => {
//         res.statusCode = 200;
//         res.setHeader('Content-Type', 'application/json');

//         let body = [];
//         req.on('data', (chunk) => {
//             body.push(chunk);
//         }).on('end', () => {
//             body = Buffer.concat(body).toString();
//             const data = JSON.parse(body);
//             const replyString = JSON.stringify(
//                 data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations'])));
//             res.end(replyString);
//         });
//     });

//     server.timeout = 1000;  // 1000 ms

//     server.listen(port, hostname, () => {
//         console.log(`Server[${process.pid}] running at http://${hostname}:${port}/`);
//     });
// }


const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

console.log(`Node process ${process.pid} started`);
var JavascriptAugmentations = require('./javascript_augmentations');
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
        const replyString = JSON.stringify(
            data.map(x => javascriptAugmenter.transform(x['src'], x['augmentations'])));
        res.end(replyString);
        // console.log("Body", body);
        // console.log("Reply", replyString);
    });
});

server.timeout = 1000;  // 1000 ms

server.listen(port, hostname, () => {
    console.log(`Server[${process.pid}] running at http://${hostname}:${port}/`);
});
