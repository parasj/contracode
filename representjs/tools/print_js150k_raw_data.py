import json
from pprint import pprint

from representjs.data.js150k import JS150kDataset
import pyjsparser


def main():
    js150k = JS150kDataset()
    eval_data = js150k.eval_data_raw()
    for row in [eval_data[1]]:
        print(row['path'])
        print(row['js'])
        js_ast = pyjsparser.parse(row['js'])
        print(json.dumps(js_ast))


if __name__ == "__main__":
    main()
