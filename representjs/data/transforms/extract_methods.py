from pprint import pprint
from typing import List

import representjs


def extract_methods(js: str) -> List[str]:
    methods = []
    return methods

if __name__ == "__main__":
    js_src = (representjs.PACKAGE_ROOT / "data" / "test_js" / "mergesort.js").read_text()
    pprint(js_ast)
