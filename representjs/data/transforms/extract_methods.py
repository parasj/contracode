import json
from typing import List

from representjs.data.transforms.util import dispatch_to_node


def extract_methods(js: str) -> List[str]:
    stdout, stderr = dispatch_to_node('preprocess_extract_methods.js', js)
    result = json.loads(stdout)
    assert isinstance(result, list)
    return result
