import representjs
from representjs.data.transforms.extract_methods import extract_methods


def test_extract_methods():
    mergesort_js = (representjs.PACKAGE_ROOT / "data" / "test_js" / "mergesort.js").read_text()
    methods = extract_methods(mergesort_js)
    assert len(methods) == 2
    assert methods[0].startswith("// Split the array")
    assert 'function mergeSort(arr)' in methods[0]
    assert methods[1].startswith("// compare the arrays")
    assert 'function merge(left, right)' in methods[1]
