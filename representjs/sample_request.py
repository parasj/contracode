import requests
import json

#transform_payload = [{'src': "const x = function(arr) { console.log('hi', arr); }", 'augmentations': [	
transform_payload = [{'src': "function x(arr) {   if (arr.length === 1) {\n    // return once we hit an array with a single item\n    return arr\n  }\n\n  const middle = Math.floor(arr.length / 2) // get the middle item of the array rounded down\n  const left = arr.slice(0, middle) // items on the left side\n  const right = arr.slice(middle) // items on the right side\n\n    console.log('sorting complete');  return merge(\n    mergeSort(left),\n    mergeSort(right)\n) }", 'augmentations': [	
        {"fn": "rename_variable", "prob": 0.25},
        {"fn": "insert_var_declaration", "prob": 0.25},
        {"fn": "terser", "prob": 0.5, "prob_mangle": 0.1},
        {"fn": "sample_lines", "prob": 0.25, "prob_keep_line": 0.9}
    ]}] * 3
headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
response = requests.post('http://127.0.0.1:3000', data=json.dumps(transform_payload), headers=headers)

print("Status code: ", response.status_code)
print("Printing Entire Post Request")
print(response.json())
