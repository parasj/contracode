import json
from tqdm import tqdm
import logging
import pickle

def build_aggregated_dataset(index_file):
    print("Loading index")
    with open(index_file) as f:
        index = f.read().split('\n')

    def load_js_ast(js_file):
        full_js_path = 'data/js150k/raw_data/' + js_file
        full_ast_path = full_js_path + '.ast'
        with open(full_ast_path, encoding='latin-1') as f_ast:
            ast = json.load(f_ast)
        with open(full_js_path, encoding='latin-1') as f_js:
            js = f_js.read()
        return {'path': js_file, 'js': js, 'ast': ast}

    print("Loaded index, dispatching futures")
    data_index = []
    for f in tqdm(index):
        try:
            data_index.append(load_js_ast(f))
        except Exception as e:
            logging.exception(e)
    return data_index

eval_dataset = build_aggregated_dataset('data/js150k/programs_eval.txt')
with open('data/js150k/eval_data.pkl', 'wb') as f:
    pickle.dump(eval_dataset, f)

training_dataset = build_aggregated_dataset('data/js150k/programs_training.txt')
with open('data/js150k/train_data.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
