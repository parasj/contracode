import fire
import sentencepiece as spm
import tqdm

from data.jsonl_dataset import JSONLinesDataset, normalize_docstring
from data.util import normalize_program

DEFAULT_INPUT = "data/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl"
DEFAULT_OUTPUT = "data/codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.txt"


def make_corpus(input=DEFAULT_INPUT, output=DEFAULT_OUTPUT):
    dataset = JSONLinesDataset(input, {"function": "function", "docstring": "docstring"})
    print("Number of functions:", len(dataset))
    print("Example original:", dataset[0]["function"])
    print("Example normalized:", normalize_program(dataset[0]["function"]))
    print("Example normalized docstring:", normalize_docstring(dataset[0]["docstring"]))

    with open(output, "w") as f:
        for ex in tqdm.tqdm(dataset, "Writing corpus to txt"):
            # Write docstring
            if ex["docstring"]:
                f.write(normalize_docstring(ex["docstring"]) + "\n")
            # Write normalized function
            function = ex["function"]
            line = normalize_program(function)
            f.write(line + "\n")

    print("Wrote corpus to:", output)


def spm_train(
    input: str, model_prefix: str, vocab_size: int, character_coverage: float, model_type: str
):  # , input_sentence_size: int, shuffle_input_sentence: str):
    # command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={model_type} --input_sentence_size={input_sentence_size} --shuffle_input_sentence={shuffle_input_sentence}"
    command = f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={model_type} --unk_piece=[UNK] --pad_piece=[PAD] --user_defined_symbols=[CLS],[SEP],[MASK],[EOL],[URL] --hard_vocab_limit=false"
    print(command)
    spm.SentencePieceTrainer.Train(command)


if __name__ == "__main__":
    fire.Fire({"make_corpus": make_corpus, "spm_train": spm_train})
