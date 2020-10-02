import argparse
import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process code into pre-trained tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--text_file_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--out_name", type=str)
    args = parser.parse_args()

    paths = [str(x) for x in Path(args.text_file_path).glob("**/*")]
    tokenizer = BertWordPieceTokenizer(clean_text=True, lowercase=False, strip_accents=True)
    tokenizer.train(
        files=paths,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<cls>", "<pad>", "<unk>", "<mask>", "[CLS]", "[SEP]", "[MASK]", "[EOL]", "[URL]", "[PAD]", "[UNK]"],
    )
    tokenizer.save_model(args.out_path, args.out_name)
    tokenizer.save(os.path.join(args.out_path, "vocab.json"))
