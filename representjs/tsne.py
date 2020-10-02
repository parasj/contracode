import pickle

import fire
import pandas as pd
import sentencepiece as spm
import torch
import tqdm

from models.code_moco import CodeMoCo
from models.code_mlm import CodeMLM
from representjs import RUN_DIR, CSNJS_DIR, DATA_DIR

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def embed_coco(checkpoint, data_path, spm_filepath=DEFAULT_SPM_UNIGRAM_FILEPATH, encoder_type="transformer", n_encoder_layers=6):
    # read data
    df = pd.read_pickle(data_path)

    # read tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    n_tokens = sp.GetPieceSize()
    pad_id = sp.PieceToId("[PAD]")

    # load model
    model = CodeMoCo(
        n_tokens=n_tokens,
        pad_id=pad_id,
        encoder_config=dict(encoder_type=encoder_type, n_encoder_layers=n_encoder_layers, project="hidden"),
    )
    state = torch.load(checkpoint)
    print(state["model_state_dict"].keys())
    model.load_state_dict(state["model_state_dict"])
    model.cuda()
    model.eval()

    def make_dataset(l):
        return embed_x

    print(df)
    print(data_path)

    out_rows = []
    with torch.no_grad():
        for row_idx in tqdm.tqdm(list(range(len(df))), desc="Table"):
            text = df.loc[row_idx]["code"]
            func_name = df.loc[row_idx]["func_name"]
            x_encoded = torch.LongTensor(sp.EncodeAsIds(text)).cuda()
            lens = torch.LongTensor([len(x_encoded)])
            try:
                embed_x = model.embed_x(x_encoded.unsqueeze(0), lens).cpu().numpy()
                out_rows.append(dict(code=text, func_name=func_name, embedding=embed_x))
            except Exception as e:
                print("Error!", e)
                continue

    tsne_out_path = DATA_DIR / "tsne"
    tsne_out_path.mkdir(parents=True, exist_ok=True)
    print("writing output to ", tsne_out_path.resolve())
    with (tsne_out_path / "tsne_out_embedded_grouped_hidden.pickle").open("wb") as f:
        pickle.dump(out_rows, f)


if __name__ == "__main__":
    fire.Fire({"embed_coco": embed_coco, "embed_bert": embed_bert})
