import pickle

import fire
import sentencepiece as spm
import torch
import tqdm

from models.code_moco import CodeMoCo
from representjs import RUN_DIR, CSNJS_DIR

DEFAULT_CSNJS_TRAIN_FILEPATH = str(CSNJS_DIR / "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz")
DEFAULT_SPM_UNIGRAM_FILEPATH = str(CSNJS_DIR / "csnjs_8k_9995p_unigram_url.model")


def embed_coco(checkpoint, data_path, spm_filepath=DEFAULT_SPM_UNIGRAM_FILEPATH):
    with open(data_path, 'rb') as f:
        matches = pickle.load(f)
    negatives = matches['negatives']
    positive_samples = {k: v for k, v in matches.items() if k != "negatives"}

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    n_tokens = sp.GetPieceSize()
    pad_id = sp.PieceToId("[PAD]")

    model = CodeMoCo(n_tokens=n_tokens, pad_id=pad_id)
    model.load_state_dict(torch.load(checkpoint))
    model.cuda()
    model.eval()

    def make_dataset(l):
        embed_x = [torch.LongTensor(sp.EncodeAsIds(item)).cuda() for item in l]
        return embed_x

    out_matches = {}
    out_negatives = []
    with torch.no_grad():
        for negative in tqdm.tqdm(make_dataset(negatives), desc='negatives'):
            x = negative.unsqueeze(0)
            out_negatives.append(model.embed(x).cpu().numpy())

        for match in positive_samples.keys():
            out_matches[match] = list()
            for positive in make_dataset(match):
                x = positive.unsqueeze(0)
                out_matches[match].append(model.embed(x).cpu().numpy())
    tsne_out_path = (RUN_DIR / 'tsne')
    tsne_out_path.mkdir(parents=True, exist_ok=True)
    with (tsne_out_path / "moco_embed.pickle").open('wb') as f:
        pickle.dump((out_matches, out_negatives), f)


if __name__ == "__main__":
    fire.Fire({"embed_coco": embed_coco})
