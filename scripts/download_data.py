import argparse
from pathlib import Path
import os

from tqdm import tqdm

REMOTE_BASE = "https://contrastive-code.s3.amazonaws.com"  # "https://people.eecs.berkeley.edu/~paras/datasets"
SHARED_BASE = Path("/work/paras/contracode/data").resolve()
DEFAULT_LOCAL_BASE = str((Path(__file__).parent.parent / "data").resolve())


def dl_cmds(dataset_path: str, extract=False, LOCAL_BASE=DEFAULT_LOCAL_BASE):
    remote_path = os.path.join(REMOTE_BASE, dataset_path)
    cache_path = (SHARED_BASE / dataset_path).resolve()
    local_path = (Path(LOCAL_BASE) / dataset_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmds = []
    if not local_path.exists():
        if cache_path.exists():
            cmds.append("rsync -avhW --no-compress --progress {} {}".format(cache_path, local_path))
        else:
            cmds.append("wget -nc -O {} {}".format(local_path, remote_path))
        if dataset_path.endswith(".tar.gz") and extract:
            cmds.append("(cd {} && tar -xzf {})".format(local_path.parent, local_path))
        elif dataset_path.endswith(".gz") and extract:
            cmds.append("gunzip -d -k {}".format(local_path))
    return cmds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ContraCode data")
    parser.add_argument("--path", type=str, default=DEFAULT_LOCAL_BASE, help="Path to save output to")
    parser.add_argument("--skip-csn", action="store_true")
    parser.add_argument("--download-hf", action="store_true")  # skipped by default as it is large
    parser.add_argument("--skip-type-prediction", action="store_true")
    parser.add_argument("--skip-code-clone", action="store_true")
    args = parser.parse_args()

    LOCAL_PATH = Path(args.path)

    cmds = []
    if args.download_hf:
        cmds.extend(dl_cmds("hf_data/feather_tok/feather_tok.tar.gz", True, LOCAL_PATH))

    if not args.skip_csn:
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_test_0.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_valid_0.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/csn_unigrams_8k_9995p.tar.gz", True, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_v2_train_supervised.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_train_supervised.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("codesearchnet_javascript/javascript_augmented.pickle.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("augmented_data/augmented_minus_compression.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("augmented_data/augmented_minus_identifier.jsonl.gz", False, LOCAL_PATH))
        cmds.extend(dl_cmds("augmented_data/augmented_minus_line_subsampling.jsonl.gz", False, LOCAL_PATH))

    if not args.skip_type_prediction:
        cmds.extend(dl_cmds("type_prediction/test_projects_gold_filtered.json", False, LOCAL_PATH))
        cmds.extend(dl_cmds("type_prediction/target_wl", False, LOCAL_PATH))
        cmds.extend(dl_cmds("type_prediction/csnjs_8k_9995p_unigram_url.model", False, LOCAL_PATH))
        cmds.extend(dl_cmds("type_prediction/train_nounk.txt", False, LOCAL_PATH))
        cmds.extend(dl_cmds("type_prediction/valid_nounk.txt", False, LOCAL_PATH))

    if not args.skip_code_clone:
        cmds.extend(dl_cmds("codeclone/full_data.json.gz", True, LOCAL_PATH))

    cmds.extend(dl_cmds("vocab/8k_bpe/8k_bpe-vocab.txt", False, LOCAL_PATH))

    print("\n".join(cmds))

    t = tqdm(cmds)
    for cmd in t:
        t.set_description("Running command: {}".format(cmd))
        t.refresh()
        os.system(cmd)
