import argparse
from pathlib import Path
import os
from tqdm import tqdm

LOCAL_JS150K = "/work/paras/data/js150k/data.tar.gz"
REMOTE_JS150K = "https://people.eecs.berkeley.edu/~paras/datasets/js150k/data.tar.gz"
REMOTE_CSNJS = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz"
REMOTE_CSNJS_TEST = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_test_0.jsonl.gz"
REMOTE_CSNJS_VALID = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_valid_0.jsonl.gz"
REMOTE_CSNJS_UNIGRAMS = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/csn_unigrams_8k_9995p.tar.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-js150k", action="store_true", help="Optional, download JS150k dataset and extract")
    args = parser.parse_args()
    cmds = []

    data_dir = Path(__file__).parent.parent / "data"
    if args.download_js150k:
        js150k_base_dir = (data_dir / "js_150k").resolve()
        js150k_base_dir.mkdir(exist_ok=True, parents=True)
        local_cached_path = Path(LOCAL_JS150K)
        if local_cached_path.exists():
            cmds.append("rsync -avhW --no-compress --progress {} {}".format(local_cached_path, str(js150k_base_dir)))
        else:
            cmds.append("wget -nc -P {} {}".format(str(js150k_base_dir), REMOTE_JS150K))
        cmds.append("(cd {} && tar -xzf {})".format(str(js150k_base_dir), str(js150k_base_dir / 'data.tar.gz')))

    csn_base_dir = (data_dir / "codesearchnet_javascript").resolve()
    csn_base_dir.mkdir(exist_ok=True, parents=True)
    cmds.append("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS))
    cmds.append("gunzip -d -k {}".format(str(csn_base_dir / 'javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz')))
    cmds.append("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS_TEST))
    cmds.append("gunzip -d -k {}".format(str(csn_base_dir / 'javascript_test_0.jsonl.gz')))
    cmds.append("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS_VALID))
    cmds.append("gunzip -d -k {}".format(str(csn_base_dir / 'javascript_valid_0.jsonl.gz')))

    cmds.append("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS_UNIGRAMS))
    cmds.append("(cd {} && tar -xzf {} --strip-components=1)".format(str(csn_base_dir), str(csn_base_dir / 'csn_unigrams_8k_9995p.tar.gz')))

    t = tqdm(cmds)
    for cmd in t:
        t.set_description("Running command: {}".format(cmd))
        t.refresh()
        os.system(cmd)
