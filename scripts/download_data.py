import argparse
from pathlib import Path
import os

LOCAL_JS150K = "/work/paras/data/js150k/data.tar.gz"
REMOTE_JS150K = "https://people.eecs.berkeley.edu/~paras/datasets/js150k/data.tar.gz"
REMOTE_CSNJS = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz"
REMOTE_CSNJS_TEST = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_test_0.jsonl.gz"
REMOTE_CSNJS_VALID = "https://people.eecs.berkeley.edu/~paras/datasets/codesearchnet_js/javascript_valid_0.jsonl.gz"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-js150k", action="store_true", help="Optional, download JS150k dataset and extract")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    if args.download_js150k:
        js150k_base_dir = (data_dir / "js_150k").resolve()
        js150k_base_dir.mkdir(exist_ok=True, parents=True)
        local_cached_path = Path(LOCAL_JS150K)
        if local_cached_path.exists():
            cmd = "rsync -avhW --no-compress --progress {} {}".format(local_cached_path, str(js150k_base_dir))
        else:
            cmd = "wget -nc -P {} {}".format(str(js150k_base_dir), REMOTE_JS150K)
        os.system(cmd)
        os.system("tar -xzf {}".format(str(js150k_base_dir / 'data.tar.gz')))

    csn_base_dir = (data_dir / "codesearchnet_javascript").resolve()
    csn_base_dir.mkdir(exist_ok=True, parents=True)
    os.system("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS))
    os.system("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS_TEST))
    os.system("wget -nc -P {} {}".format(str(csn_base_dir), REMOTE_CSNJS_VALID))
    os.system("gunzip {}".format(str(csn_base_dir / 'javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz')))
    os.system("gunzip {}".format(str(csn_base_dir / 'javascript_test_0.jsonl.gz')))
    os.system("gunzip {}".format(str(csn_base_dir / 'javascript_valid_0.jsonl.gz')))
