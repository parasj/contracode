from pathlib import Path
import os

from tqdm import tqdm

REMOTE_BASE = "https://people.eecs.berkeley.edu/~paras/datasets/"
SHARED_BASE = Path("/work/paras/data/").resolve()
LOCAL_BASE = (Path(__file__).parent.parent / "data").resolve()


def dl_cmds(dataset_path: str, extract=False):
    remote_path = os.path.join(REMOTE_BASE, dataset_path)
    cache_path = (SHARED_BASE / dataset_path).resolve()
    local_path = (LOCAL_BASE / dataset_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmds = []
    if not local_path.exists():
        if cache_path.exists():
            cmds.append("rsync -avhW --no-compress --progress {} {}".format(cache_path, local_path))
        else:
            cmds.append("wget -nc -O {} {}".format(local_path, remote_path))
        if dataset_path.endswith('.tar.gz') and extract:
            cmds.append("(cd {} && tar -xzf {})".format(local_path.parent, local_path))
        elif dataset_path.endswith('.gz') and extract:
            cmds.append("gunzip -d -k {}".format(local_path))
    return cmds


if __name__ == "__main__":
    cmds = []
    # cmds.extend(dl_cmds("js150k/data.tar.gz"))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_test_0.jsonl.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_valid_0.jsonl.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/csn_unigrams_8k_9995p.tar.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_v2_train_supervised.jsonl.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_train_supervised.jsonl.gz", False))
    cmds.extend(dl_cmds("codesearchnet_javascript/javascript_augmented.pickle.gz", False))

    print("\n".join(cmds))

    t = tqdm(cmds)
    for cmd in t:
        t.set_description("Running command: {}".format(cmd))
        t.refresh()
        os.system(cmd)
