from pathlib import Path
import os

from tqdm import tqdm

LOCAL_BASE = (Path(__file__).parent.parent / "data").resolve()


if __name__ == "__main__":
    cmds = []
    t = tqdm(cmds)
    for cmd in t:
        t.set_description("Running command: {}".format(cmd))
        t.refresh()
        os.system(cmd)
