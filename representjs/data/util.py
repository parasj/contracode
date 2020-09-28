import re
from typing import Optional, Tuple
import time
from subprocess import Popen, PIPE
from loguru import logger

from representjs import PACKAGE_ROOT


def dispatch_to_node(node_file: str, stdin: Optional[str] = None, timeout_s: int = 5) -> Tuple[bool, str, str]:
    absolute_script_path = str((PACKAGE_ROOT / "node_src" / node_file).resolve())
    p = Popen(["timeout", timeout_s, "node", absolute_script_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    if stdin is not None:
        p.stdin.write(stdin.encode())
    stdout, stderr = p.communicate()
    return_code = p.returncode
    if return_code != 0:
        logger.error("Got non-zero exit code {} for command {}".format(return_code, node_file))
    return (return_code == 0), stdout.decode().strip(), stderr.decode().strip()


class Timer:
    """from https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        logger.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def EncodeAsIds(sp, alpha, prog):
    # Encode as ids with sentencepiece
    if alpha:
        # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
        # NOTE: what is the second argument here (-1)?
        return sp.SampleEncodeAsIds(prog, -1, alpha)

    # using the best decoding
    return sp.EncodeAsIds(prog)
