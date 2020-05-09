import re
from typing import Optional, Tuple
import time
from subprocess import Popen, PIPE
from loguru import logger

from representjs import PACKAGE_ROOT


def dispatch_to_node(node_file: str, stdin: Optional[str] = None) -> Tuple[str, str]:
    absolute_script_path = str((PACKAGE_ROOT / "node_src" / node_file).resolve())
    p = Popen(['node', absolute_script_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(input=stdin.encode() if stdin is not None else None)
    return stdout.decode().strip(), stderr.decode().strip()


class Timer:
    """from https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


_newline_regex = re.compile(r'\n')
_whitespace_regex = re.compile(r'[ \t\n]+')


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        logger.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r' [EOL]', fn)
    fn = _whitespace_regex.sub(' ', fn)
    return fn