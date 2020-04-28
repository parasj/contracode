from typing import Optional, Tuple

import representjs

from subprocess import Popen, PIPE


def dispatch_to_node(node_file: str, stdin: Optional[str] = None) -> Tuple[str, str]:
    absolute_script_path = str((representjs.PACKAGE_ROOT / "node_src" / node_file).resolve())
    p = Popen(['node', absolute_script_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(input=stdin.encode() if stdin is not None else None)
    return stdout.decode().strip(), stderr.decode().strip()
