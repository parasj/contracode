def get_package_root():
    import os
    from pathlib import Path
    return Path(os.path.dirname(os.path.abspath(__file__))) / ".."


PACKAGE_ROOT = get_package_root()
