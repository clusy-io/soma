"""Make the package importable without depending on an editable install.

A research repo gets its virtualenv re-resolved often, and `uv pip install` can
drop an editable entry as a side effect of pinning something unrelated. Putting
`src/` on the path here means the tests keep running through that, rather than
failing with an import error that looks like a code problem.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT / "src", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
