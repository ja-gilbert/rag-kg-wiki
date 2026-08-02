from __future__ import annotations

import datetime as dt
from pathlib import Path

# `wiki/log.md` is a layer in its own right, not a debug aid: it is the record
# of everything that has ever been done to the wiki, and it has more than one
# writer -- compile appends COMPILE, lint appends LINT. Keeping the append here
# is what keeps the format from drifting between them.

_HEADER = "# Log\n\nAppend-only. Newest entries at the bottom.\n\n"


def append(root: Path | str, message: str) -> None:
    log = Path(root) / "log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(_HEADER, encoding="utf-8")
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"- {stamp} {message}\n")
