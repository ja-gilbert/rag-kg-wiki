from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# config.yaml is the one place the three approaches disagree, so it is read as
# plain nested dicts and handed to each component as-is: chunk_documents takes
# cfg["chunking"], build_embedder takes cfg["embedding"]. No schema layer in
# between, because a typo should surface as a KeyError naming the key rather
# than as a silently applied default.
DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
