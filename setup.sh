#!/usr/bin/env bash
# Creates the virtualenv and installs dependencies. Nothing else --
# there is no application to run yet.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then open Claude Code here and read CLAUDE.md."
