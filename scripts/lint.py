from __future__ import annotations

import argparse
from typing import Sequence

from core.config import DEFAULT_CONFIG_PATH, load_config
from wikigen.lint import LintReport, lint_wiki


def exit_code(report: LintReport) -> int:
    # Non-zero on any error-severity finding, so this can sit in CI next to the
    # tests. Warnings are reported and deliberately do not fail the run.
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check the compiled wiki for incoherence.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    report = lint_wiki(load_config(args.config))
    print(report.render())
    raise SystemExit(exit_code(report))


if __name__ == "__main__":
    main()
