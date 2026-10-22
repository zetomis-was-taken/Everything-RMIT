"""Generate a shareable Kaggle logbook notebook from STATUS.md.

STATUS.md stays the single source of truth (it lives in git and gets reviewed);
this script mirrors it into a private notebook so the team can read progress,
thinking and decisions on Kaggle without cloning the repo.

Usage:
    python work/make_logbook.py
    cd kernels/rmit-sgs-ae-do-toc-c1-logbook && kaggle kernels push
"""

from __future__ import annotations

import json
from pathlib import Path

STATUS_PATH = Path("STATUS.md")
TARGET_DIR = Path("kernels/rmit-sgs-ae-do-toc-c1-logbook")
KERNEL_SLUG = "rmit-sgs-ae-do-toc-c1-logbook"
TITLE = "RMIT_SGS_Ae-do-toc_C1_logbook"


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def split_sections(text: str) -> list[str]:
    """One cell per '## ' section, keeping the preamble as its own cell."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section.strip()]


def build_notebook(status_text: str) -> dict:
    cells = [markdown_cell(section) for section in split_sections(status_text)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata() -> dict:
    return {
        "id": f"tkhoikk/{KERNEL_SLUG}",
        "title": TITLE,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rmit-hackathon-2026"],
    }


def main() -> None:
    try:
        status_text = STATUS_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"could not read {STATUS_PATH}: {error}") from error

    notebook = build_notebook(status_text)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (TARGET_DIR / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        (TARGET_DIR / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"could not write logbook: {error}") from error

    print(f"generated {TARGET_DIR / (KERNEL_SLUG + '.ipynb')} with {len(notebook['cells'])} markdown cells")
    print(f"push with: cd {TARGET_DIR} && kaggle kernels push")


if __name__ == "__main__":
    main()
