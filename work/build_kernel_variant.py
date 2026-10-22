"""Build a kernel variant from the canonical train.py by overriding module constants.

Keeps one source of truth: the canonical kernel is the submission-producing notebook,
and experiment arms (different maxlen / weights / loss) are generated copies with a
different constant block and their own kernel id.

Usage:
    python work/build_kernel_variant.py --variant-slug rmit-sgs-ae-do-toc-c1-m512 \
        --title RMIT_SGS_Ae-do-toc_C1_m512 --set MAXLEN=512
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

CANONICAL_DIR = Path("kernels/rmit-sgs-ae-do-toc-c1")
KERNELS_DIR = Path("kernels")
# Constants that may be overridden, with the regex-safe assignment form expected in the source.
ALLOWED = {"MAXLEN", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "WEIGHT_PRESET", "LOSS_MODE", "RUN",
           "USE_LANG_PREFIX", "FOLDS_TO_RUN"}


def patch_constant(source: str, key: str, value: str) -> str:
    """Replace a module-level constant assignment by line number (AST-located).

    Pattern matching was unsafe: the module docstring documents the RUN values as
    ``RUN="full" : ...``, so a regex anchored on ``^RUN`` rewrote the docstring and
    left the real assignment untouched.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SystemExit(f"variant source is not valid Python: {error}") from error

    target_line: int | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if key in names:
                target_line = node.lineno
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == key:
                target_line = node.lineno
    if target_line is None:
        raise SystemExit(f"constant {key!r} not found at module level in train.py")

    lines = source.splitlines()
    lines[target_line - 1] = f"{key} = {value}"
    rebuilt = "\n".join(lines)
    return rebuilt + "\n" if source.endswith("\n") else rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    overrides: dict[str, str] = {}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in ALLOWED:
            raise SystemExit(f"{key!r} is not an overridable constant; allowed: {sorted(ALLOWED)}")
        overrides[key] = value

    source = (CANONICAL_DIR / "train.py").read_text(encoding="utf-8")
    for key, value in overrides.items():
        source = patch_constant(source, key, value)

    target = KERNELS_DIR / args.variant_slug
    target.mkdir(parents=True, exist_ok=True)
    try:
        (target / "train.py").write_text(source, encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"could not write variant source: {error}") from error

    try:
        canonical_meta = json.loads((CANONICAL_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read canonical metadata: {error}") from error
    metadata = dict(canonical_meta)
    metadata["id"] = f"tkhoikk/{args.variant_slug}"
    metadata["title"] = args.title
    try:
        (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"could not write variant metadata: {error}") from error

    print(f"built {target} with overrides {overrides}")
    print(f"  id={metadata['id']} title={metadata['title']}")


if __name__ == "__main__":
    main()
