"""Checks that need no data, run by CI (and locally).

Usage:
    python -m avm.ci_checks dashes    # no em or en dashes in the docs
    python -m avm.ci_checks privacy   # no tax keys, street addresses or owner-field values in tracked files
    python -m avm.ci_checks history   # the same rules on every committed version of every file (all refs)

The privacy check reads every git-tracked text file except reports/prospective/
(the hashed 2026 predictions are keyed by tax key by design). It fails on:
  - a 10-digit tax key (not part of a longer number, a decimal or a hash);
  - a street address (house number, direction, street, suffix);
  - owner-field values: a CSV column named like an owner name or mailing
    address with any value, or a data-dictionary row for a sensitive field
    whose sample values are not redacted.
Synthetic test values are allowed. When the private parcel lists exist
(<data>/reports_private/stage9/, never in CI), every address and tax key in
them is also matched exactly against tracked files.
"""

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

from avm.config import PRIVATE_REPORTS_DIR, REPO_DIR

DOCS = ["README.md", "docs/decision-log.md", "CLAUDE.md", "docs"]
DASHES = ("—", "–")

TAX_KEY_RE = re.compile(r"(?<![0-9A-Za-z.])\d{10}(?![0-9A-Za-z])")
ADDRESS_RE = re.compile(r"\b\d{1,5}(?:-\d{1,5})? [NSEWnsew] [0-9A-Za-z][0-9A-Za-z ]{0,30}? "
                        r"(?:ST|AV|AVE|DR|CT|PL|LA|LN|BL|BLVD|RD|WA|WAY|TR|PK|PKWY|TE|TER|CR|"
                        r"St|Av|Ave|Dr|Ct|Pl|La|Ln|Blvd|Rd|Way|Ter)\b")
OWNER_COL_RE = re.compile(r"(?i)owner.?(name|nm|\d|mail|bill|addr)|^owner$")
SYNTHETIC = {"1234567890", "123 N MAIN ST", "456 S MAIN ST"}  # values used in tests and docstrings
PLACEHOLDERS = {"address not in the city's address index"}  # stage9_divergence fill value, not an address
SKIP_PREFIXES = ("reports/prospective/",)
BINARY_SUFFIXES = {".png", ".jpg", ".parquet", ".pdf", ".zip", ".ots"}


def find_dashes(paths: list[Path]) -> list[str]:
    """'file:line' for every line containing an em or en dash, in .md files
    under each path (a file or a folder)."""
    hits = []
    for p in paths:
        files = sorted(p.rglob("*.md")) if p.is_dir() else [p]
        for f in files:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if any(d in line for d in DASHES):
                    hits.append(f"{f.relative_to(REPO_DIR) if f.is_relative_to(REPO_DIR) else f}:{i}")
    return hits


def scan_text(name: str, text: str, exact: set[str] = frozenset()) -> list[str]:
    """Privacy findings for one file's text ('name:line: kind'). Values are
    never echoed in full, only their kind and a masked hint."""
    hits = []
    found = [v for v in exact if v in text]  # file-level first: cheap
    for i, line in enumerate(text.splitlines(), 1):
        for m in TAX_KEY_RE.findall(line):
            if m not in SYNTHETIC:
                hits.append(f"{name}:{i}: tax key ({m[:3]}*******)")
        for m in ADDRESS_RE.findall(line):
            if m.upper() not in SYNTHETIC:
                hits.append(f"{name}:{i}: street address ({m.split(' ')[0]} ...)")
        if any(v in line for v in found):
            hits.append(f"{name}:{i}: value from the private parcel lists")
    if name.endswith(".csv"):
        hits += _csv_owner_values(name, text)
    return hits


def _csv_owner_values(name: str, text: str) -> list[str]:
    from avm.profile_data import REDACTED, SENSITIVE_FIELD_RE
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header, hits = rows[0], []
    if "/data_dictionary/" in name and {"top_values", "examples"} <= set(header):
        it, ie = header.index("top_values"), header.index("examples")
        for k, r in enumerate(rows[1:], 2):
            if r and SENSITIVE_FIELD_RE.search(r[0]) and any(r[j] not in ("", REDACTED) for j in (it, ie) if j < len(r)):
                hits.append(f"{name}:{k}: unredacted sample values for sensitive field {r[0]}")
        return hits
    for j, col in enumerate(header):
        if OWNER_COL_RE.search(col) and any(j < len(r) and r[j].strip() for r in rows[1:]):
            hits.append(f"{name}: owner-field column '{col}' has values")
    return hits


def private_values() -> set[str]:
    """Addresses and tax keys from the private parcel lists, if present."""
    import pandas as pd
    out = set()
    for f in (PRIVATE_REPORTS_DIR / "stage9").glob("**/*.csv"):
        d = pd.read_csv(f, dtype=str)
        for c in ("tk", "address"):
            if c in d:
                out |= {v for v in d[c].dropna() if v not in SYNTHETIC and v not in PLACEHOLDERS and len(v) >= 8}
    return out


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if not p.startswith(SKIP_PREFIXES)]


def privacy() -> list[str]:
    exact = private_values()
    hits = []
    for rel in tracked_files():
        p = REPO_DIR / rel
        if p.suffix.lower() in BINARY_SUFFIXES or not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += scan_text(rel, text, exact)
    return hits, len(exact)


def history(repo: Path = REPO_DIR) -> tuple[list[str], int]:
    """The privacy rules applied to every text blob reachable from any ref
    (every version of every file ever committed), not only the current tree.
    Needs the full history (CI: fetch-depth 0). Returns (findings, blobs scanned)."""
    git = lambda *a, **k: subprocess.run(["git", *a], cwd=repo, capture_output=True, check=True, **k).stdout  # noqa: E731
    first_path = {}
    for line in git("rev-list", "--all", "--objects").decode().splitlines():
        sha, _, path = line.partition(" ")
        if path:
            first_path.setdefault(sha, path)
    types = git("cat-file", "--batch-check=%(objectname) %(objecttype)", input="\n".join(first_path).encode())
    blobs = [s for s, t in (ln.split() for ln in types.decode().splitlines()) if t == "blob"]
    hits, n = [], 0
    for sha in blobs:
        path = first_path[sha]
        if path.startswith(SKIP_PREFIXES) or Path(path).suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = git("cat-file", "blob", sha).decode("utf-8")
        except UnicodeDecodeError:
            continue
        n += 1
        hits += [f"{h} [blob {sha[:8]}]" for h in scan_text(path, text)]
    return hits, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["dashes", "privacy", "history"])
    a = ap.parse_args()
    if a.check == "history":
        hits, n = history()
        if hits:
            print(f"history check failed: {len(hits)} finding(s) in past versions\n  " + "\n  ".join(hits[:50]))
            sys.exit(1)
        print(f"history check passed: {n:,} text blobs across all refs (reports/prospective/ excluded)")
        return
    if a.check == "dashes":
        hits = find_dashes([REPO_DIR / d for d in DOCS])
        if hits:
            print("em or en dashes found (project rule: none in docs):\n  " + "\n  ".join(hits))
            sys.exit(1)
        print(f"no em or en dashes in {', '.join(DOCS)}")
        return
    hits, n_exact = privacy()
    mode = (f"pattern checks plus exact match of {n_exact:,} private-list values" if n_exact else
            "pattern checks only (private parcel lists not present)")
    if hits:
        print(f"privacy check failed ({mode}): {len(hits)} finding(s)\n  " + "\n  ".join(hits[:50]))
        sys.exit(1)
    print(f"privacy check passed ({mode}) across {len(tracked_files())} tracked files (reports/prospective/ excluded)")


if __name__ == "__main__":
    main()
