"""History privacy check: a value removed from the current tree is still
found in past versions. Fake values are built at runtime so this file
itself passes the privacy check."""

import os
import subprocess

from avm.ci_checks import history

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def _repo(tmp_path, versions):
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, env=ENV)  # noqa: E731
    git("init", "-b", "main")
    for i, text in enumerate(versions):
        (tmp_path / "notes.txt").write_text(text, encoding="utf-8")
        git("add", "-A")
        git("commit", "-m", f"v{i}")
    return tmp_path


def test_history_finds_value_removed_from_current_tree(tmp_path):
    fake_key = "".join(str(d) for d in (3, 1, 4, 1, 5, 9, 2, 6, 5, 3))
    repo = _repo(tmp_path, [f"parcel {fake_key}\n", "parcel removed\n"])
    hits, n = history(repo)
    assert n == 2
    assert len(hits) == 1 and "tax key" in hits[0] and fake_key not in hits[0]


def test_history_passes_clean_repo(tmp_path):
    repo = _repo(tmp_path, ["no identifiers here\n", "still none; a decimal 0.1234567890 is fine\n"])
    assert history(repo) == ([], 2)
