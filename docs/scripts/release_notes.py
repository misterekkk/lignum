"""Writes the release notes page from CHANGELOG.md, so the notes are kept in one place.

Run before `astro build` (the docs workflow does): python scripts/release_notes.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "src" / "content" / "docs" / "release-notes.md"
REPO = "https://github.com/misterekkk/lignum/blob/main/"


def main() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    body = text.split("\n## ", 1)[1]  # drop the "# Changelog" title and its introduction
    body = "## " + body
    body = re.sub(r"^## \[Unreleased\]\n+(?=## )", "", body)  # nothing unreleased yet: no empty heading
    # Links to files of the repository (SECURITY.md, ...) point to GitHub from the site.
    body = re.sub(r"\]\((?!https?://|#)([^)]+)\)", lambda m: f"]({REPO}{m.group(1)})", body)
    OUT.write_text(
        "---\n"
        "title: Release notes\n"
        "description: What changed in each lignum version.\n"
        "---\n\n"
        "<!-- Generated from CHANGELOG.md by docs/scripts/release_notes.py: edit CHANGELOG.md. -->\n\n"
        "Versions follow [Semantic Versioning](https://semver.org/); before 1.0, minor versions may\n"
        "change the API. Install a specific version with `pip install pylignum==<version>`.\n\n" + body,
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
