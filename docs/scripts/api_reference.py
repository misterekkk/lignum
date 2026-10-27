"""Writes the Python API reference page from the installed package's signatures and docstrings.

Run before `astro build` (the docs workflow does): python scripts/api_reference.py
"""

from __future__ import annotations

import inspect
import re
import textwrap
from pathlib import Path

import lignum

OUT = Path(__file__).resolve().parent.parent / "src" / "content" / "docs" / "reference" / "python.md"
SECTION = re.compile(r"^(Parameters|Returns|Raises|Examples|Notes)\n-+\n", re.MULTILINE)


def docstring(obj) -> str:
    """numpy-style docstring as Markdown: sections become headings, examples stay code."""
    text = inspect.getdoc(obj) or ""
    text = SECTION.sub(lambda m: f"**{m.group(1)}**\n\n", text)
    lines, out = text.splitlines(), []
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # "name : type" lines, and bare names (exceptions) followed by an indented description,
        # become list items; the indented description joins them.
        m = re.match(r"^(\w[\w ,]*) : (.+)$", line)
        if m:
            out.append(f"- `{m.group(1)}` *({m.group(2)})*")
        elif re.match(r"^\w+$", line) and nxt.startswith("    "):
            out.append(f"- `{line}`")
        elif line.startswith("    ") and out and out[-1].startswith("- `"):
            out[-1] += " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out).strip()


def signature(name: str, obj) -> str:
    try:
        # Annotations are strings (postponed evaluation): unquote them, keep quoted defaults.
        sig = re.sub(r"(: |-> )'([^']*)'", r"\1\2", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("(self, ", "(").replace("(self)", "()")
    return f"```python\n{name}{sig}\n```"


def main() -> None:
    parts = [
        "---",
        "title: Python API",
        "description: Every public function and method of the lignum package.",
        "---",
        "",
        f"Generated from the docstrings of lignum {lignum.__version__} by `docs/scripts/api_reference.py`.",
        "",
    ]
    for name in lignum.__all__:
        if name != "Model":
            function = getattr(lignum, name)
            parts += [f"## `lignum.{name}`", "", signature(f"lignum.{name}", function), "", docstring(function), ""]
    parts += [
        "## `lignum.Model`",
        "",
        docstring(lignum.Model),
        "",
    ]
    for name, member in inspect.getmembers(lignum.Model):
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            parts += [f"### `Model.{name}`", "", "*Property.* " + (inspect.getdoc(member) or ""), ""]
        elif callable(member):
            parts += [f"### `Model.{name}`", "", signature(f"Model.{name}", member), "", docstring(member), ""]
    OUT.write_text("\n".join(parts).rstrip() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
