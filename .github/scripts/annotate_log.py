"""Prints the failing part of a log as a GitHub error annotation: job logs need a login, while
annotations can be read through the public API (check-runs/<job id>/annotations).

Usage: python annotate_log.py <title> <log file>
"""

import re
import sys

FAILURE = re.compile(r"fail|error|mismatch|differ|traceback|exception|assert|outside the AVX2 kernel|Expected|Which is|Actual",
                     re.IGNORECASE)


def main() -> None:
    title, path = sys.argv[1], sys.argv[2]
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError as e:
        lines = [f"no log: {e}"]
    picked = [line for line in lines if FAILURE.search(line)][-40:]
    text = "\n".join(picked + ["--- last lines ---"] + lines[-40:])[-4000:]
    print(f"::error title={title}::" + text.replace("%", "%25").replace("\r", "").replace("\n", "%0A"))


if __name__ == "__main__":
    main()
