#!/usr/bin/env python3
"""Download a public-domain held-out text for offline perplexity and needle tests."""

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "holdout.txt"

SOURCES = [
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
]
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _strip_boilerplate(text):
    start = text.find(START_MARKER)
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find(END_MARKER)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def download():
    last_error = None
    for url in SOURCES:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="ignore")
            return _strip_boilerplate(raw)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download held-out text: {last_error}")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = download()
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} characters to {OUT_PATH}")
    print("This file is gitignored and never committed.")


if __name__ == "__main__":
    sys.exit(main())
