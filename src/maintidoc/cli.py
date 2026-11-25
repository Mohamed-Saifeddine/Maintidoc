from __future__ import annotations

import argparse
from pathlib import Path

from maintidoc.agent import DocumentAgent
from maintidoc.config import settings
from maintidoc.indexing import DocumentIndexer


def index_main() -> None:
    parser = argparse.ArgumentParser(description="Indexer un PDF dans MAINTiDOC")
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    def progress(message: str, value: float) -> None:
        print(f"[{value:>5.0%}] {message}")

    stats = DocumentIndexer(settings).index(args.pdf, progress)
    print(stats.model_dump_json(indent=2))


def ask_main() -> None:
    parser = argparse.ArgumentParser(description="Interroger les documents indexés")
    parser.add_argument("question")
    args = parser.parse_args()
    result = DocumentAgent(settings).ask(args.question)
    print(result.answer)
    if result.sources:
        print("\nSources :")
        for source in result.sources:
            print(f"- {source.citation}")
