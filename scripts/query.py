"""Query the semantic RAG pipeline.

Usage:
    python scripts/query.py "What is a Pod?"
"""

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from kuberacle.config import load_rag_config
from kuberacle.factory import build_qa_system


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


def parse_args() -> argparse.Namespace:
    """Parse query CLI arguments."""
    parser = argparse.ArgumentParser(description="Ask kuberacle a question")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Retrieval depth")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print retrieval/prompt runtime metadata",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval and answer generation for one question."""
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_rag_config(CONFIG_PATH)

    qa = build_qa_system(config, PROJECT_ROOT)
    result = qa.ask(args.question, top_k=args.top_k)

    print("\nAnswer:\n")
    print(result.answer)
    print("\nCitations:")
    for citation in result.citations:
        print(f"- {citation.source_url} ({citation.chunk_id})")
    if args.verbose:
        print("\nRuntime:")
        print(f"- prompt_version: {config.prompts.version}")
        print("- retrieval_mode: semantic+bm25+rerank")


if __name__ == "__main__":
    main()
