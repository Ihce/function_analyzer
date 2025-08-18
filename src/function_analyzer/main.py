#!/usr/bin/env python3
# main.py — CLI entrypoint

import argparse
import logging

from agent import BinaryAnalysisAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autonomous binary analysis via GPT-OSS and custom tools.")
    p.add_argument("binary", help="Path to the PE binary to analyze")
    p.add_argument("--boundary-ckpt", help="Path to the boundary detector model checkpoint", default=None)
    p.add_argument("--model-id", help="Model ID (Ollama)", default="gpt-oss:20b")
    p.add_argument("--ollama-url", help="Ollama OpenAI-compatible base URL", default="http://localhost:11434/v1")
    p.add_argument("--max-iters", type=int, default=8, help="Maximum agent iterations")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    agent = BinaryAnalysisAgent(
        ollama_url=args.ollama_url,
        model_id=args.model_id,
        boundary_ckpt=args.boundary_ckpt,
    )

    narrative = agent.analyze(args.binary, max_iters=args.max_iters)
    print("\n=== Analysis ===\n")
    print(narrative)


if __name__ == "__main__":
    main()
