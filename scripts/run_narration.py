"""
run_narration.py — CLI entry point for the ChainSight narration layer.
Usage:
    python scripts\\run_narration.py --rule-events outputs\\rule_events.json --out outputs\\narration.json

Requires GEMINI_API_KEY set in the environment or a .env file (see .env.example).
"""

import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.narration import GeminiClient, GeminiClientConfig, Narrator

logger = logging.getLogger("chainsight.narration.cli")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="ChainSight Narration Layer (Gemini)")
    parser.add_argument("--rule-events", required=True, help="Path to rule_events.json from run_rules.py")
    parser.add_argument("--out", default="outputs/narration.json", help="Path to save narration JSON")
    parser.add_argument("--model", default="gemini-flash-latest", help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    client_config = GeminiClientConfig(model=args.model, temperature=args.temperature)

    try:
        client = GeminiClient(client_config)
        narrator = Narrator(client)
        result = narrator.run(args.rule_events)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Narration failed: {e}")
        raise SystemExit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved narration ({len(result['events'])} event(s) + summary) to {args.out}")


if __name__ == "__main__":
    main()
