"""Check the configured model provider answers: `python scripts/llm_smoke.py [--provider github]`."""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itg.llm import chat, load_settings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["foundry", "github"], help="override ITG_LLM_PROVIDER")
    args = ap.parse_args()
    if args.provider:
        os.environ["ITG_LLM_PROVIDER"] = args.provider
    s = load_settings()
    print(s)
    t0 = time.perf_counter()
    reply = chat([{"role": "system", "content": "You are a football commentator. One short sentence."},
                  {"role": "user", "content": "Say the line-ups are in and kick-off is close."}], s, max_tokens=60)
    print(f"{time.perf_counter() - t0:.1f}s: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
