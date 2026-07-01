"""
Quick check: OpenAI API key is valid and reachable.
Usage:
    python src/test_openai.py
    python src/test_openai.py --model gpt-4o-mini
"""
import argparse, os, sys, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY or pass --api_key")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed — run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Testing model: {args.model}")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            temperature=0,
        )
        reply = resp.choices[0].message.content.strip()
        elapsed = time.time() - t0
        print(f"Response: '{reply}'  ({elapsed:.2f}s)")
        print(f"Tokens used: prompt={resp.usage.prompt_tokens}  "
              f"completion={resp.usage.completion_tokens}")
        print("✓ OpenAI API connection OK")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
