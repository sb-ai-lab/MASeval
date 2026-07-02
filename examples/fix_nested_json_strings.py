"""Recursively parse JSON strings nested inside trace JSON files.

Recursively walks through all dict/list values and replaces any string
that parses as valid JSON (dict or list) with the parsed object.

Usage:
    python fix_nested_json_strings.py [directories...]

If no directories given, defaults to:
    data/exgentic_agent_llm_traces_filtered
    data/swebench_minimax_traces
    nlile_n4plus
    large_mas_traces
    small_mas_traces

Dry run:
    python fix_nested_json_strings.py --dry-run
"""

import json
import sys
from pathlib import Path

DEFAULT_DIRS = [
    "data/exgentic_agent_llm_traces_filtered",
    "data/swebench_minimax_traces",
    "nlile_n4plus",
    "large_mas_traces",
    "small_mas_traces",
]


def parse_nested_strings(obj, max_depth=10):
    if max_depth <= 0:
        return obj
    if isinstance(obj, dict):
        return {k: parse_nested_strings(v, max_depth - 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_nested_strings(item, max_depth - 1) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed = json.loads(obj)
                return parse_nested_strings(parsed, max_depth - 1)
            except (json.JSONDecodeError, RecursionError):
                return obj
        return obj
    else:
        return obj


def process_file(fpath, dry_run=False):
    with open(fpath) as f:
        data = json.load(f)

    fixed = parse_nested_strings(data)

    if fixed is data:
        return False

    if dry_run:
        return True

    with open(fpath, "w") as f:
        json.dump(fixed, f, indent=2, ensure_ascii=False)
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not dirs:
        base = Path(__file__).resolve().parent
        dirs = [str(base / d) for d in DEFAULT_DIRS]

    total = 0
    fixed_count = 0
    skipped = 0

    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  SKIP {d}: not found")
            continue
        files = sorted(d.glob("*.json"))
        print(f"  {d}: {len(files)} files")
        for fpath in files:
            total += 1
            try:
                changed = process_file(fpath, dry_run=dry_run)
                if changed:
                    fixed_count += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"    ERROR {fpath.name}: {e}")

    action = "WOULD FIX" if dry_run else "FIXED"
    print(f"\n{action}: {fixed_count}/{total}, unchanged: {skipped}")
    if dry_run:
        print("(dry-run mode, no files were modified)")


if __name__ == "__main__":
    main()