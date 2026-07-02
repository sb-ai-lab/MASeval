"""Build a compact diagnostic report from a saved findings+evidence JSON file.

Example:
    python examples/who_and_when/build_diagnostic_report.py \
      examples/who_and_when/who\&when_hand_gemini_findings_evidence_v7/gemini_findings_0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maseval.reporting import build_report_from_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to a saved findings/evidence JSON file")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write the report. Default: <input>.report.json",
    )
    parser.add_argument("--predicted-answer", default=None)
    parser.add_argument("--reference-answer", default=None)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = args.output_json or str(input_path.with_suffix(".report.json"))
    report = build_report_from_file(
        str(input_path),
        output_path=output_path,
        predicted_answer=args.predicted_answer,
        reference_answer=args.reference_answer,
    )
    print(json.dumps(report["status"], ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
