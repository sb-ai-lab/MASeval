from __future__ import annotations

from maseval.validators import run_on_dir

if __name__ == "__main__":
    trace_dir = r"C:\Users\twcwk\Desktop\RRR\maseval-research\test_traces"
    out_dir = r"C:\Users\twcwk\Desktop\RRR\maseval-research\test_results\validators"
    run_on_dir(trace_dir, out_dir)
