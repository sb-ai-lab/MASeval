"""Review LLM error annotations for MAS traces.

Usage:
    streamlit run examples/review_app.py

Folders are selectable from FOLDERS config below.
Review progress is saved per reviewer per folder: review_<folder>_<reviewer>.json
Status is merged from ALL reviewers so you see what's already been reviewed.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

FOLDERS = [
    {
        "name": "NoNameMAS SWEBench Traces",
        "traces": "/home/alina/Desktop/maseval-research/data/swebench_minimax_traces",
        "annotations": "/home/alina/Desktop/maseval-research/error_annotations_swebench",
    },
    {
        "name": "Exgentic Traces",
        "traces": "/home/alina/Desktop/maseval-research/data/exgentic_agent_llm_traces_filtered",
        "annotations": "/home/alina/Desktop/maseval-research/error_annotations_exgentic",
    },
    # {
    #     "name": "NoNameMAS WebArena Traces",
    #     "traces": "GHOST_dataset/webarena_traces_reddit",
    #     "annotations": "./webarena_anno",
    # },
    {
        "name": "nlile_n4plus Traces",
        "traces": "nlile_n4plus",
        "annotations": "examples/minimax-m3-nlile",
    },
]

ERROR_CATEGORIES = [
    "reasoning_planning",
    "hallucination",
    "instruction_following",
    "tool_calling",
    "mas_coordination",
    "context_state",
    "verification_termination",
    "environmental",
    "api_system",
    "ideal",
]

CATEGORY_HELP = {
    "reasoning_planning": "Flawed plans, incorrect problem identification, goal drift, ignored constraints",
    "hallucination": "Content not grounded in any input or tool output — confabulation, invented facts",
    "instruction_following": "Ignores/misreads directives: under-execution, over-execution, outside role",
    "tool_calling": "Malformed calls, wrong tool selection, misread tool output, unsatisfiable intent",
    "mas_coordination": "Breakdowns in communication/handoff across agents, context resets, withheld info",
    "context_state": "Losing prior context, repeating steps, incorrect model of current task state",
    "verification_termination": "Stopping too early, failing to recognize completion, redundant actions",
    "environmental": "Guardrails, missing resources, misconfigured environments, auth failures",
    "api_system": "Timeouts, rate limits, downstream service errors, infrastructure failures",
    "ideal": "Error-free across every dimension — the no-error reference class",
}

STATUS_ICON = {"unreviewed": "\u2b1c", "approved": "\u2705", "edited": "\u270f\ufe0f", "rejected": "\u274c"}

BASE_DIR = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_review(path: Path) -> dict:
    data = load_json(path)
    return data if data else {}


def save_review(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_trace_files(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("*.json"))


def review_filepath(folder_name: str, reviewer: str) -> Path:
    safe_reviewer = reviewer.strip().casefold().replace(" ", "_")
    safe_folder = folder_name.lower().replace(" ", "_")
    return BASE_DIR / f"review_{safe_folder}_{safe_reviewer}.json"


def review_glob_pattern(folder_name: str) -> str:
    safe_folder = folder_name.lower().replace(" ", "_")
    return f"review_{safe_folder}_*.json"


def load_all_reviews(folder_name: str, base_dir: Path) -> dict:
    merged = {}
    pattern = review_glob_pattern(folder_name)
    for rf in sorted(base_dir.glob(pattern)):
        rd = load_json(rf) or {}
        reviewer_name = rf.stem.split("_")[-1]
        for trace_name, entry in rd.items():
            if trace_name not in merged or entry.get("status", "unreviewed") != "unreviewed":
                merged[trace_name] = dict(entry)
                merged[trace_name]["_source_reviewer"] = reviewer_name
    return merged


def has_annotation(trace_file: Path, anno_dir: Path) -> bool:
    return (anno_dir / trace_file.name).exists()


def main():
    st.set_page_config(page_title="MAS Annotation Review", layout="wide")

    # ── Init session state ──────────────────────────────────────────────
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "editing_json" not in st.session_state:
        st.session_state.editing_json = None
    if "prev_trace_stem" not in st.session_state:
        st.session_state.prev_trace_stem = None
    if "folder_idx" not in st.session_state:
        st.session_state.folder_idx = 0

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        folder_names = [f["name"] for f in FOLDERS]
        prev_folder = st.session_state.folder_idx
        selected_folder = st.selectbox("Dataset", folder_names, index=prev_folder)
        st.session_state.folder_idx = folder_names.index(selected_folder)
        if st.session_state.folder_idx != prev_folder:
            st.session_state.current_idx = 0
            st.session_state.editing_json = None
            st.session_state.prev_trace_stem = None

        folder = FOLDERS[st.session_state.folder_idx]
        trace_dir = BASE_DIR / folder["traces"]
        anno_dir = BASE_DIR / folder["annotations"]

        if not trace_dir.exists():
            st.error(f"Trace dir not found: {trace_dir}")
            st.stop()

        anno_dir.mkdir(parents=True, exist_ok=True)

        reviewer = st.text_input("Reviewer name", key="reviewer_val")
        review_path = review_filepath(folder["name"], reviewer) if reviewer.strip() else None

        trace_files = get_trace_files(trace_dir)
        if not trace_files:
            st.error(f"No trace files in {trace_dir}")
            st.stop()

        # ── Load reviews: current user + all others (merged) ─────────────
        my_reviews = load_review(review_path) if review_path else {}
        all_reviews = load_all_reviews(folder["name"], BASE_DIR)
        # Current reviewer's entries take priority
        for k, v in my_reviews.items():
            if v.get("status", "unreviewed") != "unreviewed":
                all_reviews[k] = dict(v)
                all_reviews[k]["_source_reviewer"] = "you"

        # ── Annotated only toggle ──────────────────────────────────────────
        anno_only = st.toggle("Only annotated traces", value=False)

        # ── Progress ──────────────────────────────────────────────────────
        counts = {"unreviewed": 0, "approved": 0, "edited": 0, "rejected": 0}
        my_counts = {"unreviewed": 0, "approved": 0, "edited": 0, "rejected": 0}
        annotated_count = 0
        for tf in trace_files:
            s = all_reviews.get(tf.name, {}).get("status", "unreviewed")
            counts[s] = counts.get(s, 0) + 1
            ms = my_reviews.get(tf.name, {}).get("status", "unreviewed")
            my_counts[ms] = my_counts.get(ms, 0) + 1
            if has_annotation(tf, anno_dir):
                annotated_count += 1
        reviewed_total = counts["approved"] + counts["edited"] + counts["rejected"]
        my_reviewed = my_counts["approved"] + my_counts["edited"] + my_counts["rejected"]
        st.markdown(
            f"**You:** {my_reviewed} / {len(trace_files)}  \n"
            f"**All reviewers:** {reviewed_total} / {len(trace_files)}  \n"
            f"\u2705 {counts['approved']}  \u270f\ufe0f {counts['edited']}  "
            f"\u274c {counts['rejected']}  \u2b1c {counts['unreviewed']}  \n"
            f"Annotated: {annotated_count} / {len(trace_files)}"
        )

        # ── Other reviewers ────────────────────────────────────────────────
        pattern = review_glob_pattern(folder["name"])
        existing_files = sorted(BASE_DIR.glob(pattern))
        if existing_files:
            with st.expander(f"Reviewers ({len(existing_files)})"):
                for rf in existing_files:
                    name_parts = rf.stem.split("_")
                    rev_name = name_parts[-1] if name_parts else "?"
                    rd = load_json(rf) or {}
                    done = sum(1 for v in rd.values() if v.get("status") != "unreviewed")
                    total = len(rd)
                    is_you = " \u2190 you" if review_path and rf.name == review_path.name else ""
                    st.text(f"{rev_name}: {done}/{total}{is_you}")

        # ── Filter ────────────────────────────────────────────────────────
        filter_opt = st.radio("Filter", ["All", "Unreviewed", "Approved", "Edited", "Rejected"], index=0, horizontal=True)
        filter_map = {
            "All": None,
            "Unreviewed": "unreviewed",
            "Approved": "approved",
            "Edited": "edited",
            "Rejected": "rejected",
        }

        filtered_indices = []
        for i, tf in enumerate(trace_files):
            s = all_reviews.get(tf.name, {}).get("status", "unreviewed")
            status_match = filter_map[filter_opt] is None or s == filter_map[filter_opt]
            anno_match = (not anno_only) or has_annotation(tf, anno_dir)
            if status_match and anno_match:
                filtered_indices.append(i)

        # ── Navigation ────────────────────────────────────────────────────
        clamped_idx = min(st.session_state.current_idx, len(trace_files) - 1)
        st.session_state.current_idx = clamped_idx

        trace_labels = []
        for i in filtered_indices:
            tf = trace_files[i]
            s = all_reviews.get(tf.name, {}).get("status", "unreviewed")
            icon = STATUS_ICON.get(s, "\u2b1c")
            source = all_reviews.get(tf.name, {}).get("_source_reviewer", "")
            if s != "unreviewed" and source and source != "you":
                source_tag = f" ({source})"
            else:
                source_tag = ""
            no_anno = not has_annotation(tf, anno_dir)
            tag = " \u26a0no anno" if no_anno else ""
            short_name = tf.stem[:20] + ("..." if len(tf.stem) > 20 else "")
            trace_labels.append(f"{icon} {short_name}{source_tag}{tag}")

        if not trace_labels:
            st.info("No traces match the filter.")
        else:
            default_pos = next(
                (p for p, i in enumerate(filtered_indices) if i == clamped_idx),
                0,
            )
            if default_pos >= len(trace_labels):
                default_pos = 0

            selected_pos = st.selectbox(
                "Navigate",
                list(range(len(trace_labels))),
                format_func=lambda x: trace_labels[x],
                index=default_pos,
            )
            st.session_state.current_idx = filtered_indices[selected_pos]

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("\u2b05 Prev", use_container_width=True):
                cur_pos = next((p for p, i in enumerate(filtered_indices) if i == st.session_state.current_idx), None)
                if cur_pos is not None and cur_pos > 0:
                    st.session_state.current_idx = filtered_indices[cur_pos - 1]
                    st.session_state.editing_json = None
                    st.session_state.prev_trace_stem = None
                    st.rerun()
        with nav_col2:
            if st.button("Next \u27a1", use_container_width=True):
                cur_pos = next((p for p, i in enumerate(filtered_indices) if i == st.session_state.current_idx), None)
                if cur_pos is not None and cur_pos < len(filtered_indices) - 1:
                    st.session_state.current_idx = filtered_indices[cur_pos + 1]
                    st.session_state.editing_json = None
                    st.session_state.prev_trace_stem = None
                    st.rerun()

    idx = min(st.session_state.current_idx, len(trace_files) - 1)
    trace_file = trace_files[idx]

    # Reset editing JSON when trace changes
    if trace_file.stem != st.session_state.prev_trace_stem:
        st.session_state.editing_json = None
        st.session_state.prev_trace_stem = trace_file.stem

    trace_data = load_json(trace_file) or {}
    anno_data = load_json(anno_dir / trace_file.name)

    # Get status from merged reviews, but prefer current user's own review
    my_review = my_reviews.get(trace_file.name, {})
    merged_review = all_reviews.get(trace_file.name, {})
    effective_review = my_review if my_review.get("status", "unreviewed") != "unreviewed" else merged_review
    current_status = effective_review.get("status", "unreviewed")

    status_badge = {
        "unreviewed": "\u2b1c Unreviewed",
        "approved": "\u2705 Approved",
        "edited": "\u270f\ufe0f Edited",
        "rejected": "\u274c Rejected",
    }
    status_text = status_badge.get(current_status, current_status)
    if not has_annotation(trace_file, anno_dir):
        status_text += "  \u26a0 No annotation"

    # Show who reviewed it (if not current user)
    source_reviewer = merged_review.get("_source_reviewer", "")
    if current_status != "unreviewed":
        if my_review.get("status", "unreviewed") != "unreviewed":
            status_text += f"  (by you)"
        elif source_reviewer:
            status_text += f"  (by {source_reviewer})"

    st.header(f"Trace: `{trace_file.name}`")
    st.caption(f"Status: {status_text}")

    trace_col, anno_col = st.columns([3, 2])

    with trace_col:
        st.subheader("Trace")
        st.json(trace_data, expanded=True)

    with anno_col:
        st.subheader("Annotation")

        if anno_data is None:
            st.warning("No annotation file found for this trace.")
            editing_text = "{\n  \"errors\": []\n}"
            model_info = None
        else:
            model_info = f"Model: **{anno_data.get('model', '?')}**  |  Timestamp: {anno_data.get('timestamp', '?')}"
            if st.session_state.editing_json is None:
                editing_text = json.dumps(anno_data, indent=2, ensure_ascii=False)
            else:
                editing_text = st.session_state.editing_json

        if model_info:
            st.caption(model_info)

        # ── Format help ────────────────────────────────────────────────
        with st.expander("Format & categories", expanded=False):
            st.markdown("**Required format:**")
            st.code('{\n  "errors": [\n    {\n      "error_key": "<category>",\n      "step": <int>,\n      "justification": "<text>"\n    }\n  ]\n}', language="json")
            st.markdown("**Rules:**")
            st.markdown("- Each error category may appear **at most once**")
            st.markdown('- "ideal" is mutually exclusive with all other categories')
            st.markdown("- `step` is 0-indexed; use `-1` for \"ideal\"")
            st.markdown("")
            st.markdown("**Categories:**")
            for cat, desc in CATEGORY_HELP.items():
                st.markdown(f"- **{cat}** — {desc}")

        edited_json = st.text_area(
            "Annotation JSON",
            value=editing_text,
            height=400,
        )
        st.session_state.editing_json = edited_json

        parse_ok = True
        validation_warnings = []
        parsed = None
        try:
            parsed = json.loads(edited_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            parse_ok = False

        if parse_ok and parsed is not None:
            if not isinstance(parsed, dict):
                validation_warnings.append("Top-level value must be a JSON object (dict)")
            elif "errors" not in parsed:
                validation_warnings.append("Missing required key: \"errors\"")
            elif not isinstance(parsed["errors"], list):
                validation_warnings.append("\"errors\" must be a list")
            else:
                seen_keys = set()
                for i, err in enumerate(parsed["errors"]):
                    prefix = f"errors[{i}]"
                    if not isinstance(err, dict):
                        validation_warnings.append(f"{prefix}: must be a dict")
                        continue
                    if "error_key" not in err:
                        validation_warnings.append(f"{prefix}: missing \"error_key\"")
                    elif err["error_key"] not in ERROR_CATEGORIES:
                        validation_warnings.append(f"{prefix}: unknown category \"{err['error_key']}\"")
                    else:
                        if err["error_key"] in seen_keys:
                            validation_warnings.append(f"{prefix}: duplicate category \"{err['error_key']}\"")
                        seen_keys.add(err["error_key"])
                    if "step" not in err:
                        validation_warnings.append(f"{prefix}: missing \"step\"")
                    elif not isinstance(err["step"], int):
                        validation_warnings.append(f"{prefix}: \"step\" must be an integer")
                    if "justification" not in err:
                        validation_warnings.append(f"{prefix}: missing \"justification\"")
                    elif not isinstance(err["justification"], str):
                        validation_warnings.append(f"{prefix}: \"justification\" must be a string")
                if "ideal" in seen_keys and len(seen_keys) > 1:
                    validation_warnings.append("\"ideal\" cannot coexist with other error categories")

        if not parse_ok:
            pass  # hard JSON error, no submit
        elif validation_warnings:
            for vw in validation_warnings:
                st.warning(f"\u26a0 {vw}")

        st.divider()

        if not reviewer.strip():
            st.warning("\u26a0 Enter your reviewer name in the sidebar to submit.")
        else:
            # Pre-select verdict if current user already reviewed this trace
            existing_my_status = my_review.get("status", "")
            default_verdict_idx = 0
            if existing_my_status == "approved":
                default_verdict_idx = 1
            elif existing_my_status == "edited":
                default_verdict_idx = 2
            elif existing_my_status == "rejected":
                default_verdict_idx = 3

            decision = st.radio(
                "Verdict",
                ["", "\u2705 Done", "\u270f\ufe0f Done after correction", "\u274c Reject"],
                index=default_verdict_idx,
                horizontal=True,
            )

            if decision and parse_ok:
                submit = st.button("Submit", type="primary", use_container_width=True)
                if submit:
                    if decision == "\u2705 Done":
                        status = "approved"
                        final_annotation = parsed if parsed is not None else {"errors": []}
                    elif decision == "\u270f\ufe0f Done after correction":
                        status = "edited"
                        final_annotation = parsed if parsed is not None else {"errors": []}
                    else:
                        status = "rejected"
                        final_annotation = None

                    original_errors = anno_data.get("errors", []) if anno_data else []

                    my_reviews[trace_file.name] = {
                        "status": status,
                        "reviewer": reviewer.strip(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "annotation": final_annotation,
                        "original_annotation": {"errors": [dict(e) for e in original_errors]} if anno_data else None,
                    }
                    save_review(review_path, my_reviews)
                    st.session_state.editing_json = None
                    st.session_state.prev_trace_stem = None
                    # Move to next in filtered list
                    cur_pos = next((p for p, i in enumerate(filtered_indices) if i == idx), None)
                    if cur_pos is not None and cur_pos < len(filtered_indices) - 1:
                        st.session_state.current_idx = filtered_indices[cur_pos + 1]
                    st.rerun()

            # Show who last reviewed (if different from current user)
            if current_status != "unreviewed" and my_review.get("status", "unreviewed") == "unreviewed":
                st.caption(f"Previously reviewed by **{merged_review.get('reviewer', '?')}** at {merged_review.get('timestamp', '?')}")


if __name__ == "__main__":
    main()