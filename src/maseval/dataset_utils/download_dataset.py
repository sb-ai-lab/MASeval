"""Extract Langfuse traces and convert them to DataFrame."""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env")

from maseval import get_langfuse_download_client


def main(trace_name: str):
    """Extract traces from Langfuse and save to DataFrame."""

    # Initialize Langfuse client
    print("=== Downloading traces from evaluation project ===")
    lf = get_langfuse_download_client()

    # Fetch traces (you can adjust pagination as needed)
    traces_page1 = lf.api.trace.list(name=trace_name, limit=100, page=1)
    traces_page2 = lf.api.trace.list(name=trace_name, limit=100, page=2)

    all_traces = traces_page1.data + traces_page2.data
    task_ids = [item.id for item in all_traces]

    if not task_ids:
        raise ValueError(f"No tasks found in trace {trace_name}")

    print(f"Found {len(task_ids)} tasks in trace {trace_name}")

    data = []

    for task_id in tqdm(task_ids, desc="Processing traces"):
        try:
            # download trace data
            trace_data = lf.api.trace.get(task_id)

            data.append(
                {
                    "trace_id": task_id,
                    "gaia_task_id": trace_data.output["task_id"],
                    "trace_string": str(trace_data.observations),
                    "mas_response": trace_data.output["response"],
                    "gaia_gt": trace_data.output["ground_truth"],
                }
            )

        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            continue

    df = pd.DataFrame(data)

    output_file = Path(__file__).parent / f"traces_{trace_name}.xlsx"
    df.to_excel(output_file, index=False)

    print(f"\n=== Results ===")
    print(f"Processed {len(df)} traces")
    print(f"Saved to: {output_file}")
    print(f"\nDataFrame preview:")
    print(df.head())

    return df


if __name__ == "__main__":
    # dataset (gr. 2, small MASs)
    # df = main(trace_name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097")

    # dataset (gr. 1, large MASs)
    df = main(trace_name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb")
