import asyncio
import json
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv(".env")

from maseval import get_langfuse_download_client
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task

from rich import print
from tqdm import tqdm


async def main(
    gaia_eval: bool,
    name: str,
    folder_name: str = "score"
):
    """Download traces"""
    print("=== Downloading traces from evaluation project ===")
    lf = get_langfuse_download_client()

    # Download all traces
    traces_page1 = lf.api.trace.list(name=name, limit=100, page=1)
    traces_page2 = lf.api.trace.list(name=name, limit=100, page=2)
    all_traces = traces_page1.data + traces_page2.data
        
    task_ids = [item.id for item in all_traces]

    if not task_ids:
        raise ValueError(f"No tasks found in trace {name}")

    print(f"Found {len(task_ids)} tasks in trace {name}")

    # Create output folder
    output_folder = Path(folder_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Run evaluation for each task
    cnt_lost = 0
    for task in tqdm(task_ids, desc="Evaluating tasks"):
        succ = False
        retries = 0
        
        while not succ and retries < 5:
            try:
                trace_data = lf.api.trace.get(task)
                trace_dict = trace_data.dict()
                succ = True
            except Exception as e:
                retries += 1
                cnt_lost += 1
                print(f'Retry {retries} for trace {task}: {e}')
                time.sleep(10)
        
        if not succ:
            print(f'Skipping trace {task} after {retries} retries')
            continue
            
        eval_input = parse_langfuse_task(trace_data)
        
        if gaia_eval and hasattr(trace_data, "output") and trace_data.output:
            if "ground_truth" in trace_data.output and "response" in trace_data.output:
                # prepare metadata for the trace
                trace_metadata = {
                    "task_id": task,
                    "trace_id": eval_input.trace_id,
                    "ground_truth": trace_data.output["ground_truth"],
                    "mas_response": trace_data.output["response"],
                    "annotation": {},
                    "trace": trace_dict
                }
                
                # Save to JSON file
                output_file = output_folder / f"trace_{task}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(trace_metadata, f, ensure_ascii=False, indent=2, default=str)
            else:
                print(f'No response or ground_truth in {task}!')
        else:
            print(f'Empty trace {task}! Output not found.')
    
    print(f"\n=== Done! Lost {cnt_lost} retries total ===")
    print(f"Results saved to: {output_folder.absolute()}")


if __name__ == "__main__":
    asyncio.run(
        main(
            gaia_eval=True,
            name="gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb",
            # name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097",
            folder_name="large_mas_traces"
        )
    )
