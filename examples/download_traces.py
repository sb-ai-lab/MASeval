
import asyncio
import json
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv(".env")

from maseval import get_langfuse_download_client

from rich import print
from tqdm import tqdm


async def main(
    gaia_eval: bool,
    name: str,
    folder_name: str = "score"
):
    """Main example demonstrating the library usage with Langfuse traces."""
    TRACE_IDS = [
        "2a70969e03b1ef3e1a1f01742d812531",
        "a2b43cb609caafd9f9f3b79cd090c76b",
        "19400f54ed9b905fa11c6c51c1af1ca8",
        "550d8604dd43e4fff2fa3378197129e6",
        "24be15f2c4a3009b156c9ce3d3d9f436",
        "a7cf7780516fb065719877cd4a712653",
        "15955767b93b11bd0f53918e8899d6ed",
        "1ffc0a53dc6e6b3426cc53a2c9b3e3cf",
        "6e01ea05436eff84ea7b0b34a4943365",
        "ac67b1d3aa892c4b16eb0419d5bdd835",
    ]
    print("=== Downloading selected traces by ID ===")
    lf = get_langfuse_download_client()

    task_ids = TRACE_IDS
    print(f"Will download {len(task_ids)} traces")

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
            
        
        if gaia_eval and hasattr(trace_data, "output") and trace_data.output:
            if "ground_truth" in trace_data.output and "response" in trace_data.output:
                # prepare metadata for the trace
                trace_metadata = {
                    "task_id": task,
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
            folder_name="ideal_mas"
        )
    )
