import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from trail_benchmark.benchmarking.run_eval import get_prompt

load_dotenv(".env")

import glob
import json
import os

import pandas as pd
from litellm import completion
from rich import print
from tqdm import tqdm

from maseval import get_langfuse_download_client

sys.path.insert(0, str(Path(__file__).parent))
from maseval.parsers.langfuse_parser_v3 import parse_langfuse_task


def get_trail_becnh(directory_path):
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    all_data = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
                print(f"Загружен: {os.path.basename(json_file)}")
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга {json_file}: {e}")
        except Exception as e:
            print(f"Ошибка при чтении {json_file}: {e}")

    print(f"\nВсего загружено файлов: {len(all_data)}")
    print(f"Тип данных: {type(all_data)}")

    return all_data


def calculate_metrics(path_to_csv_ht, name):
    df = pd.read_csv(path_to_csv_ht)
    task_id_to_ht = dict(zip(df["trace_id"], df["human_score"]))

    file_path = f"trail_our_ds_{name}.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        results = json.load(json_file)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for task_id in results.keys():
        res = results[task_id]

        ht = task_id_to_ht.get(task_id)
        if ht is None:
            print(f"Warning: trace_id {task_id} not found in CSV")
            continue
        try:
            is_correct = int(ht)
            judge_overall_score = res["scores"]["overall"]
            if judge_overall_score >= 3.0:
                judge_overall_score = 1
            else:
                judge_overall_score = 0

            if judge_overall_score == is_correct == 1:
                TP += 1
            elif judge_overall_score == is_correct == 0:
                TN += 1
            elif judge_overall_score == 1 and is_correct == 0:
                FP += 1
            elif judge_overall_score == 0 and is_correct == 1:
                FN += 1
        except:
            continue

    total_samples = TP + TN + FP + FN
    if total_samples == 0:
        print("No samples to calculate metrics")
    else:
        accuracy = (TP + TN) / total_samples
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")


def calculate_metrics_trail_ds(name, dir, threshold: float = 4.0):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    file_path = f"trail_{name}.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        results = json.load(json_file)

    for name in results.keys():
        try:
            with open(dir + name + ".json", "r", encoding="utf-8") as f:
                content = json.load(f)
                gt_score = content["scores"][0]["overall"]
                judge_score_float = results[name]["scores"]["overall"]

                if judge_score_float < threshold:
                    judge_overall_score = 0
                else:
                    judge_overall_score = 1

                if gt_score < threshold:
                    gt_score_round = 0
                else:
                    gt_score_round = 1
        except:
            continue

        if judge_overall_score == gt_score_round == 1:
            TP += 1
        elif judge_overall_score == gt_score_round == 0:
            TN += 1
        elif judge_overall_score == 1 and gt_score_round == 0:
            FP += 1
        elif judge_overall_score == 0 and gt_score_round == 1:
            FN += 1

    total_samples = TP + TN + FP + FN
    if total_samples == 0:
        print("No samples to calculate metrics")
    else:
        accuracy = (TP + TN) / total_samples
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")


def call_litellm(trace: str, model: str = "openai/gpt-4o"):
    prompt = get_prompt(trace)
    base_params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "openrouter/" + model,
        "max_completion_tokens": 8000,
        "temperature": 0.0,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ["OPENROUTER_API_KEY"],
    }
    response = completion(**base_params)
    return response


async def eval_judge_lf_trace(name, model):
    print("=== Downloading traces from evaluation ===")
    lf = get_langfuse_download_client()

    traces_page1 = lf.api.trace.list(name=name, limit=100, page=1)
    traces_page2 = lf.api.trace.list(name=name, limit=100, page=2)

    all_traces = traces_page1.data + traces_page2.data
    task_ids = [item.id for item in all_traces]

    if not task_ids:
        raise ValueError(f"No tasks found in trace {name}")

    print(f"Found {len(task_ids)} tasks in trace {name}")

    results = {}

    # Run evaluation for each task
    for task in tqdm(task_ids, desc="Evaluating tasks: "):
        try:
            trace_data = lf.api.trace.get(task)
        except:
            time.sleep(5)
            try:
                trace_data = lf.api.trace.get(task)
            except:
                continue

        print(f"\n=== Evaluating Task {task} ===")
        try:
            eval_input = parse_langfuse_task(trace_data)
            result = call_litellm(str(eval_input), model)
            results[task] = {}
            if result.choices[0].message.content.startswith("```"):
                content = (
                    result.choices[0]
                    .message.content.split("\n", 1)[1]
                    .rsplit("\n```", 1)[0]
                )
            else:
                content = result.choices[0].message.content
            result_dict = json.loads(content)

            results[task]["scores"] = result_dict["scores"][0]
            results[task]["errors"] = result_dict["errors"]
        except:
            print("Something went wrong! Skip task.")
            continue

    file_path = f"trail_our_ds_{name}.json"
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


async def eval_judge_with_trail_bench(model, name, trail_bench_path):
    print("=== Downloading traces from evaluation ===")
    all_traces = get_trail_becnh(trail_bench_path)

    results = {}

    # Run evaluation for each task
    for task in tqdm(all_traces, desc="Evaluating tasks: "):

        print(f"\n=== Evaluating Task {task} ===")
        try:
            result = call_litellm(str(task["spans"]), model)
            results[task["trace_id"]] = {}
            if result.choices[0].message.content.startswith("```"):
                content = (
                    result.choices[0]
                    .message.content.split("\n", 1)[1]
                    .rsplit("\n```", 1)[0]
                )
            else:
                content = result.choices[0].message.content
            result_dict = json.loads(content)

            results[task["trace_id"]]["scores"] = result_dict["scores"][0]
            results[task["trace_id"]]["errors"] = result_dict["errors"]
        except:
            print("Something went wrong! Skip task.")
            continue

    file_path = f"trail_{name}.json"
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


if __name__ == "__main__":
    # SMALL MASs
    # name="gaia_task_07aac7b1-ffc3-4787-8e4c-7fb522156097"
    # csv_ht = 'traces_export_new.csv'

    # path_to_trail_gaia_ds = "/home/alina/Desktop/maseval-research/trail_benchmark/benchmarking/data/GAIA"

    # LARGE MASs
    name = "gaia_task_db0c3ed0-a4af-4442-bb6f-884d6da055cb"
    csv_ht = "traces_export_id_old.csv"
    full_path = "/home/alina/Desktop/maseval-research/trail_benchmark/benchmarking/processed_annotations_gaia/"

    # asyncio.run(
    #     eval_judge_lf_trace(
    #         model="google/gemini-2.5-flash",
    #         name=name
    #     )
    # )
    # calculate_metrics(csv_ht, name=name)

    calculate_metrics_trail_ds(name="trail_bench", dir=full_path, threshold=2.5)
