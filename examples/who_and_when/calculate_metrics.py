import json
import glob
import pandas as pd

df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")
# df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet")

# Path to JSON files
path = "/Users/alina/maseval-research/examples/who_and_when/who&when_handcrafted_gemini_all_metrics_6ens/*.json"
files = glob.glob(path)

# Counters
tp = tn = fp = fn = 0
correct_mistake_agent, correct_mistake_agent_judge = 0, 0

# Process each file
for file in files:
    found = False
    mistake_agent_true = df.iloc[int(file.split("_")[-1].split(".")[0])]['mistake_agent']
    data = json.load(open(file))
    gt = data['gt']
    score = data['summarizer_score']['scores'][0]['score']
    
    pred = 0 if score == 'poor' or score == 'fair'  else 1
    if pred == 0 and score == 'poor':
        for llm_metric in data.keys():
            if found:
                break
            if data.get(llm_metric):
                if llm_metric not in ['gt', 'label_answer']:
                    for score in data[llm_metric]['scores']:
                        if score['score'] == 'poor' or score['score'] == 'fair':
                            if score['justification']:
                                if mistake_agent_true.lower().replace(" ", "") in score['justification'].lower().replace(" ", ""):
                                    correct_mistake_agent += 1
                                    found = True
                                    break
        judge_justification = data['summarizer_score']['scores'][0]['justification']
        if mistake_agent_true.lower().replace(" ", "") in judge_justification.lower().replace(" ", ""):
            correct_mistake_agent_judge += 1
            
    if gt == 1 and pred == 1:
        tp += 1
    elif gt == 0 and pred == 0:
        tn += 1
    elif gt == 0 and pred == 1:
        fp += 1
    elif gt == 1 and pred == 0:
        fn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
who_accuracy = correct_mistake_agent / len(files) if files else 0
who_judge_accuracy = correct_mistake_agent_judge / len(files) if files else 0
print(f"Accuracy: {accuracy:.4f}")
print(f"Who llm-metrics Accuracy: {who_accuracy:.4f}")
print(f"Who Judge Accuracy: {who_judge_accuracy:.4f}")