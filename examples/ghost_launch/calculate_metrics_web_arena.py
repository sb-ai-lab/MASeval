import os
import json

def get_id_and_filename(score_folder, trace_folder):
    out = []
    for fn in os.listdir(score_folder):
        if fn.endswith('.json'):
            with open(os.path.join(score_folder, fn), 'r') as f:
                data = json.load(f)
            if 'id' in data:
                trace_fn = f"{data['id']}.json"
                full_trace_path = os.path.join(trace_folder, trace_fn)
                out.append((data['id'], fn, full_trace_path))
    return out

def get_last_reward(trace_path):
    with open(trace_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        steps = data.get('trajectory', [data])
    elif isinstance(data, list):
        steps = data
    else:
        return None
    rewards = [step.get('reward') for step in steps if 'reward' in step]
    if rewards:
        return rewards[-1]
    return None

def get_summarizer_score(score_path):
    with open(score_path, 'r') as f:
        data = json.load(f)
    s = data.get('summarizer_score', {})
    if isinstance(s, dict) and 'scores' in s:
        scores = s['scores']
        if scores:
            return scores[0]['score']
    return None

def f1_score(preds, trues):
    tp = sum((p == 'ideal' and t == 1.0) for p, t in zip(preds, trues))
    fp = sum((p == 'ideal' and t == 0.0) for p, t in zip(preds, trues))
    fn = sum((p != 'ideal' and t == 1.0) for p, t in zip(preds, trues))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

score_folder = "examples/ghost_launch/scores/webarena"
trace_folder = "data/webarena_traces_reddit"

results = get_id_and_filename(score_folder, trace_folder)
summarizer_scores = []
last_rewards = []
for id_, score_fn, trace_fn in results:
    s_score = get_summarizer_score(os.path.join(score_folder, score_fn))
    reward = get_last_reward(trace_fn)
    summarizer_scores.append(s_score)
    last_rewards.append(reward)
f1 = f1_score(summarizer_scores, last_rewards)
print('f1:', f1)
