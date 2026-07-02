import json
import os

TP = 0
FP = 0
TN = 0
FN = 0


def analyze_json_file(file_path):
    global TP, FP, TN, FN

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summarizer_score = (
        data.get("summarizer_score", {}).get("scores", [{}])[0].get("score", "").lower()
    )
    our_judge_score = 1 if summarizer_score == "ideal" else 0

    annotation = data.get("annotation", {})
    their_judge_score = None

    if "scores" in annotation:
        overall_score = annotation.get("scores", [{}])[0].get("overall", 0)
        their_judge_score = 1 if overall_score >= 4 else 0  # >=4 -> 1, <=3.5 -> 0

    else:
        annotation_values = list(annotation.values())

        if not annotation_values:
            return

        if all(value == 0 for value in annotation_values):
            their_judge_score = 1
        elif all(value == 1 for value in annotation_values):
            their_judge_score = 0
        else:
            return

    if their_judge_score == 1 and our_judge_score == 1:
        TP += 1
        print(f"  TP")
    elif their_judge_score == 1 and our_judge_score == 0:
        FN += 1
        print(f"  FN")
    elif their_judge_score == 0 and our_judge_score == 0:
        TN += 1
        print(f"  TN")
    elif their_judge_score == 0 and our_judge_score == 1:
        FP += 1
        print(f"  FP")


directory = "examples/trail_all"
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            analyze_json_file(file_path)
        except Exception as e:
            print(f"Error during proccessing {file_path}: {e}")

print("\n" + "=" * 50)
print(f"TP (True Positive): {TP}")
print(f"FP (False Positive): {FP}")
print(f"TN (True Negative): {TN}")
print(f"FN (False Negative): {FN}")
print("=" * 50)

total = TP + FP + TN + FN
if total > 0:
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
