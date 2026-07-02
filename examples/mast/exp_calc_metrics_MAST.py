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

    annotation = data.get("annotation", {})
    annotation_values = list(annotation.values())

    summarizer_score = (
        data.get("summarizer_score", {}).get("scores", [{}])[0].get("score", "").lower()
    )

    if all(value == 0 for value in annotation_values):
        if summarizer_score in ["ideal"]:
            TP += 1
            print(f"{file_path}: TP")
        else:
            FN += 1
            print(f"{file_path}: FN")

    else:
        if summarizer_score in ["fair", "poor"]:
            TN += 1
            print(f"{file_path}: TN")
        else:
            FP += 1
            print(f"{file_path}: FP")


directory = "data/mast_results_all_22_10_25"
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        analyze_json_file(file_path)

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
