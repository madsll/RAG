import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from dotenv import load_dotenv
from openai import OpenAI
import os
import statistics
# ragas imports
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy

# Initialize OpenAI client
load_dotenv()
api_key_get = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key_get)

# --- Load your JSONs ---
with open("C:/Users/NX83SQ/Documents/GitHub/RAG/Results/result_open_qwen4b.json", encoding="utf-8") as f:
    results = json.load(f)

with open("C:/Users/NX83SQ/Documents/GitHub/RAG/iamsar_tool_sequences.json") as f:
    ground_truth = json.load(f)

# --- Build lookup for ground truth tools ---
gt_lookup = {item["scenario_id"]: item["tool_sequence"] for item in ground_truth}

# --- Collect metrics ---
metrics_data = []

for res in results:
    scenario_id = res["query"]["scenario_id"]
    answer = res["answer"]
    context = res["context"]
    query = str(res["query"])

    # Step 1: Answer quality (RAGAS)
    dataset = EvaluationDataset.from_dict([
        {
            "user_input": query,             # original query
            "response": answer,              # generated answer
            "retrieved_contexts": [context]  # supporting context(s)
        }
    ])
    # scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    
    # Step 2: Tool sequence validation
    predicted = res["tool_suggestions"]
    ground_truth_tools = gt_lookup.get(scenario_id, [])

    all_tools = list(set(predicted) | set(ground_truth_tools))
    y_true = [1 if t in ground_truth_tools else 0 for t in all_tools]
    y_pred = [1 if t in predicted else 0 for t in all_tools]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics_data.append({
    "scenario_id": scenario_id,
    # "faithfulness": scores["faithfulness"],
    # "relevance": scores["answer_relevancy"],
    "precision": precision,
    "recall": recall,
    "f1": f1
    })
    # print(scores["answer_relevancy"])

# --- Convert to DataFrame for easy plotting ---
df = pd.DataFrame(metrics_data)

# --- Summary statistics ---
summary = df.mean(numeric_only=True)
print("\n=== Overall Summary ===")
print(summary)

# --- Plot distributions ---
# plt.figure(figsize=(12,6))
# sns.boxplot(data=df[["faithfulness","relevance"]])
# plt.title("Distribution of Answer Quality Metrics Across Scenarios")
# plt.ylabel("Score")
# plt.show()

# plt.figure(figsize=(12,6))
# sns.boxplot(data=df[["precision","recall","f1"]])
# plt.title("Distribution of Tool Matching Metrics Across Scenarios")
# plt.ylabel("Score")
# plt.show()

# --- Plot averages as bar chart ---
# plt.figure(figsize=(10,6))
# summary.plot(kind="bar", color="skyblue")
# plt.title("Average Metrics Across All Scenarios")
# plt.ylabel("Average Score")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
