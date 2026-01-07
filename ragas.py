import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics import f1_score

# File Paths | Change to own
BASE_DIR = Path("C:/Users/YN64GA/Rasmus/IAMSAR/Code/ragas/RAG_main")
RESULTS_DIR = BASE_DIR / "Results"
GROUND_TRUTH_PATH = BASE_DIR / "iamsar_tool_sequences.json"

load_dotenv()

base_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    timeout=600,
    max_retries=3
)

evaluator_llm = LangchainLLMWrapper(langchain_llm=base_llm, bypass_n=True)
evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

custom_run_config = RunConfig(
    timeout=600,
    max_retries=10,
    max_wait=60,
    max_workers=4
)

model_files = {
    "Qwen-4B": "result_open_qwen4b.json",
    "Qwen-0.6B": "result_open_qwen06b.json",
    "MiniLM-L6": "result_faiss_index.json",
    "OpenAI-Small": "result_open_ai.json",
    "OpenAI-Large": "result_open_ai_large.json"
}

def load_and_evaluate_all():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    gt_lookup = {item["scenario_id"]: item["tool_sequence"] for item in ground_truth}

    all_comparison_data = []

    for model_name, file_name in model_files.items():
        file_path = RESULTS_DIR / file_name
        if not file_path.exists():
            continue

        print(f"Evaluating {model_name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        dataset = EvaluationDataset.from_dict([
            {
                "user_input": str(res["query"]),
                "response": res["answer"],
                "retrieved_contexts": [res["context"]]
            } for res in results
        ])

        ragas_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=custom_run_config 
        )
        ragas_df = ragas_result.to_pandas()

        for i, res in enumerate(results):
            row_scores = ragas_df.iloc[i]
            predicted = res.get("tool_suggestions", [])
            gt_tools = gt_lookup.get(res["query"]["scenario_id"], [])
            
            all_tools = list(set(predicted) | set(gt_tools))
            f1 = f1_score([1 if t in gt_tools else 0 for t in all_tools], 
                          [1 if t in predicted else 0 for t in all_tools], 
                          zero_division=0) if all_tools else 1.0

            all_comparison_data.append({
                "Embedding_Model": model_name,
                "Faithfulness": row_scores["faithfulness"],
                "Relevance": row_scores["answer_relevancy"],
                "Tool_F1": f1
            })

    return pd.DataFrame(all_comparison_data)

if __name__ == "__main__":
    df_results = load_and_evaluate_all()
    
    summary = df_results.groupby("Embedding_Model").mean().reset_index()
    plot_data = summary.melt(id_vars="Embedding_Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=plot_data, x="Embedding_Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Ragas Scores Comparison of Embedding Models", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(format(height, '.2f'), 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

    print("\n=== Overall Summary ===")
    print(summary.to_string(index=False))