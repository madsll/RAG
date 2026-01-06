import json

# from RAG_openai import generate_response
from RAG_main import generate_response
def run_batch(prompts_file, output_file):
    # Load prompts
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = []

    for i, query in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {query}")
        input = str(query)
        # Run RAG pipeline
        answer, context, tool_suggestions = generate_response(input)

        # Collect result
        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "tool_suggestions": tool_suggestions
        }
        results.append(result)
        # print(f"✓ Completed {i+1}/{len(prompts)}")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to {output_file}")


if __name__ == "__main__":
    # Adjust paths to your JSON files
    prompts_file = "C:/Users/NX83SQ/Documents/GitHub/RAG/iamsar_scenarios.json"
    output_file = "C:/Users/NX83SQ/Documents/GitHub/RAG/Results/result_open_qwen4b.json"

    run_batch(prompts_file, output_file)
