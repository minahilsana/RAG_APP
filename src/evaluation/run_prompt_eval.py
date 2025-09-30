from src.evaluation.prompt_evaluator import evaluate_with_prompt
import textwrap
import json

def pretty_print_eval(eval_json: dict):
    print("\n\033[95m\033[1m=== Evaluation Results (Prompt Template) ===\033[0m")
    for key, value in eval_json.items():
        if isinstance(value, dict):
            value = json.dumps(value, indent=2)  # pretty print nested dict
        else:
            value = str(value)  # ensure always string

        wrapped = textwrap.fill(value, width=90)
        print(f"- {key.replace('_', ' ').capitalize()}: {wrapped}")


if __name__ == "__main__":
    try:
        while True:
            q = input("\nAsk a question (or type 'exit'): ")
            if q.lower() in {"exit", "quit"}:
                break

            report = evaluate_with_prompt(q)

            print("\n\033[92m\033[1m=== RAG Answer ===\033[0m")
            print(report["answer"])

            pretty_print_eval(report["evaluation"])

    except KeyboardInterrupt:
        print("\nExiting.")
