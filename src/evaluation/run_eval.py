# src/evaluation/run_eval.py
import re
import textwrap
from src.evaluation.langchain_evaluator import evaluate_query

def normalize_value(val):
    """Turn evaluator value into human-friendly Pass/Fail/Unknown."""
    if val is None:
        return "Unknown"
    if isinstance(val, dict):
        val = val.get("value") or val.get("label") or str(val)
    val = str(val).strip()
    val_clean = re.sub(r"[*_`]", "", val)
    low = val_clean.lower()

    positives = {"y", "yes", "true", "correct", "pass", "yep"}
    negatives = {"n", "no", "false", "incorrect", "fail"}

    if low in positives:
        return "Pass"
    if low in negatives:
        return "Fail"
    if low.startswith("y") or " yes" in low:
        return "Pass"
    if low.startswith("n") or " no" in low:
        return "Fail"

    return val_clean

def pretty_print_results(results):
    for k, v in results.items():
        if isinstance(v, dict):
            raw_value = v.get("value") or v.get("label") or v.get("result") or None
            score = v.get("score")
            reasoning = v.get("reasoning") or v.get("explanation") or ""
        else:
            raw_value = v
            score = None
            reasoning = ""

        status = normalize_value(raw_value)
        score_text = f" ({score})" if score not in (None, "") else ""
        print(f"- {k.capitalize()}: {status}{score_text}")

        if reasoning:
            paras = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
            for p in paras:
                wrapped = textwrap.fill(p, width=90, replace_whitespace=False)
                indented = "\n   ".join(wrapped.splitlines())
                print(f"   ↳ {indented}\n")
        else:
            print("   ↳ (no reasoning returned)\n")

if __name__ == "__main__":
    try:
        while True:
            q = input("\nAsk a question (or type 'exit'): ")
            if q.lower() in {"exit", "quit"}:
                break

            report = evaluate_query(q)

            print("\n\033[92m\033[1m=== RAG Answer ===\033[0m")
            print(report["answer"])

            print("\n\033[93m\033[1m=== Evaluation Results ===\033[0m")
            pretty_print_results(report["results"])

    except KeyboardInterrupt:
        print("\nExiting.")
