from langchain.evaluation import load_evaluator
from langchain_fireworks import ChatFireworks
from ..retrieval_generation.query_rag import build_rag_chain, raw_context
from dotenv import load_dotenv
import os


# Load Fireworks API key
load_dotenv()
api_key = os.getenv("FIREWORKS_API_KEY")
if not api_key:
    raise ValueError("FIREWORKS_API_KEY not found in .env file")

# Define Fireworks LLM for evaluation
eval_llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    temperature=0.0,
    max_tokens=512
)

# Load evaluators with Fireworks LLM
conciseness_eval = load_evaluator("labeled_criteria", criteria="conciseness", llm=eval_llm)
correctness_eval = load_evaluator("labeled_criteria", criteria="correctness", llm=eval_llm)
relevance_eval = load_evaluator("labeled_criteria", criteria="relevance", llm=eval_llm)


def evaluate_query(question: str):
    rag = build_rag_chain()
    result = rag(question)

    answer = result["answer"]
    docs = result["retrieved_docs"]
    reference = raw_context(docs)

    evaluations = {
        "conciseness": conciseness_eval.evaluate_strings(
            input=question,
            prediction=answer,
            reference=reference
        ),
        "correctness": correctness_eval.evaluate_strings(
            input=question,
            prediction=answer,
            reference=reference
        ),
        "relevance": relevance_eval.evaluate_strings(
            input=question,
            prediction=answer,
            reference=reference
        )
    }

    return {
        "question": question,
        "answer": answer,
        "reference": reference[:400] + "..." if len(reference) > 400 else reference,
        "results": evaluations
    }
