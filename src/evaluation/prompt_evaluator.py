import json
import os
import textwrap
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from src.retrieval_generation.query_rag import build_rag_chain, raw_context
from langchain_fireworks import ChatFireworks

def load_api_key() -> str:
    """Load Fireworks API key from .env file."""
    load_dotenv()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found in .env file")
    return api_key


# --- Setup LLM for evaluation ---
api_key = load_api_key()
eval_llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    api_key=api_key
)


# --- Define Prompt Template ---
evaluation_prompt = PromptTemplate.from_template("""
        You are an evaluator of a RAG APP that is specifically designed to answer queries on HR policies(Leaves, OPD and IPD related queries only). Judge the RAG system responses by comparing them against a known ground truth.

        QUESTION: {question}
        CONTEXT: {context}
        RAG_RESPONSE: {model_answer}

        Evaluation Criteria:
        1. Response Validation: Does the answer fully respond to the user QUESTION without avoiding or refusing unnecessarily?
        2. Response Consistency: Is the answer logically consistent throughout (no contradictions or self-conflicts)?
        3. Correctness: The RAG_RESPONSE must directly answer the QUESTION and align with the facts presented in the CONTEXT. It should not contradict or misrepresent the CONTEXT.
        4. Contextual Relevance: Is the answer grounded in the given context?
        5. Hallucination: The RAG_RESPONSE must only contain information derived from the CONTEXT. Any information not supported by the CONTEXT will be considered a hallucination.

        Return your judgment strictly in JSON format with keys:
        {{
        "response_validation": "Valid/Invalid",
        "response_consistency": "Consistent/Inconsistent",
        "correctness": "Correct/Incorrect",
        "contextual_relevance": "Relevant/Irrelevant",
        "hallucination": "Present/Absent",
        "overall_score": "Pass/Fail",
        "reasoning": "Brief explanation for each judgment"
        }}
    """)

# --- Evaluation function ---
def evaluate_with_prompt(question: str):
    rag_chain = build_rag_chain()
    # run rag
    rag_result = rag_chain(question)
    rag_answer = rag_result["answer"]
    context = raw_context(rag_result["retrieved_docs"])
    # print(f"\n\033[94m\033[1m=== Retrieved Context ===\033[0m\n{context}\n")
    # format prompt
    formatted_prompt = evaluation_prompt.format(
        question=question,
        context=context,
        model_answer=rag_answer
    )

    # run evaluator LLM
    evaluation = eval_llm.invoke(formatted_prompt)

    try:
        eval_json = json.loads(evaluation.content)
    except Exception:
        eval_json = {"raw_output": evaluation.content}

    return {
        "answer": rag_answer,
        "evaluation": eval_json
    }
