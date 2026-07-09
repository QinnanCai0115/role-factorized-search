import requests
import pdb
import sys
sys.path.append('./')
# # from generator_llms.claude_api import get_claude_response as get_llm_response
# from generator_llms.local_inst import call_llm as get_llm_response
import argparse
import pandas as pd
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

def call_llm(prompt: str) -> str:
    """
    Call the LLM with a simple prompt and get a short response.
    """
    # Prepare the payload for the API call
    headers = {"Content-Type": "application/json"}
    
    # Format as chat messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    # MODEL = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    MODEL = "Qwen/Qwen2.5-7B-Instruct"

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2048,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

def retrieve(query: str, endpoint: str):
    """
    Send a query to a retrieval endpoint and return the top documents as a formatted string.
    
    Args:
        query (str): The input query string.
        endpoint (str): The URL of the retrieval service.
    
    Returns:
        str: A formatted string of retrieved documents.
    """
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        results = response.json()['result']
    except Exception as e:
        print(f"[ERROR] Retrieval failed for query: {query}\n{e}")
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def ircot_answer(question, endpoint, max_steps=4):
    """
    Answer a question using the IRCoT method (Interleaved Retrieval with Chain-of-Thought),
    and return both the reasoning steps, per-step retrievals, and full retrieval context.

    Args:
        question (str): The input question.
        endpoint (str): The retrieval endpoint URL.
        max_steps (int): Maximum number of reasoning-retrieval steps.

    Returns:
        dict: {
            'chain_of_thought': List of reasoning steps,
            'retrievals': List of retrieved contexts per step,
            'all_context': Concatenated string of all retrieved documents
        }
    """
    cot_chain = []
    retrievals = []
    all_context_blocks = []

    # Initial retrieval using the question
    initial_context = retrieve(question, endpoint)
    retrievals.append({
        "query": question,
        "context": initial_context
    })
    all_context_blocks.append(initial_context)
    retrieved_context = initial_context

    for step in range(max_steps):
        # Build prompt with current reasoning and context
        prompt = build_prompt(question, retrieved_context, cot_chain)

        # Generate the next step in the chain of thought
        next_cot = call_llm(prompt).strip()
        cot_chain.append(next_cot)
        query_only = re.sub(r"<search_complete>.*?</search_complete>", "", next_cot, flags=re.IGNORECASE).strip()

        # Stop condition
        if "<search_complete>yes</search_complete>" in next_cot.lower():
            break
        
        # Retrieve based on the latest reasoning step
        new_context = retrieve(query_only, endpoint)
        retrievals.append({
            "query": query_only,
            "context": new_context
        })
        all_context_blocks.append(new_context)
        
        # Add to accumulated context
        retrieved_context += "\n" + new_context

    return {
        "chain_of_thought": cot_chain,
        "retrievals": retrievals,
        "all_context": "\n".join(all_context_blocks)
    }



def build_prompt(question, context, cot_chain):
    """
    Construct a prompt for the LLM based on the question, retrieved evidence, and previous reasoning steps.

    Args:
        question (str): The original input question.
        context (str): Retrieved text context.
        cot_chain (list): List of reasoning steps so far.

    Returns:
        str: A prompt formatted for the LLM.
    """
    prompt = f"Question: {question}\n\n"

    if context:
        prompt += "Context:\n" + context.strip() + "\n"

    if cot_chain:
        prompt += "\nChain of Thought:\n"
        for i, step in enumerate(cot_chain, 1):
            prompt += f"{i}. {step.strip()}\n"

    prompt += (
        "\nPlease reason through the problem step by step based on the given context. "
        "Only provide the final answer if you are confident and have sufficient information. "
        "Otherwise, continue with intermediate reasoning steps.\n\n"
        "At the end of your response, write one of the following tags to indicate whether further information is needed:\n"
        "    <search_complete>Yes</search_complete>   if no more information is needed.\n"
        "    <search_complete>No</search_complete>    if more information is needed.\n\n"
        "Now continue reasoning:"
    )

    return prompt



def load_previous_results(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_results(result_dict, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

def process_single_question(row, endpoint):
    try:
        # question = row["question"]
        # golden_answers = row.get("golden_answers", [])
        question = row['reward_model']['ground_truth']['question']
        golden_answers = row['reward_model']['ground_truth']['target'].tolist()
        golden_answers = golden_answers.tolist() if hasattr(golden_answers, "tolist") else golden_answers
        extra_info = row.get("extra_info", {})
        data_source = extra_info.get("source", None) if isinstance(extra_info, dict) else None

        ircot_result = ircot_answer(question, endpoint)

        return question, {
            "golden_answers": golden_answers,
            "data_source": data_source,
            "chain_of_thought": ircot_result["chain_of_thought"],
            "retrievals": ircot_result["retrievals"],
            "all_context": ircot_result["all_context"]
        }
    except Exception as e:
        print(f"[ERROR] Failed to process question: {row.get('question', '[unknown]')} — {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', default="data/nq_hotpotqa_train/test_e5_ug.parquet", help='Path to input parquet file')
    parser.add_argument('--input_file', default="data/mirage/mirage_test.parquet", help='Path to input parquet file')
    parser.add_argument("--endpoint", default="http://127.0.0.1:3000/retrieve", help="Retrieval API endpoint URL")
    parser.add_argument('--output_file', default="data/ircot/results_mirage_7b_medcorp.json", help="Path to save output JSON")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of parallel workers")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load input and previous results
    df = pd.read_parquet(args.input_file)
    # results = load_previous_results(args.output_file)
    # processed_questions = set(results.keys())
    results = {}

    # print(f"Loaded {len(processed_questions)} previously processed questions.")
    # df = df[~df["question"].isin(processed_questions)]

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_single_question, row, args.endpoint): idx
            for idx, row in df.iterrows()
        }

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                question, result = future.result()
                if question and result:
                    with lock:
                        results[question] = result

                pbar.update(1)

                # Periodic save
                if pbar.n % 50 == 0:
                    save_results(results, args.output_file)

    # Final save
    save_results(results, args.output_file)
    print(f"\n✅ Final results saved to {args.output_file}")
