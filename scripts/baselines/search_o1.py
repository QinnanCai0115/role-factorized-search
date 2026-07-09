import requests
import pdb
import sys
sys.path.append('./')
import argparse
import pandas as pd
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


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
    MODEL = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    # MODEL = "Qwen/Qwen2.5-7B-Instruct"

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

def retrieve(query: str, endpoint: str = 'http://127.0.0.1:3000/retrieve'):
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


def extract_search_query_from_text(text: str) -> str:
    """Extract the query string between <|begin_search_query|> and <|end_search_query|>."""
    match = re.search(r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>", text, re.DOTALL)
    return match.group(1).strip() if match else None

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def search_o1_answer(
    question: str,
    endpoint: str,
    max_turns: int = 4,
):
    """
    Search-o1 style reasoning loop for a single question.
    Simulates an agent that interleaves search and reasoning.
    """
    # Initialize prompt state
    system_instruction = get_multiqa_search_o1_instruction(max_turns)
    reasoning_chain = []
    retrievals = []
    all_contexts = []
    search_count = 0

    current_prompt = f"{system_instruction}\nQuestion: {question}\nAssistant:"
    
    for turn in range(max_turns):
        # Step 1: Reasoning step generation
        model_response = call_llm(current_prompt).strip()
        reasoning_chain.append(model_response)
        current_prompt += "\n" + model_response

        # Step 2: Extract search query if present
        search_query = extract_search_query_from_text(model_response)

        if search_query is None or search_count >= max_turns:
            break  # No more search or reached search cap
        
        # Step 3: Retrieve context
        context = retrieve(search_query, endpoint)
        retrievals.append({
            "query": search_query,
            "context": context
        })

        # Step 4: Insert new information using analysis prompt
        previous_steps = "\n".join(reasoning_chain)
        analysis_prompt = get_webpage_to_reasonchain_instruction(previous_steps, search_query, context)
        analysis_result = call_llm(analysis_prompt).strip()

        # Step 5: Inject <|begin_search_result|>...<|end_search_result|>
        result_injection = f"\n{BEGIN_SEARCH_RESULT}\n{analysis_result}\n{END_SEARCH_RESULT}\n"
        current_prompt += result_injection
        all_contexts.append(result_injection)
        reasoning_chain.append(result_injection)
        search_count += 1
    
    return {
        "chain_of_thought": reasoning_chain,
        "retrievals": retrievals,
        "all_context": "\n".join(all_contexts)
    }


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

        search_o1_result = search_o1_answer(question, endpoint)
        
        return question, {
            "golden_answers": golden_answers,
            "chain_of_thought": search_o1_result["chain_of_thought"],
            "retrievals": search_o1_result["retrievals"],
            "all_context": search_o1_result["all_context"]
        }
    except Exception as e:
        print(f"[ERROR] Failed to process question: {row.get('question', '[unknown]')} — {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', default="data/nq_hotpotqa_train/test_e5_ug.parquet", help='Path to input parquet file')
    parser.add_argument('--input_file', default="data/mirage/mirage_medcorp_test.parquet", help='Path to input parquet file')
    parser.add_argument("--endpoint", default="http://127.0.0.1:3000/retrieve", help="Retrieval API endpoint URL")
    parser.add_argument('--output_file', default="data/search_o1/results_medcorp_mirage.json", help="Path to save output JSON")
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
