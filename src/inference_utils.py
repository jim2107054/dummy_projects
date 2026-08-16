import re
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metric_utils import rouge_l_f1

SYSTEM_PROMPT = "আপনি একজন অভিজ্ঞ চিকিৎসক। রোগীর প্রশ্ন মনোযোগ দিয়ে পড়ুন এবং চিকিৎসাগতভাবে সঠিক, স্পষ্ট ও সহানুভূতিশীল উত্তর বাংলায় দিন।"

def load_model_for_inference(model_path: str = "/kaggle/working/final_model") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the merged model and tokenizer in bfloat16 for evaluation or submission."""
    print(f"Loading merged model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded and set to eval mode.")
    return model, tokenizer

def generate_candidates(
    model, 
    tokenizer, 
    patient_input: str, 
    k: int = 4, 
    max_new_tokens: int = 260, 
    min_new_tokens: int = 60, 
    temperature: float = 0.7, 
    top_p: float = 0.9
) -> list[str]:
    """Generate k candidate responses for a single patient query."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input}
    ]
    
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to manual Alpaca-style prompt
        prompt_text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{patient_input}\n\n"
            f"### Response:\n"
        )
        
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            num_return_sequences=k,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
    candidates = []
    for out in outputs:
        new_tokens = out[input_len:]
        candidate = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        candidates.append(candidate)
        
    return candidates

def length_ok(text: str, target_range: tuple[int, int] = (65, 145)) -> bool:
    """Check if the response length in words falls within the target range."""
    word_count = len(str(text).split())
    return target_range[0] <= word_count <= target_range[1]

def mbr_select(candidates: list[str]) -> str:
    """Select the most canonical candidate using Minimum Bayes Risk over pairwise ROUGE-L."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""
        
    # Filter to length_ok candidates if any exist, otherwise use all candidates
    ok_candidates = [c for c in candidates if length_ok(c)]
    pool = ok_candidates if ok_candidates else candidates
    
    if len(pool) == 1:
        return pool[0]
        
    best_candidate = pool[0]
    best_score = -1.0
    
    for i, cand_i in enumerate(pool):
        scores = []
        for j, cand_j in enumerate(pool):
            if i != j:
                scores.append(rouge_l_f1(cand_i, cand_j))
        mean_score = sum(scores) / len(scores) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = cand_i
            
    return best_candidate

def clean_output(text: str) -> str:
    """Clean the generated response by removing echoed prompts and extra whitespaces."""
    # Collapse multiple spaces and newlines
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    
    # Remove system prompt leaks if any
    escaped_sys = re.escape(SYSTEM_PROMPT)
    cleaned = re.sub(escaped_sys, "", cleaned, flags=re.IGNORECASE).strip()
    
    # Remove template tags if model generates them
    cleaned = re.sub(r"###\s*(System|Instruction|Input|Response|User|Assistant):", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("<s>", "").replace("</s>", "").strip()
    
    return cleaned

def run_inference(model, tokenizer, df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Generate medical responses for the given dataset with MBR selection and progress updates."""
    results = []
    start_time = time.time()
    
    total_rows = len(df)
    print(f"Starting inference on {total_rows} rows (k={k})...")
    
    for idx, row in df.iterrows():
        row_id = row["id"]
        patient_input = row["input"]
        
        # 1. Generate candidates
        candidates = generate_candidates(model, tokenizer, patient_input, k=k)
        
        # 2. Minimum Bayes Risk selection
        selected = mbr_select(candidates)
        
        # 3. Clean output
        output_text = clean_output(selected)
        
        results.append({
            "id": row_id,
            "output": output_text
        })
        
        # Print progress every 50 rows
        count = idx + 1
        if count % 50 == 0 or count == total_rows:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            rem_time = avg_time * (total_rows - count)
            
            # Format time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            rem_str = f"{int(rem_time // 60)}m {int(rem_time % 60)}s"
            print(f"Processed {count}/{total_rows} rows | Elapsed: {elapsed_str} | Est. Remaining: {rem_str}")
            
    return pd.DataFrame(results)
