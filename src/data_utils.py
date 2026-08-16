import os
import re
import hashlib
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def find_data_file(filename: str) -> str:
    """Find a data file in the local workspace or Kaggle input directories."""
    candidates = [
        Path(filename),
        Path("dataset") / filename,
        Path("/kaggle/input/competitions/nascenia-ai-hackathon") / filename,
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename}. Ensure it is present locally or in Kaggle inputs."
    )

def load_official(train_path: str = None, test_path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv, validating expected columns."""
    if train_path is None:
        train_path = find_data_file("train.csv")
    if test_path is None:
        test_path = find_data_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate columns
    required_train = {"id", "input", "output"}
    required_test = {"id", "input"}

    missing_train = required_train - set(train_df.columns)
    if missing_train:
        raise ValueError(f"train.csv is missing required columns: {missing_train}")

    missing_test = required_test - set(test_df.columns)
    if missing_test:
        raise ValueError(f"test.csv is missing required columns: {missing_test}")

    # Standardize data types and clean whitespace
    train_df = train_df[["id", "input", "output"]].copy()
    train_df["id"] = train_df["id"].astype(str)
    train_df["input"] = train_df["input"].fillna("").astype(str).str.strip()
    train_df["output"] = train_df["output"].fillna("").astype(str).str.strip()

    test_df = test_df[["id", "input"]].copy()
    test_df["id"] = test_df["id"].astype(str)
    test_df["input"] = test_df["input"].fillna("").astype(str).str.strip()

    return train_df, test_df

def profile_official(train_df: pd.DataFrame) -> None:
    """Print statistical profiling information of the training DataFrame."""
    print("=== Official Dataset Profiling ===")
    row_count = len(train_df)
    print(f"Total row count: {row_count}")

    # Length stats
    input_char_lens = train_df["input"].str.len()
    input_word_lens = train_df["input"].apply(lambda x: len(x.split()))
    output_char_lens = train_df["output"].str.len()
    output_word_lens = train_df["output"].apply(lambda x: len(x.split()))

    print("\nInput word length stats:")
    print(input_word_lens.describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nOutput word length stats:")
    print(output_word_lens.describe(percentiles=[0.25, 0.5, 0.75]))

    # Duplicates
    dup_inputs = train_df["input"].duplicated().sum()
    dup_outputs = train_df["output"].duplicated().sum()
    print(f"\nDuplicate inputs: {dup_inputs} ({dup_inputs/row_count*100:.2f}%)")
    print(f"Duplicate outputs: {dup_outputs} ({dup_outputs/row_count*100:.2f}%)")

    # Brand tokens / forensic analysis
    chatdoctor_count = train_df["output"].str.contains("চ্যাটডক্ট", regex=False).sum()
    nascenia_doc_count = train_df["output"].str.contains("নাসেনিয়া ডক", regex=False).sum()
    print(f"\nBrand token 'চ্যাটডক্ট' appearance count: {chatdoctor_count}")
    print(f"Brand token 'নাসেনিয়া ডক' appearance count: {nascenia_doc_count}")

    # Boilerplate stubs
    print("\nTop 15 most common exact outputs:")
    top_outputs = train_df["output"].value_counts().head(15)
    for idx, (val, count) in enumerate(top_outputs.items(), 1):
        words = len(val.split())
        print(f"{idx}. Count: {count} | Words: {words} | Output: {val[:80]}...")

    # Common openers
    print("\nTop 15 most common first-15-character openers:")
    openers = train_df["output"].apply(lambda x: x[:15]).value_counts().head(15)
    for idx, (val, count) in enumerate(openers.items(), 1):
        print(f"{idx}. Count: {count} | Opener: {repr(val)}")

    # Sanity checks
    expected_rows = 108954
    expected_median_words = 92
    
    median_out = output_word_lens.median()
    row_diff = abs(row_count - expected_rows)
    median_diff = abs(median_out - expected_median_words)

    if row_diff > 1000 or median_diff > 10 or chatdoctor_count == 0:
        print("\n[WARN] Dataset profile diverges meaningfully from expected characteristics!")
    else:
        print("\n[PASS] Dataset profile matches expected forensic characteristics.")

def clean_official(train_df: pd.DataFrame) -> pd.DataFrame:
    """Remove boilerplate stubs, filter output lengths, and drop empty rows."""
    before_count = len(train_df)
    
    # Identify boilerplate-only answers: count > 40 AND word count < 15
    output_counts = train_df["output"].value_counts()
    boilerplate_outputs = []
    for val, count in output_counts.items():
        if count > 40 and len(val.split()) < 15:
            boilerplate_outputs.append(val)
            
    print(f"Cleaning: Identified {len(boilerplate_outputs)} unique boilerplate stub/non-answers.")
    
    # Filter out boilerplate rows
    cleaned_df = train_df[~train_df["output"].isin(boilerplate_outputs)].copy()
    rows_lost_boilerplate = before_count - len(cleaned_df)
    print(f"Removed {rows_lost_boilerplate} rows due to boilerplate stubs.")
    
    # Filter output word counts to [10, 300]
    cleaned_df["output_word_len"] = cleaned_df["output"].apply(lambda x: len(x.split()))
    cleaned_df = cleaned_df[cleaned_df["output_word_len"].between(10, 300)].copy()
    rows_lost_length = before_count - rows_lost_boilerplate - len(cleaned_df)
    print(f"Removed {rows_lost_length} rows due to output word length outside [10, 300].")

    # Drop empty/whitespace-only input or output
    cleaned_df = cleaned_df[
        (cleaned_df["input"].str.strip() != "") & 
        (cleaned_df["output"].str.strip() != "")
    ].copy()
    
    cleaned_df = cleaned_df.drop(columns=["output_word_len"]).reset_index(drop=True)
    after_count = len(cleaned_df)
    print(f"Final row count: {after_count} (Removed {before_count - after_count} rows total)")
    
    return cleaned_df

def load_external_normalized() -> pd.DataFrame:
    """Load, inspect, and normalize shetumohanto/doctor_qa_bangla."""
    print("Loading external dataset shetumohanto/doctor_qa_bangla...")
    ds = load_dataset("shetumohanto/doctor_qa_bangla")
    
    # Print schema
    print("\n--- External Dataset Schema Discovery ---")
    print(ds)
    for split in ds.keys():
        print(f"Split '{split}' column names: {ds[split].column_names}")
        print(f"First 2 examples in split '{split}':")
        for i in range(min(2, len(ds[split]))):
            print(f"  Example {i}: {ds[split][i]}")
            
    # Auto-detect question/answer columns
    question_candidates = ["input", "question", "patient", "prompt", "instruction", "query", "text_input"]
    answer_candidates = ["output", "answer", "doctor", "response", "reply", "completion", "text_output"]
    
    # Look at train split columns
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    columns = ds[split_name].column_names
    
    matched_q = [c for c in columns if any(q in c.lower() for q in question_candidates)]
    matched_a = [c for c in columns if any(a in c.lower() for a in answer_candidates)]
    
    df_raw = ds[split_name].to_pandas()
    
    # Handle auto-detection cases
    if len(matched_q) == 1 and len(matched_a) == 1:
        q_col, a_col = matched_q[0], matched_a[0]
        print(f"Auto-detected columns: question='{q_col}', answer='{a_col}'")
        inputs = df_raw[q_col].astype(str).str.strip()
        outputs = df_raw[a_col].astype(str).str.strip()
    elif len(columns) == 1 and columns[0] == "text":
        # Known structure for this dataset
        print("Dataset contains a single 'text' column. Attempting regex parsing of [INST] pattern...")
        pattern = re.compile(r"\[INST\](.*?)(?:\[/INST\]|inst\s+turns)(.*)", re.DOTALL | re.IGNORECASE)
        inputs, outputs = [], []
        for val in df_raw["text"].astype(str):
            match = pattern.search(val)
            if match:
                q, a = match.group(1), match.group(2)
                inputs.append(q.replace("<s>", "").replace("</s>", "").strip())
                outputs.append(a.replace("<s>", "").replace("</s>", "").strip())
            else:
                inputs.append("")
                outputs.append("")
    else:
        raise ValueError(
            f"Could not auto-detect question/answer columns. Columns: {columns}. "
            f"Matched question candidates: {matched_q}, Matched answer candidates: {matched_a}"
        )
        
    # Build normalized DataFrame
    external_df = pd.DataFrame({
        "input": inputs,
        "output": outputs
    })
    
    # Generate stable IDs via MD5 of the input
    external_df["id"] = external_df["input"].apply(
        lambda x: "ext_" + hashlib.md5(x.encode("utf-8", errors="ignore")).hexdigest()[:10]
    )
    external_df["source"] = "external"
    
    # Apply quality filters
    before_len = len(external_df)
    external_df = external_df[
        (external_df["input"].str.strip() != "") & 
        (external_df["output"].str.strip() != "") &
        (external_df["input"].str.len() >= 5) &
        (external_df["output"].str.len() >= 20)
    ].copy()
    
    # Drop exact input+output duplicates
    external_df = external_df.drop_duplicates(subset=["input", "output"]).reset_index(drop=True)
    print(f"External dataset normalized: {len(external_df)} rows kept out of {before_len}")
    
    return external_df[["id", "input", "output", "source"]]

def merge_and_dedupe(official_clean_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """Merge official and external data, dropping external rows that leak official prompts."""
    # Ensure official df has source column
    off_df = official_clean_df.copy()
    if "source" not in off_df.columns:
        off_df["source"] = "official"
        
    ext_df = external_df.copy()
    
    # Normalize prompts for exact matching comparison
    official_prompts = set(off_df["input"].str.strip())
    
    # Filter external dataset to remove prompts present in official dataset
    ext_clean_mask = ~ext_df["input"].str.strip().isin(official_prompts)
    ext_kept = ext_df[ext_clean_mask].copy()
    ext_dropped_count = len(ext_df) - len(ext_kept)
    
    merged_df = pd.concat([off_df, ext_kept], ignore_index=True)
    
    print("\n=== Merge and Deduplication Summary ===")
    print(f"Official rows: {len(off_df)}")
    print(f"External rows kept: {len(ext_kept)}")
    print(f"External rows dropped (leakage prevention): {ext_dropped_count}")
    print(f"Merged total rows: {len(merged_df)}")
    
    return merged_df

def make_splits(official_clean_df: pd.DataFrame, val_size: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified-by-length train/val split of the official data only."""
    df = official_clean_df.copy()
    df["output_len"] = df["output"].apply(lambda x: len(x.split()))
    
    # Bin the length into 10 quantiles for stratification
    try:
        df["len_bin"] = pd.qcut(df["output_len"], q=10, labels=False, duplicates="drop")
    except Exception:
        # Fallback to simple uniform cut if qcut fails
        df["len_bin"] = pd.cut(df["output_len"], bins=10, labels=False)
        
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["len_bin"]
    )
    
    # Clean up temporary columns
    train_df = train_df.drop(columns=["output_len", "len_bin"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["output_len", "len_bin"]).reset_index(drop=True)
    
    print("\n=== Splits Created ===")
    print(f"SFT Train rows: {len(train_df)}")
    print(f"SFT Val rows: {len(val_df)}")
    
    return train_df, val_df
