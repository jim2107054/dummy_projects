import pandas as pd
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer
import bert_score

def token_level_f1(pred: str, ref: str) -> float:
    """Compute bag-of-whitespace-tokens F1 between prediction and reference."""
    pred_tokens = str(pred).split()
    ref_tokens = str(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
        
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * (precision * recall) / (precision + recall)

def rouge_l_f1(pred: str, ref: str) -> float:
    """Compute ROUGE-L F1 using whitespace tokens as approximation.
    
    Stemming is not used since it is not meaningful for Bengali script.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # Note: Whitespace tokens are used by default in rouge_scorer,
    # which is an approximation of the true metric if the official grader
    # uses a Bengali-aware tokenizer.
    return scorer.score(str(ref), str(pred))["rougeL"].fmeasure

def bert_score_f1(preds: list[str], refs: list[str]) -> list[float]:
    """Compute BERTScore F1 with default Bengali model, falling back to multilingual BERT."""
    try:
        # Attempt to compute using lang="bn"
        _, _, F1 = bert_score.score(preds, refs, lang="bn", verbose=False)
        print("[INFO] BERTScore computed successfully using default Bengali model.")
        return F1.tolist()
    except Exception as e:
        # Fall back to bert-base-multilingual-cased on failure
        print(f"[WARN] BERTScore with lang='bn' failed: {e}. Falling back to 'bert-base-multilingual-cased'...")
        _, _, F1 = bert_score.score(
            preds, refs, model_type="bert-base-multilingual-cased", verbose=False
        )
        print("[INFO] BERTScore computed successfully using 'bert-base-multilingual-cased' fallback.")
        return F1.tolist()

def composite_score(preds: list[str], refs: list[str]) -> tuple[float, pd.DataFrame]:
    """Compute composite score: 0.5 * BERTScore_F1 + 0.3 * Token_F1 + 0.2 * ROUGE-L_F1."""
    token_scores = [token_level_f1(p, r) for p, r in zip(preds, refs)]
    rouge_scores = [rouge_l_f1(p, r) for p, r in zip(preds, refs)]
    bert_scores = bert_score_f1(preds, refs)
    
    composite = [
        0.5 * b + 0.3 * t + 0.2 * r
        for b, t, r in zip(bert_scores, token_scores, rouge_scores)
    ]
    
    df = pd.DataFrame({
        "prediction": preds,
        "reference": refs,
        "bert_score_f1": bert_scores,
        "token_level_f1": token_scores,
        "rouge_l_f1": rouge_scores,
        "composite_score": composite
    })
    
    return float(np.mean(composite)), df

def score_predictions_csv(pred_csv_path: str, ref_csv_path: str, id_col: str = "id", pred_col: str = "output", ref_col: str = "output") -> float:
    """Load predictions and references from CSVs, compute metrics and print detailed analysis."""
    pred_df = pd.read_csv(pred_csv_path)[[id_col, pred_col]].copy()
    ref_df = pd.read_csv(ref_csv_path)[[id_col, ref_col]].copy()
    
    pred_df[id_col] = pred_df[id_col].astype(str)
    ref_df[id_col] = ref_df[id_col].astype(str)
    
    merged = pd.merge(pred_df, ref_df, on=id_col, suffixes=("_pred", "_ref"))
    if len(merged) == 0:
        raise ValueError(f"No overlapping rows found between {pred_csv_path} and {ref_csv_path} on ID column '{id_col}'.")
        
    preds = merged[f"{pred_col}_pred"].fillna("").astype(str).tolist()
    refs = merged[f"{ref_col}_ref"].fillna("").astype(str).tolist()
    
    mean_score, df_metrics = composite_score(preds, refs)
    
    merged["bert_score_f1"] = df_metrics["bert_score_f1"]
    merged["token_level_f1"] = df_metrics["token_level_f1"]
    merged["rouge_l_f1"] = df_metrics["rouge_l_f1"]
    merged["composite_score"] = df_metrics["composite_score"]
    
    print(f"\n--- Composite Evaluation Results ---")
    print(f"Matched rows: {len(merged)}")
    print(f"Mean Composite Score: {mean_score:.4f}")
    print(f"Mean BERTScore F1:    {np.mean(merged['bert_score_f1']):.4f}")
    print(f"Mean Token Level F1:  {np.mean(merged['token_level_f1']):.4f}")
    print(f"Mean ROUGE-L F1:      {np.mean(merged['rouge_l_f1']):.4f}")
    
    print("\n--- Top 10 Lowest Scoring Examples ---")
    worst = merged.nsmallest(10, "composite_score")
    for idx, row in worst.iterrows():
        print(f"\nRow ID: {row[id_col]} | Composite Score: {row['composite_score']:.4f}")
        print(f"  BERTScore F1: {row['bert_score_f1']:.4f} | Token F1: {row['token_level_f1']:.4f} | ROUGE-L: {row['rouge_l_f1']:.4f}")
        print(f"  Reference:  {row[f'{ref_col}_ref']}")
        print(f"  Prediction: {row[f'{pred_col}_pred']}")
        
    return mean_score
