"""
Evaluation functions for the Multilingual Health QA project.
Computes ROUGE-1 and ROUGE-L scores against reference answers.
"""

import pandas as pd
from rouge_score import rouge_scorer


def compute_rouge(predictions, references):
    """
    Compute ROUGE-1 and ROUGE-L F1 scores.
    
    Args:
        predictions: list of predicted answers
        references: list of reference answers
    
    Returns:
        dict with rouge1 and rougeL scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores)
    }


def evaluate_by_language(predictions, references, subsets):
    """
    Compute ROUGE scores broken down by language subset.
    
    Args:
        predictions: list of predicted answers
        references: list of reference answers  
        subsets: list of language subset labels
    
    Returns:
        dict with per-language ROUGE scores
    """
    df = pd.DataFrame({
        'prediction': predictions,
        'reference': references,
        'subset': subsets
    })
    
    results = {}
    for lang in df['subset'].unique():
        lang_df = df[df['subset'] == lang]
        scores = compute_rouge(
            lang_df['prediction'].tolist(),
            lang_df['reference'].tolist()
        )
        results[lang] = scores
    
    return results
