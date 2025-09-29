"""
Metriche di valutazione per text generation
"""
import logging
from typing import List, Dict
import numpy as np
from collections import Counter
import string

# Import librerie metriche
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    logging.warning("rouge-score not installed. ROUGE metrics will not be available.")
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    from nltk.translate.bleu_score import SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    BLEU_AVAILABLE = True
except ImportError:
    logging.warning("NLTK not installed. BLEU metrics will not be available.")
    BLEU_AVAILABLE = False


def compute_generation_metrics(predictions: List[str], 
                              references: List[str],
                              metrics: List[str] = None) -> Dict:
    """
    Calcola metriche per text generation
    
    Args:
        predictions: Testi generati
        references: Testi di riferimento
        metrics: Lista metriche da calcolare (default: tutte)
    Returns:
        Dict con tutte le metriche
    """
    if metrics is None:
        metrics = ['rouge', 'bleu', 'exact_match', 'f1']
    
    results = {}
    
    if 'rouge' in metrics and ROUGE_AVAILABLE:
        rouge_scores = compute_rouge_scores(predictions, references)
        results.update(rouge_scores)
    
    if 'bleu' in metrics and BLEU_AVAILABLE:
        bleu_scores = compute_bleu_scores(predictions, references)
        results.update(bleu_scores)
    
    if 'exact_match' in metrics:
        results['exact_match'] = compute_exact_match(predictions, references)
    
    if 'f1' in metrics:
        results['f1_score'] = compute_f1_score(predictions, references)
    
    return results


def compute_rouge_scores(predictions: List[str], 
                        references: List[str]) -> Dict:
    """
    Calcola ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    
    Args:
        predictions: Testi generati
        references: Testi di riferimento
    Returns:
        Dict con ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                      use_stemmer=True)
    
    scores = {'rouge_1': [], 'rouge_2': [], 'rouge_l': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge_1'].append(score['rouge1'].fmeasure)
        scores['rouge_2'].append(score['rouge2'].fmeasure)
        scores['rouge_l'].append(score['rougeL'].fmeasure)
    
    return {
        'rouge_1': np.mean(scores['rouge_1']),
        'rouge_2': np.mean(scores['rouge_2']),
        'rouge_l': np.mean(scores['rouge_l'])
    }


def compute_bleu_scores(predictions: List[str], 
                       references: List[str]) -> Dict:
    """
    Calcola BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    
    Args:
        predictions: Testi generati
        references: Testi di riferimento
    Returns:
        Dict con BLEU scores
    """
    if not BLEU_AVAILABLE:
        return {}
    
    # Tokenizza
    tokenized_predictions = [word_tokenize(pred.lower()) for pred in predictions]
    tokenized_references = [[word_tokenize(ref.lower())] for ref in references]
    
    smoother = SmoothingFunction()
    bleu_scores = {}
    
    # Calcola BLEU per n-gram 1-4
    for n in range(1, 5):
        weights = tuple([1/n] * n + [0] * (4-n))
        scores = []
        
        for pred, ref in zip(tokenized_predictions, tokenized_references):
            score = sentence_bleu(
                ref, 
                pred, 
                weights=weights,
                smoothing_function=smoother.method1
            )
            scores.append(score)
        
        bleu_scores[f'bleu_{n}'] = np.mean(scores)
    
    # Corpus BLEU
    corpus_bleu_score = corpus_bleu(
        tokenized_references,
        tokenized_predictions,
        smoothing_function=smoother.method1
    )
    bleu_scores['corpus_bleu'] = corpus_bleu_score
    
    return bleu_scores


def compute_exact_match(predictions: List[str],
                       references: List[str]) -> float:
    """
    Calcola Exact Match accuracy
    
    Args:
        predictions: Testi generati
        references: Testi di riferimento
    Returns:
        Percentuale di match esatti
    """
    def normalize_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    matches = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_text(pred) == normalize_text(ref)
    )
    
    return matches / len(predictions) if predictions else 0.0


def compute_f1_score(predictions: List[str],
                    references: List[str]) -> float:
    """
    Calcola F1 score basato su token overlap
    
    Args:
        predictions: Testi generati
        references: Testi di riferimento
    Returns:
        F1 score medio
    """
    def get_tokens(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()
    
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = get_tokens(pred)
        ref_tokens = get_tokens(ref)
        
        # Edge cases
        if not pred_tokens and not ref_tokens:
            f1_scores.append(1.0)
            continue
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        
        # Token overlap
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            f1_scores.append(0.0)
            continue
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0