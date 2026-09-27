"""
Evaluation Metrics for Amazon ML Challenge 2026.
Entity Resolution Challenge (Macro-Averaged F_0.5 Score).
"""

import numpy as np
from typing import List, Dict, Set

def calculate_f05_score(true_matches: List[str], predicted_matches: List[str]) -> float:
    """
    Calculates the F_0.5 score for a single Source 1 entity.
    
    Args:
        true_matches: Ground truth list of matching Source 2/Source 3 entity IDs.
        predicted_matches: Predicted list of matching Source 2/Source 3 entity IDs.
        
    Returns:
        float: The F_0.5 score for this entity.
    """
    true_set = set(true_matches)
    pred_set = set(predicted_matches)
    
    # Handle singletons (entities with no true matches)
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0  # Correctly predicted no matches
        else:
            return 0.0  # False merge on a singleton
            
    # Handle cases where model predicts no matches but true matches exist
    if len(pred_set) == 0:
        return 0.0
        
    true_positives = len(true_set.intersection(pred_set))
    
    precision = true_positives / len(pred_set)
    recall = true_positives / len(true_set)
    
    if precision + recall == 0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def calculate_macro_f05(ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]) -> float:
    """
    Calculates the macro-averaged F_0.5 score across all Source 1 entities.
    
    Args:
        ground_truth: Dictionary mapping Source 1 entity_id to a list of true matching IDs.
        predictions: Dictionary mapping Source 1 entity_id to a list of predicted matching IDs.
        
    Returns:
        float: The macro-averaged F_0.5 score.
    """
    scores = []
    
    for s1_entity, true_matches in ground_truth.items():
        # Ensure we have predictions for this entity
        pred_matches = predictions.get(s1_entity, [])
        score = calculate_f05_score(true_matches, pred_matches)
        scores.append(score)
        
    if not scores:
        return 0.0
        
    return float(np.mean(scores))
