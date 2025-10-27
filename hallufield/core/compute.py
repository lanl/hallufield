"""
HalluField Computation Module

This module implements the second stage of the HalluField pipeline: computing
hallucination detection metrics based on energy and entropy analysis across
multiple temperatures.

The core idea is to model LLM responses as token paths in an energy landscape,
where hallucinations manifest as unstable or erratic behavior when temperature
and likelihood vary.

Author: HalluField Team
License: MIT
"""

import os
import logging
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

from hallufield.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta,
    get_semantic_ids,
    logsumexp_by_id,
    predictive_entropy_rao,
)
from hallufield.utils.helpers import setup_logger


# Temperature weights for multi-temperature analysis
TEMPERATURE_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]


class HalluFieldComputer:
    """
    Compute HalluField scores for hallucination detection.
    
    This class analyzes energy and entropy landscapes across multiple temperatures
    to detect semantic instability indicative of hallucinations.
    
    The HalluField score combines:
    1. Base energy at different temperatures (token log-likelihoods)
    2. Changes in potential (energy differences)
    3. Entropy variations across semantic clusters
    4. Semantic entropy from entailment-based clustering
    
    Attributes:
        entailment_model: Model for semantic entailment checking
        cache_dir: Directory for caching entailment predictions
        weights: Temperature weights for analysis
    """
    
    def __init__(
        self,
        entailment_model: str = "deberta",
        cache_dir: str = "./cache",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize HalluField computer.
        
        Args:
            entailment_model: Type of entailment model ('deberta' supported)
            cache_dir: Directory for caching computations
            weights: List of temperature weights (defaults to TEMPERATURE_WEIGHTS)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.weights = weights or TEMPERATURE_WEIGHTS
        
        setup_logger()
        
        # Load entailment model
        logging.info("Loading entailment model...")
        if entailment_model == "deberta":
            cache_path = self.cache_dir / "deberta_entailment_cache.pkl"
            self.entailment_model = EntailmentDeberta(
                entailment_cache_id=str(cache_path)
            )
        else:
            raise ValueError(f"Unsupported entailment model: {entailment_model}")
        
        logging.info("Entailment model loaded successfully")
    
    def compute_metrics(
        self,
        generations: Dict[str, Any],
        temperature: float,
        limit: int = 10000
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Compute per-item metrics for hallucination detection.
        
        Args:
            generations: Dictionary of generated responses with structure:
                {
                    'item_id': {
                        'most_likely_answer': {...},
                        'responses': [(response, log_liks, entropy, ...)],
                        'question': str,
                        'context': str
                    }
                }
            temperature: Temperature value used for generation
            limit: Maximum number of items to process
            
        Returns:
            Tuple of (metrics_df, statistics):
                - metrics_df: DataFrame with computed metrics
                - statistics: Dictionary with processing stats
        """
        # Initialize metric storage
        metrics = {
            'item_ids': [],
            'labels': [],
            'base_energy': [],
            'base_entropy': [],
            'delta_potential': [],
            'delta_potential_1st': [],
            'delta_entropy_ids': [],
            'semantic_entropy': [],
        }
        
        count = 0
        count_norepeat = 0
        
        logging.info(f"Computing metrics for temperature {temperature}")
        
        for item_id in generations.keys():
            count += 1
            if count > limit:
                break
            
            metrics['item_ids'].append(item_id)
            
            # Extract base answer (greedy/low-temperature)
            base_answer = generations[item_id]['most_likely_answer']
            label = base_answer['accuracy']
            metrics['labels'].append(label)
            
            # Compute base metrics
            base_potential = -np.mean(base_answer['token_log_likelihoods'])
            base_entropy = np.mean(base_answer['entropy'])
            
            metrics['base_energy'].append(base_potential)
            metrics['base_entropy'].append(base_entropy)
            
            # Extract sampled responses
            responses = generations[item_id]['responses']
            
            # Limit to 50 responses for computational efficiency
            if len(responses) > 50:
                responses = responses[:50]
            
            # Extract response text and log-likelihoods
            text_responses = [r[0] for r in responses]
            log_liks = [r[1] for r in responses]
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
            
            # Compute semantic entropy (only at T=1.0)
            if temperature == 1.0:
                semantic_ids = get_semantic_ids(
                    text_responses,
                    model=self.entailment_model,
                    strict_entailment=True,
                    example=generations[item_id]
                )
                
                unique_ids, log_likelihood_per_semantic_id = logsumexp_by_id(
                    semantic_ids,
                    log_liks_agg,
                    agg='sum_normalized'
                )
                
                pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            else:
                pe = 0.0
            
            metrics['semantic_entropy'].append(pe)
            
            # Compute delta metrics (changes from base)
            base_sliced_ids = base_answer['sliced_ids']
            
            delta_potentials = []
            delta_entropies = []
            
            for response in responses:
                response_text, response_log_liks, response_entropy, _, _, response_ids = response
                
                # Check if token sequence changed
                sequence_changed = not np.array_equal(response_ids, base_sliced_ids)
                
                # Compute potential and entropy for this response
                response_potential = -np.mean(response_log_liks)
                response_entropy_mean = np.mean(response_entropy)
                
                # Compute deltas (only if sequence changed)
                delta_pot = (response_potential - base_potential) if sequence_changed else 0.0
                delta_ent = (response_entropy_mean - base_entropy) if sequence_changed else 0.0
                
                delta_potentials.append(delta_pot)
                delta_entropies.append(delta_ent)
                
                if sequence_changed:
                    count_norepeat += 1
            
            # Aggregate delta metrics
            metrics['delta_potential'].append(np.mean(delta_potentials))
            metrics['delta_potential_1st'].append(np.mean(delta_potentials))
            metrics['delta_entropy_ids'].append(np.mean(delta_entropies))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Item ID': metrics['item_ids'],
            'Label': metrics['labels'],
            f'Base Energy {temperature}': metrics['base_energy'],
            f'Base Entropy {temperature}': metrics['base_entropy'],
            f'Δ1st Potential {temperature}': metrics['delta_potential'],
            f'ΔEntropy IDs {temperature}': metrics['delta_entropy_ids'],
            f'Semantic Entropy {temperature}': metrics['semantic_entropy'],
        })
        
        stats = {
            'n_items': count,
            'n_norepeat': count_norepeat,
            'norepeat_ratio': count_norepeat / count if count > 0 else 0.0
        }
        
        logging.info(f"Computed metrics for {count} items")
        logging.info(f"Non-repeat ratio: {stats['norepeat_ratio']:.3f}")
        
        return df, stats
    
    def compute_hallufield_score(
        self,
        df_merged: pd.DataFrame,
        formula: str = "default"
    ) -> pd.Series:
        """
        Compute HalluField score from multi-temperature metrics.
        
        The HalluField score combines energy and entropy variations across
        temperatures to detect semantic instability.
        
        This implementation uses the exact formulas from the paper:
        - "default" corresponds to "HalluField-B" from the paper
        - "with_semantic_entropy" corresponds to "HalluField-Sem-F12E" from the paper
        
        Args:
            df_merged: DataFrame with metrics from multiple temperatures
            formula: Formula type ('default' or 'with_semantic_entropy')
            
        Returns:
            Series with HalluField scores
            
        Note:
            These formulas require temperatures 1.0, 1.5, and 2.0 at minimum.
            The 'with_semantic_entropy' formula also requires temperature 2.5.
        """
        if formula == "default":
            # HalluField-B: Best performing formula without semantic entropy
            # From paper experiments, this formula achieved the best AUC
            score = (
                1.5 * df_merged["Base Energy 1.5"] +
                2.0 * df_merged["Base Energy 2.0"] +
                (df_merged["Δ1st Potential 1.0"] + df_merged["ΔEntropy IDs 1.0"]) +
                (df_merged["Δ1st Potential 1.5"] / 2.25 + df_merged["ΔEntropy IDs 1.5"] / 1.5)
            )
        
        elif formula == "with_semantic_entropy":
            # HalluField-Sem-F12E: Best performing formula with semantic entropy
            # This is the full HalluField formula reported in the paper
            score = (
                0.4 * (
                    2.0 * df_merged["Base Energy 2.0"] +
                    1.5 * df_merged["Base Energy 1.5"] +
                    df_merged["Base Energy 1.0"]
                ) +
                0.6 * (
                    (df_merged["Δ1st Potential 1.0"] + df_merged["ΔEntropy IDs 1.0"]) +
                    (df_merged["Δ1st Potential 1.5"] / 2.25 + df_merged["ΔEntropy IDs 1.5"] / 1.5) +
                    (df_merged["Δ1st Potential 2.0"] / 4.0 + df_merged["ΔEntropy IDs 2.0"] / 2.0)
                ) +
                2.5 * df_merged["Semantic Entropy 1.0"]
            )
        
        else:
            raise ValueError(f"Unknown formula: {formula}")
        
        return score
    
    def evaluate_metrics(
        self,
        df: pd.DataFrame,
        metric_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate hallucination detection performance for all metrics.
        
        Uses Youden's J statistic for threshold selection and computes
        standard classification metrics.
        
        Args:
            df: DataFrame with 'Label' column and metric columns
            metric_columns: List of columns to evaluate (None = all except Label/Item ID)
            
        Returns:
            Dictionary mapping metric names to performance metrics:
                {
                    'metric_name': {
                        'AUC': float,
                        'Accuracy': float,
                        'Precision': float,
                        'Recall': float,
                        'F1_Score': float,
                        'Best_Threshold': float,
                        ...
                    }
                }
        """
        if metric_columns is None:
            metric_columns = [
                col for col in df.columns
                if col not in ['Item ID', 'Label']
            ]
        
        results = {}
        
        for col in metric_columns:
            try:
                results[col] = self._compute_single_metric(
                    true_labels=df['Label'].values,
                    scores=df[col].values
                )
            except Exception as e:
                logging.warning(f"Failed to compute metrics for {col}: {e}")
                results[col] = None
        
        return results
    
    def _compute_single_metric(
        self,
        true_labels: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute classification metrics for a single score column.
        
        Uses Youden's J statistic for optimal threshold selection.
        
        Args:
            true_labels: Binary ground truth labels (0 = correct, 1 = hallucination)
            scores: Predicted scores (higher = more likely hallucination)
            
        Returns:
            Dictionary with performance metrics
        """
        true_labels = np.asarray(true_labels)
        scores = np.asarray(scores)
        
        # Handle label direction
        auc_normal = roc_auc_score(true_labels, scores)
        auc_reversed = roc_auc_score(1 - true_labels, scores)
        
        if auc_normal >= auc_reversed:
            final_labels = true_labels
            direction = "Normal"
        else:
            final_labels = 1 - true_labels
            direction = "Reversed"
        
        # Youden's J threshold selection
        fpr, tpr, thresholds = roc_curve(final_labels, scores)
        
        # Remove infinite threshold
        valid = ~np.isinf(thresholds)
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
        
        # Find best threshold
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        threshold = float(thresholds[best_idx])
        
        # Make predictions
        predictions = (scores >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'Label_Direction': direction,
            'Best_Threshold': round(threshold, 4),
            'AUC': round(roc_auc_score(final_labels, scores), 4),
            'Average_Precision': round(average_precision_score(final_labels, scores), 4),
            'Accuracy': round(accuracy_score(final_labels, predictions), 4),
            'Precision': round(precision_score(final_labels, predictions, zero_division=0), 4),
            'Recall': round(recall_score(final_labels, predictions, zero_division=0), 4),
            'F1_Score': round(f1_score(final_labels, predictions, zero_division=0), 4),
            'Confusion_Matrix': confusion_matrix(final_labels, predictions).tolist(),
        }
        
        return metrics
    
    def process_dataset(
        self,
        data_dir: str,
        dataset_name: str,
        model_name: str,
        output_dir: str = "./results",
        save_csv: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a dataset.
        
        Args:
            data_dir: Directory containing generated data
            dataset_name: Name of dataset
            model_name: Model identifier
            output_dir: Directory to save results
            save_csv: Whether to save DataFrame as CSV
            
        Returns:
            Dictionary with results including:
                - merged_df: DataFrame with all metrics
                - evaluation: Performance metrics
                - hallufield_score: HalluField scores
        """
        model_name_fmt = model_name.replace("/", "_").lower()
        
        # Load data for each temperature
        all_dfs = []
        
        for temp in self.weights:
            temp_str = str(temp).replace(".", "")
            gen_path = Path(data_dir) / f"temperature{temp_str}" / \
                      f"{dataset_name}_{model_name_fmt}_validation_temp{temp}_generations.pkl"
            
            if not gen_path.exists():
                logging.warning(f"File not found: {gen_path}")
                continue
            
            logging.info(f"Loading data from: {gen_path}")
            
            with open(gen_path, "rb") as f:
                generations = pickle.load(f)
            
            # Compute metrics
            df, stats = self.compute_metrics(generations, temp)
            all_dfs.append(df)
            
            logging.info(f"Temperature {temp}: {stats}")
        
        if not all_dfs:
            raise ValueError("No data files found")
        
        # Merge DataFrames
        df_merged = all_dfs[0].copy()
        label_series = df_merged.set_index("Item ID")["Label"]
        df_merged = df_merged.drop(columns=["Label"])
        
        for df_next in all_dfs[1:]:
            df_merged = pd.merge(
                df_merged,
                df_next.drop(columns=["Label"]),
                on="Item ID",
                how="outer"
            )
        
        # Restore labels
        df_merged["Label"] = df_merged["Item ID"].map(label_series)
        
        # Compute HalluField scores using the exact formulas from the paper
        # "HalluField" corresponds to "HalluField-B" from paper (best without semantic entropy)
        # "HalluFieldSE" corresponds to "HalluField-Sem-F12E" from paper (best with semantic entropy)
        df_merged["HalluField"] = self.compute_hallufield_score(df_merged, "default")
        df_merged["HalluFieldSE"] = self.compute_hallufield_score(df_merged, "with_semantic_entropy")
        
        logging.info(f"Final DataFrame shape: {df_merged.shape}")
        
        # Evaluate all metrics
        evaluation = self.evaluate_metrics(df_merged)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation metrics
        import json
        results_file = output_path / f"{dataset_name}_{model_name_fmt}_metrics.json"
        with open(results_file, "w") as f:
            json.dump(evaluation, f, indent=4)
        
        logging.info(f"Saved metrics to: {results_file}")
        
        # Save DataFrame
        if save_csv:
            csv_file = output_path / f"{dataset_name}_{model_name_fmt}_merged.csv"
            df_merged.to_csv(csv_file, index=False)
            logging.info(f"Saved DataFrame to: {csv_file}")
        
        return {
            'merged_df': df_merged,
            'evaluation': evaluation,
            'hallufield_score': df_merged["HalluField"].values,
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute HalluField hallucination detection metrics"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing generated data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for entailment model"
    )
    
    args = parser.parse_args()
    
    # Initialize computer
    computer = HalluFieldComputer(
        entailment_model="deberta",
        cache_dir=args.cache_dir,
    )
    
    # Process dataset
    results = computer.process_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("HALLUFIELD RESULTS")
    print("="*80)
    
    hallufield_metrics = results['evaluation'].get('HalluField', {})
    if hallufield_metrics:
        print(f"\nHalluField Performance:")
        print(f"  AUC: {hallufield_metrics['AUC']:.4f}")
        print(f"  Accuracy: {hallufield_metrics['Accuracy']:.4f}")
        print(f"  Precision: {hallufield_metrics['Precision']:.4f}")
        print(f"  Recall: {hallufield_metrics['Recall']:.4f}")
        print(f"  F1 Score: {hallufield_metrics['F1_Score']:.4f}")
    
    print("\n" + "="*80)
    logging.info("Processing complete!")


if __name__ == "__main__":
    main()
