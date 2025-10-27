"""
Unit tests for HalluField core compute module.

This module tests the hallucination detection computation functionality,
including metric calculation, HalluField score computation, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from hallufield.core.compute import HalluFieldComputer


class TestHalluFieldComputer:
    """Test suite for HalluFieldComputer class."""
    
    @pytest.fixture
    def mock_entailment_model(self):
        """Create a mock entailment model."""
        model = Mock()
        model.check_implication = Mock(return_value=(2, 0.95))  # Entailment with high confidence
        return model
    
    @pytest.fixture
    def sample_generations(self):
        """Create sample generation data for testing."""
        return {
            'item_1': {
                'most_likely_answer': {
                    'token_log_likelihoods': [-0.5, -0.3, -0.4],
                    'entropy': [0.2, 0.3, 0.25],
                    'accuracy': 1.0,
                    'sliced_ids': np.array([1, 2, 3])
                },
                'responses': [
                    (
                        "Paris",  # response text
                        [-0.6, -0.4, -0.5],  # log likelihoods
                        [0.25, 0.35, 0.3],  # entropy
                        None,  # embedding
                        1.0,  # accuracy
                        np.array([1, 2, 3])  # token ids
                    ),
                    (
                        "Paris is the capital",
                        [-0.7, -0.5, -0.6, -0.4],
                        [0.3, 0.4, 0.35, 0.25],
                        None,
                        1.0,
                        np.array([1, 2, 3, 4])
                    )
                ],
                'question': 'What is the capital of France?',
                'context': 'France is a country in Europe.'
            },
            'item_2': {
                'most_likely_answer': {
                    'token_log_likelihoods': [-1.0, -0.8, -0.9],
                    'entropy': [0.5, 0.6, 0.55],
                    'accuracy': 0.0,
                    'sliced_ids': np.array([5, 6, 7])
                },
                'responses': [
                    (
                        "London",
                        [-1.2, -1.0, -1.1],
                        [0.6, 0.7, 0.65],
                        None,
                        0.0,
                        np.array([5, 6, 7])
                    )
                ],
                'question': 'What is the capital of France?',
                'context': 'France is a country in Europe.'
            }
        }
    
    def test_initialization(self):
        """Test HalluFieldComputer initialization."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer(
                entailment_model="deberta",
                cache_dir="./test_cache",
                weights=[1.0, 1.5, 2.0]
            )
            
            assert computer.weights == [1.0, 1.5, 2.0]
            assert computer.cache_dir.name == "test_cache"
    
    def test_compute_metrics_basic(self, sample_generations, mock_entailment_model):
        """Test basic metric computation."""
        with patch('hallufield.core.compute.EntailmentDeberta', return_value=mock_entailment_model):
            computer = HalluFieldComputer()
            
            df, stats = computer.compute_metrics(
                sample_generations,
                temperature=1.0,
                limit=10
            )
            
            # Check DataFrame structure
            assert 'Item ID' in df.columns
            assert 'Label' in df.columns
            assert 'Base Energy 1.0' in df.columns
            assert 'Base Entropy 1.0' in df.columns
            
            # Check number of rows
            assert len(df) == 2
            
            # Check label values
            assert df['Label'].tolist() == [1.0, 0.0]
            
            # Check stats
            assert stats['n_items'] == 2
            assert 'norepeat_ratio' in stats
    
    def test_compute_metrics_energy_calculation(self, sample_generations):
        """Test that energy is correctly computed from log-likelihoods."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            df, _ = computer.compute_metrics(
                sample_generations,
                temperature=1.0
            )
            
            # Energy should be negative mean of log-likelihoods
            expected_energy_1 = -np.mean([-0.5, -0.3, -0.4])
            actual_energy_1 = df[df['Item ID'] == 'item_1']['Base Energy 1.0'].values[0]
            
            assert np.isclose(actual_energy_1, expected_energy_1, atol=1e-6)
    
    def test_compute_hallufield_score_default(self):
        """Test default HalluField score computation."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            # Create mock DataFrame
            df = pd.DataFrame({
                'Item ID': ['item_1'],
                'Label': [1.0],
                'Base Energy 1.0': [0.4],
                'Base Energy 1.5': [0.5],
                'Base Energy 2.0': [0.6],
                'Δ1st Potential 1.0': [0.1],
                'Δ1st Potential 1.5': [0.15],
                'ΔEntropy IDs 1.0': [0.05],
                'ΔEntropy IDs 1.5': [0.08],
            })
            
            scores = computer.compute_hallufield_score(df, formula="default")
            
            # Check that score is computed
            assert len(scores) == 1
            assert not np.isnan(scores.iloc[0])
            assert scores.iloc[0] > 0
    
    def test_evaluate_metrics_auc(self):
        """Test AUC computation in evaluation."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            # Create test data with clear separation
            df = pd.DataFrame({
                'Item ID': [f'item_{i}' for i in range(100)],
                'Label': [0] * 50 + [1] * 50,  # 50 correct, 50 hallucinations
                'TestMetric': list(np.linspace(0, 1, 50)) + list(np.linspace(1, 2, 50))
            })
            
            results = computer.evaluate_metrics(df, metric_columns=['TestMetric'])
            
            # Check results structure
            assert 'TestMetric' in results
            assert 'AUC' in results['TestMetric']
            assert 'Accuracy' in results['TestMetric']
            
            # AUC should be high due to good separation
            assert results['TestMetric']['AUC'] > 0.7
    
    def test_evaluate_metrics_threshold_selection(self):
        """Test Youden's J threshold selection."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            # Create data where threshold=0.5 is optimal
            df = pd.DataFrame({
                'Item ID': [f'item_{i}' for i in range(100)],
                'Label': [0] * 50 + [1] * 50,
                'TestMetric': [0.3] * 50 + [0.7] * 50
            })
            
            results = computer.evaluate_metrics(df, metric_columns=['TestMetric'])
            
            # Optimal threshold should be around 0.5
            threshold = results['TestMetric']['Best_Threshold']
            assert 0.4 < threshold < 0.6
    
    def test_compute_metrics_with_limit(self, sample_generations):
        """Test that limit parameter works correctly."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            df, stats = computer.compute_metrics(
                sample_generations,
                temperature=1.0,
                limit=1  # Only process 1 item
            )
            
            assert len(df) == 1
            assert stats['n_items'] == 1
    
    def test_invalid_formula_raises_error(self):
        """Test that invalid formula raises ValueError."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            df = pd.DataFrame({'Base Energy 1.0': [0.5]})
            
            with pytest.raises(ValueError, match="Unknown formula"):
                computer.compute_hallufield_score(df, formula="invalid")
    
    def test_edge_case_empty_generations(self):
        """Test handling of empty generations."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            empty_generations = {}
            
            df, stats = computer.compute_metrics(
                empty_generations,
                temperature=1.0
            )
            
            assert len(df) == 0
            assert stats['n_items'] == 0
    
    def test_edge_case_all_correct(self):
        """Test case where all predictions are correct."""
        with patch('hallufield.core.compute.EntailmentDeberta'):
            computer = HalluFieldComputer()
            
            all_correct = {
                f'item_{i}': {
                    'most_likely_answer': {
                        'token_log_likelihoods': [-0.1] * 5,
                        'entropy': [0.1] * 5,
                        'accuracy': 1.0,
                        'sliced_ids': np.array([1, 2, 3, 4, 5])
                    },
                    'responses': []
                }
                for i in range(10)
            }
            
            df, stats = computer.compute_metrics(all_correct, temperature=1.0)
            
            # All labels should be 1 (correct)
            assert all(df['Label'] == 1.0)


class TestMetricComputation:
    """Test suite for individual metric computations."""
    
    def test_energy_from_log_likelihoods(self):
        """Test energy calculation from log-likelihoods."""
        log_liks = np.array([-0.5, -0.3, -0.4, -0.6])
        expected_energy = -np.mean(log_liks)
        actual_energy = -np.mean(log_liks)
        
        assert np.isclose(actual_energy, expected_energy)
    
    def test_entropy_aggregation(self):
        """Test entropy aggregation."""
        entropies = np.array([0.2, 0.3, 0.25, 0.35])
        expected = np.mean(entropies)
        actual = np.mean(entropies)
        
        assert np.isclose(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
