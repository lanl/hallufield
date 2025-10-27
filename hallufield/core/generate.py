"""
HalluField Response Generation Module

This module handles the first stage of the HalluField pipeline: generating
multiple responses from LLMs at various temperatures to collect token-level
statistics for hallucination detection.

Author: HalluField Team
License: MIT
"""

import gc
import os
import logging
import random
import re
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import evaluate

from hallufield.utils.helpers import (
    setup_logger,
    get_reference,
    remove_space_grammar,
)
from hallufield.utils.prompts import (
    BRIEF_PROMPTS,
    construct_fewshot_prompt_from_indices,
    construct_p_true_few_shot_prompt,
)
from hallufield.utils.metrics import create_metric_function
from hallufield.models.huggingface_models import HuggingfaceModel


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


class ResponseGenerator:
    """
    Generate multiple responses from LLMs for hallucination detection.
    
    This class handles:
    - Loading and configuring language models
    - Generating responses at multiple temperatures
    - Extracting token-level log-likelihoods and embeddings
    - Computing baseline entropy and potential metrics
    
    Attributes:
        model: The language model instance
        tokenizer: The model's tokenizer
        temperatures: List of temperatures for generation
        num_generations: Number of responses per temperature
        output_dir: Directory for saving results
    """
    
    def __init__(
        self,
        model_name: str,
        temperatures: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
        num_generations: int = 10,
        max_new_tokens: int = 100,
        output_dir: str = "./gendata",
        device: str = "cuda",
        load_in_8bit: bool = False,
        random_seed: int = 42,
    ):
        """
        Initialize the response generator.
        
        Args:
            model_name: HuggingFace model identifier
            temperatures: List of sampling temperatures
            num_generations: Number of responses to generate per sample
            max_new_tokens: Maximum tokens in generated response
            output_dir: Directory to save generated data
            device: Device to run model on ('cuda' or 'cpu')
            load_in_8bit: Whether to load model in 8-bit precision
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.temperatures = temperatures
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.output_dir = Path(output_dir)
        self.device = device
        self.random_seed = random_seed
        
        # Create output directories
        for temp in temperatures:
            temp_dir = self.output_dir / f"temperature{str(temp).replace('.', '')}"
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_global_seed(random_seed)
        
        # Setup logging
        setup_logger()
        
        # Load model
        logging.info(f"Loading model: {model_name}")
        self.model = HuggingfaceModel(
            model_name,
            max_new_tokens=max_new_tokens,
            device=device,
            load_in_8bit=load_in_8bit,
        )
        
        logging.info("Model loaded successfully")
    
    def generate_responses(
        self,
        dataset: Dataset,
        num_samples: int,
        use_context: bool = True,
        num_few_shot: int = 2,
        brief_prompt: str = "default",
        compute_p_true: bool = False,
        p_true_num_fewshot: int = 20,
    ) -> Dict[str, Any]:
        """
        Generate responses for a dataset.
        
        Args:
            dataset: HuggingFace dataset with 'question' and 'context' fields
            num_samples: Number of samples to process
            use_context: Whether to include context in prompts
            num_few_shot: Number of few-shot examples
            brief_prompt: Type of brief prompt to use
            compute_p_true: Whether to compute P(True) metric
            p_true_num_fewshot: Number of examples for P(True) few-shot
            
        Returns:
            Dictionary containing:
                - generations: Generated responses with metadata
                - experiment_details: Configuration and prompt information
                - accuracies: List of accuracy scores
        """
        # Split dataset into answerable/unanswerable
        answerable_indices, unanswerable_indices = self._split_dataset(dataset)
        
        # Select few-shot examples
        prompt_indices = random.sample(answerable_indices, num_few_shot)
        remaining_answerable = list(set(answerable_indices) - set(prompt_indices))
        
        # Create prompt template
        BRIEF = BRIEF_PROMPTS[brief_prompt]
        make_prompt = self._get_make_prompt_fn(use_context)
        
        prompt = construct_fewshot_prompt_from_indices(
            dataset, prompt_indices, BRIEF, False, make_prompt
        )
        
        # Handle P(True) computation if requested
        p_true_data = None
        if compute_p_true:
            p_true_data = self._compute_p_true_prompt(
                dataset, remaining_answerable, prompt, BRIEF,
                make_prompt, p_true_num_fewshot
            )
            remaining_answerable = list(
                set(remaining_answerable) - set(p_true_data['indices'])
            )
        
        # Generate responses
        generations = {}
        accuracies = []
        
        # Select samples to process
        possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        k = min(num_samples, len(possible_indices))
        sample_indices = random.sample(possible_indices, k) if k > 0 else []
        
        # Create metric function
        metric_fn = create_metric_function("squad_v2")
        
        logging.info(f"Generating responses for {len(sample_indices)} samples")
        
        for idx in tqdm(sample_indices, desc="Generating responses"):
            example = dataset[idx]
            question = example["question"]
            context = example.get("context", "")
            
            generations[example['id']] = {
                'question': question,
                'context': context
            }
            
            # Create prompt for this example
            current_input = make_prompt(context, question, None, BRIEF, False)
            local_prompt = prompt + current_input
            local_prompt = remove_space_grammar(local_prompt)
            
            # Generate multiple responses
            full_responses = []
            
            for gen_idx in range(self.num_generations + 1):
                # First generation uses low temperature (greedy-ish)
                temperature = 0.1 if gen_idx == 0 else 1.0
                
                # Generate response
                result = self.model.predict(
                    local_prompt,
                    temperature,
                    return_full_output=True
                )
                
                predicted_answer = result['response']
                token_log_likelihoods = result['token_log_likelihoods']
                embedding = result.get('embedding')
                entropy = result['entropy']
                sliced_ids = result['token_ids']
                
                # Move embedding to CPU
                if embedding is not None:
                    embedding = embedding.cpu()
                
                # Compute accuracy
                acc = metric_fn(predicted_answer, example, self.model)
                
                if gen_idx == 0:
                    # Store base response (greedy)
                    base_sliced_ids = sliced_ids
                    accuracies.append(acc)
                    
                    generations[example['id']]['most_likely_answer'] = {
                        'prompt': local_prompt,
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'entropy': entropy.cpu().numpy(),
                        'embedding': embedding,
                        'accuracy': acc,
                        'sliced_ids': np.asarray(sliced_ids.cpu())
                    }
                    generations[example['id']]['reference'] = get_reference(example)
                else:
                    # Store sampled response
                    full_responses.append((
                        predicted_answer,
                        token_log_likelihoods,
                        entropy.cpu().numpy(),
                        embedding,
                        acc,
                        np.asarray(sliced_ids.cpu())
                    ))
            
            generations[example['id']]['responses'] = full_responses
            
            # Periodic cleanup
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Compute overall accuracy
        accuracy = np.mean(accuracies) if accuracies else 0.0
        logging.info(f"Overall accuracy: {accuracy:.4f}")
        
        # Prepare experiment details
        experiment_details = {
            'model_name': self.model_name,
            'num_samples': num_samples,
            'num_generations': self.num_generations,
            'temperatures': self.temperatures,
            'prompt_indices': prompt_indices,
            'prompt': prompt,
            'BRIEF': BRIEF,
            'accuracy': accuracy,
            'sample_indices': sample_indices,
        }
        
        if p_true_data:
            experiment_details['p_true'] = p_true_data
        
        return {
            'generations': generations,
            'experiment_details': experiment_details,
            'accuracies': accuracies,
        }
    
    def _split_dataset(
        self,
        dataset: Dataset
    ) -> Tuple[List[int], List[int]]:
        """
        Split dataset into answerable and unanswerable questions.
        
        Args:
            dataset: Dataset to split
            
        Returns:
            Tuple of (answerable_indices, unanswerable_indices)
        """
        def has_answer(ex):
            return len(ex["answers"]["text"]) > 0
        
        answerable = [i for i, ex in enumerate(dataset) if has_answer(ex)]
        unanswerable = [i for i, ex in enumerate(dataset) if not has_answer(ex)]
        
        return answerable, unanswerable
    
    def _get_make_prompt_fn(self, use_context: bool):
        """
        Create prompt formatting function.
        
        Args:
            use_context: Whether to include context in prompts
            
        Returns:
            Function to format prompts
        """
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if use_context and context:
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
        
        return make_prompt
    
    def _compute_p_true_prompt(
        self,
        dataset: Dataset,
        indices: List[int],
        prompt: str,
        brief: str,
        make_prompt,
        num_fewshot: int
    ) -> Dict[str, Any]:
        """
        Compute P(True) few-shot prompt.
        
        Args:
            dataset: Dataset
            indices: Available indices
            prompt: Base prompt
            brief: Brief instruction
            make_prompt: Prompt formatting function
            num_fewshot: Number of few-shot examples
            
        Returns:
            Dictionary with P(True) prompt and metadata
        """
        p_true_indices = random.sample(indices, num_fewshot)
        
        p_true_prompt, p_true_responses, len_p_true = \
            construct_p_true_few_shot_prompt(
                model=self.model,
                dataset=dataset,
                indices=p_true_indices,
                prompt=prompt,
                brief=brief,
                brief_always=False,
                make_prompt=make_prompt,
                num_generations=self.num_generations,
                metric=create_metric_function("squad_v2")
            )
        
        return {
            'indices': p_true_indices,
            'prompt': p_true_prompt,
            'responses': p_true_responses,
            'length': len_p_true,
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        dataset_name: str,
        split: str = "validation",
        temperature: float = 1.0
    ) -> None:
        """
        Save generation results to disk.
        
        Args:
            results: Results dictionary from generate_responses()
            dataset_name: Name of dataset
            split: Dataset split ('train' or 'validation')
            temperature: Temperature used for generation
        """
        model_name_fmt = self.model_name.replace("/", "_").lower()
        temp_str = str(temperature).replace(".", "")
        
        # Save to temperature-specific directory
        output_dir = self.output_dir / f"temperature{temp_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generations
        gen_path = output_dir / f"{dataset_name}_{model_name_fmt}_{split}_temp{temperature}_generations.pkl"
        with open(gen_path, 'wb') as f:
            pickle.dump(results['generations'], f)
        
        logging.info(f"Saved generations to: {gen_path}")
        
        # Save experiment details
        detail_path = output_dir / f"{dataset_name}_{model_name_fmt}_temp{temperature}_experiment_details.pkl"
        with open(detail_path, 'wb') as f:
            pickle.dump(results['experiment_details'], f)
        
        logging.info(f"Saved experiment details to: {detail_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate LLM responses for hallucination detection"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["squad", "trivia_qa", "nq", "bioasq", "svamp"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of generations per sample"
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help="Temperatures for generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gendata",
        help="Output directory"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    from datasets import load_dataset as hf_load_dataset
    dataset = hf_load_dataset(args.dataset)
    
    # Initialize generator
    generator = ResponseGenerator(
        model_name=args.model_name,
        temperatures=args.temperatures,
        num_generations=args.num_generations,
        output_dir=args.output_dir,
        load_in_8bit=args.load_in_8bit,
        random_seed=args.random_seed,
    )
    
    # Generate for each temperature
    for temp in args.temperatures:
        logging.info(f"Generating at temperature: {temp}")
        
        results = generator.generate_responses(
            dataset=dataset["validation"],
            num_samples=args.num_samples,
        )
        
        generator.save_results(
            results=results,
            dataset_name=args.dataset,
            split="validation",
            temperature=temp
        )
    
    logging.info("Generation complete!")


if __name__ == "__main__":
    main()
