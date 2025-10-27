"""
Example: Basic HalluField Usage

This example demonstrates the complete HalluField pipeline:
1. Generate responses from an LLM
2. Compute HalluField scores
3. Detect hallucinations

Author: HalluField Team
"""

from hallufield.core.generate import ResponseGenerator
from hallufield.core.compute import HalluFieldComputer
from datasets import load_dataset

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DATASET_NAME = "squad"
    NUM_SAMPLES = 10  # Small number for quick demo
    
    print("="*80)
    print("HalluField Example: Detecting LLM Hallucinations")
    print("="*80)
    
    # Step 1: Load Dataset
    print("\n[1/4] Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    print(f"✓ Loaded {DATASET_NAME} dataset")
    
    # Step 2: Generate Responses
    print("\n[2/4] Generating responses...")
    generator = ResponseGenerator(
        model_name=MODEL_NAME,
        temperatures=[1.0, 1.5, 2.0],  # Fewer temperatures for demo
        num_generations=5,  # Fewer generations for demo
        output_dir="./demo_data",
        load_in_8bit=True,  # Use 8-bit for memory efficiency
    )
    
    results = generator.generate_responses(
        dataset=dataset["validation"],
        num_samples=NUM_SAMPLES,
        use_context=True,
        num_few_shot=2,
    )
    
    print(f"✓ Generated {NUM_SAMPLES} samples")
    print(f"  Overall accuracy: {results['experiment_details']['accuracy']:.2%}")
    
    # Save results for each temperature
    for temp in [1.0, 1.5, 2.0]:
        generator.save_results(
            results=results,
            dataset_name=DATASET_NAME,
            split="validation",
            temperature=temp
        )
    
    # Step 3: Compute HalluField Scores
    print("\n[3/4] Computing HalluField scores...")
    computer = HalluFieldComputer(
        entailment_model="deberta",
        cache_dir="./demo_cache",
        weights=[1.0, 1.5, 2.0],
    )
    
    hallufield_results = computer.process_dataset(
        data_dir="./demo_data",
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        output_dir="./demo_results",
    )
    
    print("✓ Computed HalluField scores")
    
    # Step 4: Display Results
    print("\n[4/4] Results Summary")
    print("="*80)
    
    # Get HalluField metrics
    hallufield_metrics = hallufield_results['evaluation'].get('HalluField', {})
    
    if hallufield_metrics:
        print("\nHalluField Performance:")
        print(f"  AUC-ROC:           {hallufield_metrics['AUC']:.4f}")
        print(f"  Accuracy:          {hallufield_metrics['Accuracy']:.4f}")
        print(f"  Precision:         {hallufield_metrics['Precision']:.4f}")
        print(f"  Recall:            {hallufield_metrics['Recall']:.4f}")
        print(f"  F1 Score:          {hallufield_metrics['F1_Score']:.4f}")
        print(f"  Optimal Threshold: {hallufield_metrics['Best_Threshold']:.4f}")
    
    # Display per-sample results
    print("\nPer-Sample Analysis (first 5 samples):")
    print("-" * 80)
    
    df = hallufield_results['merged_df']
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        item_id = row['Item ID']
        label = row['Label']
        score = row['HalluField']
        
        # Get the original question if available
        gen_data = results['generations'].get(item_id, {})
        question = gen_data.get('question', 'N/A')
        
        status = "✓ Correct" if label == 0 else "✗ Hallucination"
        
        print(f"\nSample {idx + 1}:")
        print(f"  Question: {question[:60]}...")
        print(f"  Status: {status}")
        print(f"  HalluField Score: {score:.4f}")
    
    print("\n" + "="*80)
    print("Demo complete! Check ./demo_results/ for detailed outputs.")
    print("="*80)


if __name__ == "__main__":
    main()
