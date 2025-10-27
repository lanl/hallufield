#!/bin/bash
# Batch Generation Script for HalluField
# This script generates responses for multiple models and datasets

# Configuration
DATASETS=("squad" "trivia_qa" "nq" "bioasq" "svamp")
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.1"
)
TEMPERATURES=(1.0 1.5 2.0 2.5 3.0)
NUM_SAMPLES=100
NUM_GENERATIONS=10
OUTPUT_DIR="./gendata"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASETS=("$2")
            shift 2
            ;;
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "HalluField Batch Generation"
echo "========================================"
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Temperatures: ${TEMPERATURES[@]}"
echo "Samples per dataset: $NUM_SAMPLES"
echo "Generations per sample: $NUM_GENERATIONS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "Processing model: $model"
    echo "----------------------------------------"
    
    # Process each dataset
    for dataset in "${DATASETS[@]}"; do
        echo "  Dataset: $dataset"
        
        # Run generation for all temperatures
        python -m hallufield.core.generate \
            --model_name "$model" \
            --dataset "$dataset" \
            --num_samples "$NUM_SAMPLES" \
            --num_generations "$NUM_GENERATIONS" \
            --temperatures ${TEMPERATURES[@]} \
            --output_dir "$OUTPUT_DIR" \
            --load_in_8bit
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed: $dataset"
        else
            echo "  ✗ Failed: $dataset"
        fi
    done
done

echo ""
echo "========================================"
echo "Batch generation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
