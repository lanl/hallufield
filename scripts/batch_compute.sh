#!/bin/bash
# Batch Computation Script for HalluField
# This script computes HalluField scores for generated data

# Configuration
DATASETS=("squad" "trivia_qa" "nq" "bioasq" "svamp")
MODELS=(
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-3.1-8B"
    "mistralai/Mistral-7B-v0.1"
)
DATA_DIR="./gendata"
OUTPUT_DIR="./results"
CACHE_DIR="./cache"

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
        --data_dir)
            DATA_DIR="$2"
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
echo "HalluField Batch Computation"
echo "========================================"
echo "Datasets: ${DATASETS[@]}"
echo "Models: ${MODELS[@]}"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Cache directory: $CACHE_DIR"
echo "========================================"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Process each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "Processing model: $model"
    echo "----------------------------------------"
    
    # Process each dataset
    for dataset in "${DATASETS[@]}"; do
        echo "  Dataset: $dataset"
        
        # Run computation
        python -m hallufield.core.compute \
            --data_dir "$DATA_DIR" \
            --dataset "$dataset" \
            --model_name "$model" \
            --output_dir "$OUTPUT_DIR" \
            --cache_dir "$CACHE_DIR"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed: $dataset"
        else
            echo "  ✗ Failed: $dataset"
        fi
    done
done

echo ""
echo "========================================"
echo "Batch computation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Generate summary report
echo ""
echo "Generating summary report..."
python scripts/generate_summary.py \
    --results_dir "$OUTPUT_DIR" \
    --output "summary_report.txt"
