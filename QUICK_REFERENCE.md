# HalluField Quick Reference

## 📦 Installation

```bash
pip install hallufield
# or
git clone https://github.com/yourusername/hallufield.git && cd hallufield && pip install -e .
```

## 🚀 Quick Start

### Python API

```python
from hallufield.core.generate import ResponseGenerator
from hallufield.core.compute import HalluFieldComputer
from datasets import load_dataset

# 1. Generate Responses
generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-7b-hf",
    temperatures=[1.0, 1.5, 2.0],
    num_generations=10,
    output_dir="./gendata"
)

dataset = load_dataset("squad")
results = generator.generate_responses(
    dataset=dataset["validation"],
    num_samples=100
)

# Save results
for temp in [1.0, 1.5, 2.0]:
    generator.save_results(results, "squad", "validation", temp)

# 2. Compute HalluField Scores
computer = HalluFieldComputer(
    entailment_model="deberta",
    cache_dir="./cache"
)

metrics = computer.process_dataset(
    data_dir="./gendata",
    dataset_name="squad",
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="./results"
)

# 3. View Results
print(f"AUC: {metrics['evaluation']['HalluField']['AUC']:.4f}")
print(f"Accuracy: {metrics['evaluation']['HalluField']['Accuracy']:.4f}")
```

### Command Line

```bash
# Generate
hallufield-generate \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset squad \
    --num_samples 100 \
    --temperatures 1.0 1.5 2.0

# Compute
hallufield-compute \
    --data_dir ./gendata \
    --dataset squad \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./results

# Batch Processing
bash scripts/batch_generate.sh
bash scripts/batch_compute.sh
```

## 📊 Configuration

Create `my_config.yaml`:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  load_in_8bit: true

generation:
  temperatures: [1.0, 1.5, 2.0]
  num_generations: 10
```

Use in code:

```python
import yaml

with open('my_config.yaml') as f:
    config = yaml.safe_load(f)

generator = ResponseGenerator(**config['model'], **config['generation'])
```

## 🔧 Common Tasks

### Use Different Model

```python
generator = ResponseGenerator(
    model_name="mistralai/Mistral-7B-v0.1"
)
```

### Memory-Efficient Generation

```python
generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-13b-hf",
    load_in_8bit=True  # Reduces memory by ~50%
)
```

### Multi-GPU

```python
generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto"  # Automatic distribution
)
```

### Custom Temperatures

```python
generator = ResponseGenerator(
    temperatures=[0.7, 1.0, 1.3, 1.6, 2.0]
)
```

### Process Specific Samples

```python
# First 100 samples
results = generator.generate_responses(dataset, num_samples=100)

# Specific indices
specific_indices = [0, 10, 20, 30, 40]
# (implement custom sampling in your code)
```

## 📈 Metrics & Evaluation

### Available Metrics

- `HalluField`: Main score (default formula)
- `HalluFieldSE`: Score with semantic entropy
- `Base Energy X`: Energy at temperature X
- `Δ1st Potential X`: Potential change at temperature X
- `Semantic Entropy X`: Semantic entropy at temperature X

### Access Specific Metrics

```python
# Get all metrics
all_metrics = metrics['evaluation']

# Get HalluField performance
hallu_perf = all_metrics['HalluField']
print(f"AUC: {hallu_perf['AUC']}")
print(f"Precision: {hallu_perf['Precision']}")
print(f"Recall: {hallu_perf['Recall']}")

# Get DataFrame with scores
df = metrics['merged_df']
print(df[['Item ID', 'Label', 'HalluField']].head())
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Solution 1: 8-bit quantization
generator = ResponseGenerator(..., load_in_8bit=True)

# Solution 2: Smaller model
generator = ResponseGenerator(model_name="meta-llama/Llama-2-7b-hf")

# Solution 3: Fewer generations
generator = ResponseGenerator(num_generations=5)
```

### Slow Performance

```python
# Use cache
computer = HalluFieldComputer(cache_dir="./cache")

# Limit samples
results = generator.generate_responses(dataset, num_samples=50)

# Multi-GPU
generator = ResponseGenerator(device_map="auto")
```

### Module Not Found

```bash
pip uninstall hallufield
pip install -e .
```

## 📚 Documentation

- Full docs: `README.md`
- Installation: `docs/INSTALL.md`
- Contributing: `CONTRIBUTING.md`
- Examples: `examples/basic_usage.py`
- Configuration: `configs/default_config.yaml`

## 🛠️ Development

```bash
# Setup
make install-dev

# Test
make test

# Format
make format

# Lint
make lint

# Clean
make clean
```

## 📞 Get Help

- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Email: support@hallufield.org
- Paper: https://arxiv.org/abs/2509.10753

## 🎓 Citation

```bibtex
@article{bhattarai2025hallufield,
  title={HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling},
  author={Bhattarai, Manish and others},
  journal={arXiv preprint arXiv:2509.10753},
  year={2025}
}
```

---

For complete documentation, see `README.md`
