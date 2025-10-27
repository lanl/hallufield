# 🔥 HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2509.10753-b31b1b.svg)](https://arxiv.org/abs/2509.10753)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


**HalluField** is a novel field-theoretic approach for detecting hallucinations in Large Language Models (LLMs). Inspired by thermodynamics, HalluField models an LLM's response as a collection of discrete likelihood token paths, each associated with energy and entropy, to quantify semantic stability and detect hallucinations.

## 🌟 Key Features

- **Physics-Inspired**: Grounded in thermodynamic principles, drawing analogies to the first law of thermodynamics
- **No Fine-Tuning Required**: Works directly on model output logits without auxiliary neural networks
- **State-of-the-Art Performance**: Achieves leading hallucination detection across multiple models and datasets
- **Computationally Efficient**: Practical and scalable for real-world applications
- **Multi-Dataset Support**: Tested on SQuAD, TriviaQA, Natural Questions, BioASQ, and SVAMP

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
  - [Data Generation](#1-data-generation)
  - [HalluField Computation](#2-hallufield-computation)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hallufield.git
cd hallufield
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models (optional - will auto-download on first run):
```bash
python scripts/download_models.py
```

## 🚀 Quick Start

### Basic Example

```python
from hallufield import HalluFieldDetector
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize HalluField detector
detector = HalluFieldDetector(model, tokenizer)

# Detect hallucinations
question = "What is the capital of France?"
context = "France is a country in Europe..."
result = detector.detect(question, context, num_generations=10)

print(f"Hallucination Score: {result['hallufield_score']:.4f}")
print(f"Is Hallucination: {result['is_hallucination']}")
```

### Command Line Usage

Generate responses and compute HalluField scores:

```bash
# Generate model responses
python scripts/generate.py \
    --dataset squad \
    --model_name meta-llama/Llama-2-7b-hf \
    --num_samples 100 \
    --num_generations 10

# Compute HalluField scores
python scripts/compute_hallufield.py \
    --dataset squad \
    --model_name meta-llama/Llama-2-7b-hf
```

## 📊 Pipeline Overview

HalluField operates in two main stages:

### Stage 1: Response Generation
- Generate multiple responses from the LLM at different temperatures
- Extract token-level log-likelihoods and embeddings
- Compute baseline metrics (entropy, potential)

### Stage 2: HalluField Computation
- Analyze energy landscape variations across temperatures
- Compute semantic entropy using entailment models
- Calculate HalluField score combining energy and entropy metrics
- Detect hallucinations based on semantic instability

```
Input Question → [Generation Stage] → Multiple Responses
                        ↓
                Token Likelihoods & Embeddings
                        ↓
              [Computation Stage] → Energy Analysis
                        ↓                    ↓
                Entropy Analysis    Semantic Clustering
                        ↓                    ↓
                        → HalluField Score ←
                                ↓
                        Hallucination Detection
```

## 📖 Usage

### 1. Data Generation

Generate model responses with multiple temperatures:

```python
from hallufield.core.generate import ResponseGenerator
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad")

# Initialize generator
generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-7b-hf",
    temperatures=[1.0, 1.5, 2.0, 2.5, 3.0],
    num_generations=10
)

# Generate responses
results = generator.generate(
    dataset=dataset["validation"],
    num_samples=100,
    use_context=True
)
```

### 2. HalluField Computation

Compute hallucination scores from generated responses:

```python
from hallufield.core.compute import HalluFieldComputer

# Initialize computer
computer = HalluFieldComputer(
    entailment_model="deberta",
    cache_dir="./cache"
)

# Compute HalluField scores
metrics = computer.compute(
    generations=results,
    temperatures=[1.0, 1.5, 2.0, 2.5, 3.0]
)

# Access results
print(f"AUC: {metrics['HalluField']['AUC']:.4f}")
print(f"Accuracy: {metrics['HalluField']['Accuracy']:.4f}")
```

### Batch Processing

Process multiple models and datasets:

```bash
# Run batch generation
bash scripts/batch_generate.sh

# Run batch computation
bash scripts/batch_compute.sh
```

## ⚙️ Configuration

### Model Configuration

Edit `configs/model_config.yaml`:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  max_new_tokens: 100
  device: "cuda"
  load_in_8bit: false

generation:
  temperatures: [1.0, 1.5, 2.0, 2.5, 3.0]
  num_generations: 10
  num_few_shot: 2
```

### Dataset Configuration

Edit `configs/dataset_config.yaml`:

```yaml
datasets:
  squad:
    path: "squad"
    use_context: true
    answerable_only: true
  
  trivia_qa:
    path: "trivia_qa"
    use_context: false
```

## 🤖 Supported Models

- **LLaMA Family**: LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B, LLaMA-3.1, LLaMA-3.2
- **Mistral**: Mistral-7B-v0.1, Mistral-7B-Instruct
- **Phi**: Phi-3-mini, Phi-3.5-mini
- **Falcon**: Falcon-7B-Instruct, Falcon-40B-Instruct

## 📚 Supported Datasets

- **SQuAD**: Question answering with context
- **TriviaQA**: Open-domain question answering
- **Natural Questions (NQ)**: Real questions from Google search
- **BioASQ**: Biomedical question answering
- **SVAMP**: Math word problems

## 📝 Citation

If you use HalluField in your research, please cite our paper:

```bibtex
@article{vu2025hallufield,
  title={HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling},
  author={Vu, Minh and Tran, Brian K and Shah, Syed A and Zollicoffer, Geigh and Hoang-Xuan, Nhat and Bhattarai, Manish},
  journal={arXiv preprint arXiv:2509.10753},
  year={2025}
}
```

## 🔬 Methodology

HalluField uses a field-theoretic approach inspired by thermodynamics:

1. **Energy Landscape**: Models responses as token paths with associated energies (negative log-likelihoods)
2. **Entropy Analysis**: Computes entropy at different temperatures to measure uncertainty
3. **Semantic Clustering**: Groups semantically equivalent responses using entailment models
4. **Stability Detection**: Identifies hallucinations through erratic energy landscape behavior



## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or use 8-bit quantization
python scripts/generate.py --load_in_8bit --batch_size 1
```

**Issue**: Entailment model loading fails
```bash
# Solution: Manually download DeBERTa model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v2-xlarge-mnli')"
```


## 🙏 Acknowledgments

- Built on top of [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Semantic entropy implementation inspired by [Semantic Uncertainty](https://github.com/lorenzkuhn/semantic_uncertainty)
- DeBERTa entailment model from Microsoft




**Copyright Notice**

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

LANL O#5004

**License**

This program is Open-Source under the BSD-3 License.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
