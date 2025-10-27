# Installation Guide

This guide provides detailed instructions for installing HalluField on various platforms and configurations.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 16GB+ VRAM (recommended)
- **RAM**: 16GB+ (32GB+ recommended for large models)
- **Storage**: 50GB+ free space for models and data

### CUDA Setup (for GPU)

```bash
# Check CUDA version
nvcc --version

# Install CUDA 11.7+ if needed
# Follow: https://developer.nvidia.com/cuda-downloads

# Verify PyTorch can access GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Quick Install

### Option 1: Install from PyPI (Recommended)

```bash
pip install hallufield
```

### Option 2: Install from Source

```bash
# Clone repository
git clone https://github.com/yourusername/hallufield.git
cd hallufield

# Install in development mode
pip install -e .
```

## 📦 Detailed Installation

### Step 1: Create Virtual Environment

We strongly recommend using a virtual environment:

#### Using venv

```bash
# Create environment
python -m venv hallufield-env

# Activate on Linux/macOS
source hallufield-env/bin/activate

# Activate on Windows
hallufield-env\Scripts\activate
```

#### Using conda

```bash
# Create environment
conda create -n hallufield python=3.9

# Activate
conda activate hallufield
```

### Step 2: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.7
pip install torch --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install HalluField

```bash
# Basic installation
pip install hallufield

# Or from source
git clone https://github.com/yourusername/hallufield.git
cd hallufield
pip install -e .
```

### Step 4: Install Optional Dependencies

```bash
# For development
pip install hallufield[dev]

# For visualization
pip install hallufield[viz]

# Install all extras
pip install hallufield[dev,viz]
```

## 🎯 Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    build-essential \
    git \
    wget

# Install CUDA (if using GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Continue with standard installation
pip install hallufield
```

### macOS

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Install HalluField (CPU only for M1/M2 Macs)
pip install hallufield
```

### Windows

```powershell
# Install Python from python.org
# Download and install from: https://www.python.org/downloads/

# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Install CUDA Toolkit (for GPU)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install HalluField
pip install hallufield
```

## 🔧 Advanced Configuration

### Using 8-bit Quantization (Memory Efficient)

```bash
# Install bitsandbytes
pip install bitsandbytes

# Use in code
from hallufield.core.generate import ResponseGenerator

generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-13b-hf",
    load_in_8bit=True  # Reduces memory by ~50%
)
```

### Multi-GPU Setup

```bash
# Install accelerate
pip install accelerate

# Configure
python -m accelerate.commands.config

# Use in code
generator = ResponseGenerator(
    model_name="meta-llama/Llama-2-70b-hf",
    device_map="auto"  # Automatically distribute across GPUs
)
```

### Custom Model Cache Directory

```bash
# Set environment variable
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/hf_home

# Or in Python
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'
```

## ✅ Verify Installation

```python
# Test basic import
python -c "import hallufield; print(hallufield.__version__)"

# Test GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Run quick test
python -m pytest tests/ -v
```

## 📥 Downloading Models

HalluField will automatically download models on first use. To pre-download:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download LLaMA-2-7B
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download DeBERTa entailment model
from transformers import AutoModel
entailment_model = AutoModel.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
```

## 🐛 Troubleshooting

### Issue: CUDA out of memory

**Solution 1**: Use 8-bit quantization
```python
generator = ResponseGenerator(..., load_in_8bit=True)
```

**Solution 2**: Reduce batch size
```python
generator.generate_responses(..., batch_size=1)
```

**Solution 3**: Use smaller model
```python
generator = ResponseGenerator(model_name="meta-llama/Llama-2-7b-hf")
```

### Issue: Module not found

**Solution**: Reinstall in development mode
```bash
pip uninstall hallufield
pip install -e .
```

### Issue: Slow model loading

**Solution**: Use model cache
```bash
export TRANSFORMERS_CACHE=/fast/storage/path
```

### Issue: Permission denied on Linux

**Solution**: Use user installation
```bash
pip install --user hallufield
```

### Issue: SSL certificate error

**Solution**: Disable SSL verification (not recommended for production)
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org hallufield
```

## 🔄 Updating

```bash
# Update from PyPI
pip install --upgrade hallufield

# Update from source
cd hallufield
git pull
pip install -e . --upgrade
```

## 🗑️ Uninstallation

```bash
# Uninstall package
pip uninstall hallufield

# Remove cache (optional)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/hallufield
```

## 💾 Storage Requirements

| Component | Size | Purpose |
|-----------|------|---------|
| HalluField package | ~50MB | Core code |
| LLaMA-2-7B | ~13GB | Language model |
| LLaMA-2-13B | ~26GB | Language model |
| DeBERTa-XLarge | ~1.5GB | Entailment model |
| Generated data (per 1000 samples) | ~2-5GB | Response cache |

## 🌐 Offline Installation

```bash
# On machine with internet:
pip download hallufield -d hallufield_packages/
pip download torch torchvision torchaudio -d hallufield_packages/

# Transfer hallufield_packages/ to offline machine

# On offline machine:
pip install --no-index --find-links hallufield_packages/ hallufield
```

## 📧 Getting Help

If you encounter issues:

1. Check [Troubleshooting](#-troubleshooting) section
2. Search [GitHub Issues](https://github.com/yourusername/hallufield/issues)
3. Ask on [Discussions](https://github.com/yourusername/hallufield/discussions)
4. Email: support@hallufield.org

## 🎓 Next Steps

After installation:

1. Read the [Quick Start Guide](README.md#quick-start)
2. Run the [Basic Example](examples/basic_usage.py)
3. Explore the [API Documentation](docs/API.md)
4. Check out [Advanced Usage](docs/ADVANCED.md)

---

Happy hallucination detecting! 🔥
