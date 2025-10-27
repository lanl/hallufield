# Contributing to HalluField

Thank you for your interest in contributing to HalluField! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

We welcome contributions in many forms:

- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📖 Documentation improvements
- 🧪 Test coverage improvements
- 💡 Ideas and suggestions

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/hallufield.git
cd hallufield

# Add upstream remote
git remote add upstream https://github.com/original/hallufield.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 style guidelines and use automated tools to enforce consistency:

```bash
# Format code with black
black hallufield/

# Sort imports with isort
isort hallufield/

# Lint with flake8
flake8 hallufield/

# Type check with mypy
mypy hallufield/
```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Follow Google-style docstring format
- Update README.md if adding new features
- Add examples for new functionality

Example docstring:
```python
def compute_metric(data: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute hallucination metric from data.
    
    Args:
        data: Input data array of shape (n_samples, n_features)
        threshold: Decision threshold for classification
        
    Returns:
        Computed metric value between 0 and 1
        
    Raises:
        ValueError: If data is empty or invalid
        
    Example:
        >>> data = np.random.rand(100, 10)
        >>> score = compute_metric(data, threshold=0.7)
        >>> print(f"Score: {score:.4f}")
    """
    pass
```

### Testing

Write tests for all new functionality:

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=hallufield tests/

# Run specific test file
pytest tests/test_compute.py
```

Test structure:
```python
import pytest
from hallufield.core.compute import HalluFieldComputer

def test_compute_metrics():
    """Test metric computation."""
    computer = HalluFieldComputer()
    # ... test implementation

def test_invalid_input():
    """Test error handling for invalid input."""
    with pytest.raises(ValueError):
        # ... test that raises error
```

## 🔄 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### 2. Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for GPT-4 models

- Implement GPT-4 tokenizer compatibility
- Add configuration for GPT-4 API
- Update documentation with GPT-4 examples

Closes #123
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 3. Submit Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template:
   - Describe what changes you made
   - Link related issues
   - Add screenshots if applicable
   - List any breaking changes

4. Wait for review and address feedback

### 4. PR Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## 🐛 Reporting Bugs

### Before Reporting

- Check if the bug has already been reported
- Try to reproduce with the latest version
- Collect relevant information (error messages, logs, etc.)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load model '...'
2. Run generation with '....'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
 - OS: [e.g., Ubuntu 20.04]
 - Python version: [e.g., 3.9.7]
 - HalluField version: [e.g., 0.1.0]
 - CUDA version: [e.g., 11.7]

**Additional context**
Add any other context about the problem.

**Error logs**
```
[Paste error logs here]
```
```

## 💡 Feature Requests

We welcome feature suggestions! Please:

1. Check if the feature has already been requested
2. Clearly describe the feature and its use case
3. Explain why it would be useful
4. Provide examples if possible

## 📚 Documentation

### Building Documentation

```bash
cd docs
make html
```

View at `docs/_build/html/index.html`

### Documentation Structure

- `README.md`: Main documentation
- `docs/`: Detailed documentation
- `examples/`: Usage examples
- Docstrings in code

## 🧪 Testing

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py    # Shared pytest configuration
```

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/

# With coverage
pytest --cov=hallufield --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## 🔧 Development Tools

### Useful Commands

```bash
# Check code quality
make lint

# Format code
make format

# Run tests
make test

# Build documentation
make docs

# Clean build artifacts
make clean
```

## 📊 Performance Benchmarks

When adding performance-critical code:

1. Add benchmarks
2. Document performance characteristics
3. Compare with baseline

```python
import time

def benchmark_computation():
    start = time.time()
    # ... your code
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.4f}s")
```

## 🌟 Recognition

Contributors are recognized in:
- GitHub contributors page
- CONTRIBUTORS.md file
- Release notes for significant contributions

## 📞 Getting Help

- 💬 Discussions: Use GitHub Discussions for questions
- 📧 Email: your.email@example.com
- 📖 Documentation: Check docs/ directory
- 🐛 Issues: Report bugs on GitHub Issues

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone.

### Expected Behavior

- Be respectful and considerate
- Be collaborative and constructive
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment or discriminatory behavior
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

### Enforcement

Violations may result in temporary or permanent ban from the project.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HalluField! 🎉
