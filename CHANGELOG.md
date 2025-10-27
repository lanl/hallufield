# Changelog

All notable changes to HalluField will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial repository structure
- Core generation and computation modules
- Support for multiple LLM architectures
- Batch processing scripts
- Comprehensive documentation

## [0.1.0] - 2025-01-XX

### Added
- Initial release of HalluField
- Response generation at multiple temperatures
- HalluField score computation
- Semantic entropy calculation
- Support for LLaMA, Mistral, Phi, and Falcon models
- Support for SQuAD, TriviaQA, NQ, BioASQ, and SVAMP datasets
- Command-line interface for generation and computation
- Python API for programmatic usage
- DeBERTa-based entailment model integration
- Multi-temperature energy landscape analysis
- Youden's J threshold selection for binary classification
- Comprehensive metrics (AUC, Precision, Recall, F1)
- 8-bit and 4-bit quantization support
- Multi-GPU support via accelerate
- Caching system for entailment predictions
- Example scripts and notebooks
- Full documentation and installation guide

### Features
- Field-theoretic hallucination detection
- Thermodynamics-inspired energy and entropy analysis
- No fine-tuning required
- Works directly on model logits
- Computationally efficient
- State-of-the-art performance

### Documentation
- README with quick start guide
- Detailed installation instructions
- API reference documentation
- Usage examples
- Contributing guidelines
- MIT License

### Infrastructure
- Setup.py for package distribution
- Requirements.txt for dependencies
- Makefile for development tasks
- GitHub Actions CI/CD (coming soon)
- Docker support (coming soon)

## [0.0.1] - 2024-12-XX

### Added
- Initial proof of concept
- Basic response generation
- Simple hallucination scoring
- Research code and experiments

---

## Version History

### Version Numbering
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, backwards compatible

### Release Schedule
- Major releases: As needed
- Minor releases: Monthly
- Patch releases: As needed

## Upgrade Guide

### From 0.0.x to 0.1.0

This is the first official release. If you were using research code:

1. Update imports:
   ```python
   # Old
   from generate import ResponseGenerator
   
   # New
   from hallufield.core.generate import ResponseGenerator
   ```

2. Update configuration:
   - Configuration is now YAML-based
   - See `configs/default_config.yaml` for structure

3. Update data paths:
   - Generated data now saved in `./gendata/` by default
   - Results saved in `./results/` by default

4. Update API calls:
   ```python
   # Old
   generator.generate(dataset, num_samples=100)
   
   # New
   generator.generate_responses(
       dataset=dataset,
       num_samples=100
   )
   ```

## Deprecation Policy

Features will be deprecated with at least one minor version warning before removal.

Example:
- Version 0.5.0: Feature marked as deprecated with warning
- Version 0.6.0: Feature still works but warning is shown
- Version 0.7.0: Feature removed

## Security

### Reporting Vulnerabilities
Please report security vulnerabilities to security@hallufield.org

### Security Updates
- Critical: Within 24 hours
- High: Within 1 week
- Medium: Within 1 month
- Low: Next minor release

## Contributors

Thank you to all contributors! See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full list.

## Links

- [Homepage](https://hallufield.org)
- [Documentation](https://docs.hallufield.org)
- [GitHub](https://github.com/yourusername/hallufield)
- [Paper](https://arxiv.org/abs/2509.10753)
- [Issues](https://github.com/yourusername/hallufield/issues)
