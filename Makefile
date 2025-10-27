.PHONY: help install install-dev test lint format clean docs build upload

# Default target
.DEFAULT_GOAL := help

# Color output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)HalluField Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install package in production mode
	@echo "$(BLUE)Installing HalluField...$(NC)"
	pip install -e .
	@echo "$(GREEN)✓ Installation complete$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing HalluField with dev dependencies...$(NC)"
	pip install -e ".[dev,viz]"
	pre-commit install
	@echo "$(GREEN)✓ Development installation complete$(NC)"

test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=hallufield --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

test-quick: ## Run tests without coverage
	@echo "$(BLUE)Running quick tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	flake8 hallufield/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 hallufield/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy hallufield/
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black hallufield/ tests/ examples/
	isort hallufield/ tests/ examples/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

format-check: ## Check code formatting without modifying
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check hallufield/ tests/ examples/
	isort --check-only hallufield/ tests/ examples/

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built: docs/_build/html/index.html$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	python -m http.server 8000 -d docs/_build/html

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	python -m build
	@echo "$(GREEN)✓ Build complete: dist/$(NC)"

upload-test: build ## Upload to TestPyPI
	@echo "$(BLUE)Uploading to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Upload to TestPyPI complete$(NC)"

upload: build ## Upload to PyPI
	@echo "$(RED)⚠ Uploading to PyPI (production)$(NC)"
	python -m twine upload dist/*
	@echo "$(GREEN)✓ Upload to PyPI complete$(NC)"

download-models: ## Download required models
	@echo "$(BLUE)Downloading models...$(NC)"
	python scripts/download_models.py
	@echo "$(GREEN)✓ Models downloaded$(NC)"

setup-data: ## Setup sample data for testing
	@echo "$(BLUE)Setting up sample data...$(NC)"
	python scripts/setup_sample_data.py
	@echo "$(GREEN)✓ Sample data ready$(NC)"

demo: ## Run demo example
	@echo "$(BLUE)Running demo...$(NC)"
	python examples/basic_usage.py
	@echo "$(GREEN)✓ Demo complete$(NC)"

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python scripts/benchmark.py
	@echo "$(GREEN)✓ Benchmarks complete$(NC)"

check: lint test ## Run linters and tests
	@echo "$(GREEN)✓ All checks passed$(NC)"

release-check: format-check lint test build ## Check before release
	@echo "$(GREEN)✓ Ready for release$(NC)"

# CI/CD targets
ci-install: ## Install for CI/CD
	pip install -e ".[dev]"

ci-test: ## Run tests in CI/CD
	pytest tests/ -v --cov=hallufield --cov-report=xml

ci-lint: ## Run linters in CI/CD
	flake8 hallufield/
	black --check hallufield/
	isort --check-only hallufield/

# Development helpers
watch-test: ## Watch and re-run tests on file changes
	@echo "$(BLUE)Watching for changes...$(NC)"
	pytest-watch -- tests/ -v

shell: ## Start IPython shell with package imported
	@echo "$(BLUE)Starting IPython shell...$(NC)"
	ipython -c "import hallufield; from hallufield.core import *; print('HalluField loaded')"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t hallufield:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --gpus all -v $(PWD):/workspace hallufield:latest

# GPU check
check-gpu: ## Check GPU availability
	@echo "$(BLUE)Checking GPU...$(NC)"
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
