.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
PYTHON   ?= python
PIP      ?= pip
SRC      := src
TESTS    := tests
REPORTS  := reports

.PHONY: help install install-dev lint format test test-unit test-integration \
        coverage clean docs

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:  ## Install the package (core deps only)
	$(PIP) install -e "."

install-dev:  ## Install dev + all optional deps
	$(PIP) install -e ".[all]"
	$(PIP) install -r requirements-dev.txt

# ---------------------------------------------------------------------------
# Linting & formatting
# ---------------------------------------------------------------------------
lint:  ## Run ruff linter
	ruff check $(SRC) $(TESTS)

format:  ## Run ruff formatter
	ruff format $(SRC) $(TESTS)

lint-fix:  ## Auto-fix ruff lint issues
	ruff check --fix $(SRC) $(TESTS)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:  ## Run all tests with pytest
	pytest $(TESTS)

test-unit:  ## Run only unit tests
	pytest $(TESTS)/unit

test-integration:  ## Run only integration tests
	pytest $(TESTS)/integration

coverage:  ## Run tests with coverage report
	pytest --cov=$(SRC)/hotelling --cov-report=html --cov-report=term-missing $(TESTS)

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
clean:  ## Remove build artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------
docs:  ## Build Sphinx docs (placeholder)
	@echo "Docs build not yet configured."
