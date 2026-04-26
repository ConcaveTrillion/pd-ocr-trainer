.PHONY: install setup reset remove-venv reset-full upgrade-deps test lint \
	py-lint py-lint-fix lint-fix format pre-commit-check ci build clean \
	clean-logs clean-cache run run-verbose export-models help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up development environment
	@echo "📦 Installing dependencies..."
	uv sync --group all-dev
	@echo "🪝 Setting up pre-commit hooks..."
	uv run pre-commit install
	@echo "✅ Installation complete!"

setup: install ## Alias for install

remove-venv: ## Remove the virtual environment
	@echo "🗑️  Removing existing virtual environment..."
	rm -rf .venv
	@echo "✅ Virtual environment removed!"

reset: ## Rebuild virtual environment (keeps UV cache)
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory remove-venv
	@$(MAKE) --no-print-directory install
	@echo "✅ Environment Reset!"

reset-full: ## Nuclear option: clear everything and redownload
	@echo "🔄 FULL RESET: Clearing all caches and virtual environment..."
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory remove-venv
	@echo "🧹 Clearing UV cache..."
	uv cache clean
	@echo "⬇️ Dependencies should download fresh now."
	@$(MAKE) --no-print-directory install
	@echo "💥 Full reset complete! Everything is fresh!"

upgrade-deps: ## Upgrade dependencies and sync local environment
	@echo "⬆️ Upgrading dependency lockfile..."
	uv lock --upgrade
	@echo "📦 Syncing upgraded dependencies..."
	uv sync --group all-dev
	@echo "✅ Dependencies upgraded and environment synced!"

test: ## Run tests with parallelization
	@echo "🧪 Running tests (parallelized)..."
	uv run pytest -n auto -v -ra

test-verbose: ## Run tests with verbose output and parallelization
	@echo "🧪 Running tests (verbose mode, parallelized)..."
	uv run pytest -n auto -v -ra

test-single: ## Run one pytest node id (usage: make test-single TEST='tests/...::test_name')
	@if [ -z "$(TEST)" ]; then \
		echo "❌ Missing TEST parameter."; \
		echo "   Example: make test-single TEST='tests/test_training.py::test_training_setup'"; \
		exit 1; \
	fi
	@echo "🧪 Running single test: $(TEST)"
	uv run pytest -n auto "$(TEST)"

test-k: ## Run tests by pytest -k expression (usage: make test-k K='pattern')
	@if [ -z "$(K)" ]; then \
		echo "❌ Missing K parameter."; \
		echo "   Example: make test-k K='test_config'"; \
		exit 1; \
	fi
	@echo "🧪 Running tests with -k: $(K)"
	uv run pytest -n auto -k "$(K)"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@$(MAKE) --no-print-directory py-lint

py-lint: ## Run Python linting checks
	@echo "🐍 Running Python linting checks..."
	uv run pre-commit run ruff-check --all-files

lint-fix: ## Auto-fix lint issues
	@echo "🛠️  Auto-fixing lint issues..."
	@$(MAKE) --no-print-directory py-lint-fix

py-lint-fix: ## Auto-fix Python lint issues
	@echo "🐍 Auto-fixing Python lint issues..."
	uv run pre-commit run ruff-format --all-files
	uv run pre-commit run ruff-check --all-files

format: ## Format code
	@echo "✨ Formatting code..."
	uv run ruff format
	@$(MAKE) --no-print-directory lint

pre-commit-check: ## Run pre-commit on all files
	@echo "🪝 Running pre-commit on all files..."
	uv run pre-commit run --all-files

ci: ## Run complete CI pipeline (install, pre-commit, test, build)
	@echo "🚀 Running complete CI pipeline..."
	@$(MAKE) --no-print-directory install
	@$(MAKE) --no-print-directory pre-commit-check
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory build
	@echo "✅ CI pipeline complete!"

build: ## Build distribution packages (wheel and sdist)
	@echo "📦 Building distribution packages..."
	uv build
	@echo "✅ Build complete! Check dist/ directory for packages."

run: ## Run the training UI
	@echo "🚀 Starting OCR Training UI..."
	uv run pd-ocr-trainer-ui

run-verbose: ## Run the training UI with verbose output
	@echo "🚀 Starting OCR Training UI (verbose mode)..."
	uv run pd-ocr-trainer-ui

export-models: ## Copy trained model artifacts into the workspace exports directory
	@echo "📦 Exporting trained models to the workspace..."
	../scripts/export-models.sh

clean: ## Clean up cache, temporary files, and logs (keeps venv and UV cache)
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaning coverage files..."
	rm -rf htmlcov/ 2>/dev/null || true
	rm -f coverage.xml 2>/dev/null || true
	@echo "🧹 Cleaning build artifacts..."
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	@$(MAKE) --no-print-directory clean-logs
	@$(MAKE) --no-print-directory clean-cache
	@echo "✅ Cache cleanup complete!"

clean-logs: ## Remove session logs
	@echo "🧹 Cleaning session logs..."
	rm -rf .log* 2>/dev/null || true
	@echo "✅ Log cleanup complete!"

clean-cache: ## Remove cache and temporary files
	@echo "🧹 Clearing cache..."
	rm -rf .cache/ 2>/dev/null || true
	@echo "✅ Cache cleared!"

.DEFAULT_GOAL := help
