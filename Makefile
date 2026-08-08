AI ?=
LOG := .ci-ai.log

ifdef AI
_goals := $(or $(MAKECMDGOALS),ci)
.PHONY: $(_goals)
$(_goals):
	@rm -f $(LOG)
	@$(MAKE) --no-print-directory AI= $@ > $(LOG) 2>&1 \
		&& echo "✅ $@ passed (log: $(LOG))" \
		|| (echo "❌ $@ failed:"; uv run scripts/ai_filter_log.py $(LOG); echo "(full log: $(LOG))"; exit 1)

else

.PHONY: install setup reset remove-venv reset-full upgrade-deps test lint \
	py-lint py-lint-fix lint-fix format pre-commit-check update-hooks ci ci-slow build clean \
	clean-logs clean-cache run run-verbose export-models \
	release-patch release-minor release-major _do-release help \
	local-setup dev-local check-local-editable run-local run-local-verbose \
	python-local

# ---------------------------------------------------------------------------
# Peer-repo discovery for *-local targets
# ---------------------------------------------------------------------------
# `make *-local` workflows install / run pd-ocr-trainer against a sibling
# pdomain-book-tools checkout instead of the pinned commit in pyproject.toml.
# PEER_BOOK_TOOLS is the absolute path if the sibling exists, or empty
# otherwise. The require-peer guard (used inside each *-local recipe) prints
# a clear message and exits 1 when the sibling is missing — no surprise
# failures from raw uv errors. `make local-setup` clones the sibling if
# missing.
PEER_BOOK_TOOLS_PATH := ../pdomain-book-tools
PEER_BOOK_TOOLS_REPO := https://github.com/ConcaveTrillion/pdomain-book-tools.git
PEER_BOOK_TOOLS := $(realpath $(PEER_BOOK_TOOLS_PATH))

define _require_peer_book_tools
	@if [ -z "$(PEER_BOOK_TOOLS)" ]; then \
		echo "❌ Peer repo not found at $(PEER_BOOK_TOOLS_PATH)."; \
		echo "   This *-local target requires pdomain-book-tools as a sibling checkout."; \
		echo "   Run: make local-setup"; \
		echo "   (or clone manually: git clone $(PEER_BOOK_TOOLS_REPO) $(PEER_BOOK_TOOLS_PATH))"; \
		exit 1; \
	fi
endef

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up development environment (uv sync + pre-commit hooks)
	@echo "📦 Installing dependencies..."
	uv sync --group all-dev
	@echo "🪝 Setting up pre-commit hooks..."
	@[ -f .git/hooks/pre-commit ] || uv run pre-commit install
	@echo "✅ Setup complete!"

install: ## Install pd-ocr-trainer-ui as a uv tool from local source
	uv tool install --reinstall .
	@echo "pd-ocr-trainer-ui installed. Run: pd-ocr-trainer-ui"

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
	@$(MAKE) --no-print-directory update-hooks
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

update-hooks: ## Bump pinned pre-commit hook revisions in .pre-commit-config.yaml
	@echo "⬆️  Updating pinned pre-commit hook revisions..."
	@# The hook exits non-zero when it rewrites the config, which is the success
	@# case here, so its status is not the target's status.
	-@uv run pre-commit run pre-commit-update --all-files --hook-stage manual
	@echo "✅ Hook revisions updated — review the .pre-commit-config.yaml diff."

ci: ## Run complete CI pipeline (setup, pre-commit, test, build)
	@echo "🚀 Running complete CI pipeline..."
	@$(MAKE) --no-print-directory setup
	@$(MAKE) --no-print-directory pre-commit-check
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory build
	@echo "✅ CI pipeline complete!"

ci-slow: ci ## Full pre-flight for releases (alias of ci today; reserved for slower checks if added later)

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

release-patch: ## Release: bump patch, run ci-slow, tag, push (e.g. v0.1.0 -> v0.1.1)
	@$(MAKE) --no-print-directory _do-release BUMP=patch

release-minor: ## Release: bump minor, run ci-slow, tag, push (e.g. v0.1.0 -> v0.2.0)
	@$(MAKE) --no-print-directory _do-release BUMP=minor

release-major: ## Release: bump major, run ci-slow, tag, push (e.g. v0.1.0 -> v1.0.0)
	@$(MAKE) --no-print-directory _do-release BUMP=major

_do-release:
	@BUMP=$(or $(BUMP),minor) ./scripts/do-release.sh

# ---------------------------------------------------------------------------
# Local editable workflow (requires ../pdomain-book-tools sibling checkout)
# ---------------------------------------------------------------------------
# Each target self-checks for the peer repo and exits cleanly if absent.
# UV_NO_SYNC=1 on *-run* targets prevents `uv run` from re-resolving the lock
# (which would silently overwrite the editable install with the pinned tag).

local-setup: ## [local-dev] Clone ../pdomain-book-tools if missing and set up the editable workspace
	@if [ -d "$(PEER_BOOK_TOOLS_PATH)" ]; then \
		echo "✅ Peer repo already at $(PEER_BOOK_TOOLS_PATH) — skipping clone."; \
	else \
		echo "📥 Cloning pdomain-book-tools from $(PEER_BOOK_TOOLS_REPO)..."; \
		git clone "$(PEER_BOOK_TOOLS_REPO)" "$(PEER_BOOK_TOOLS_PATH)"; \
	fi
	@$(MAKE) --no-print-directory dev-local
	@echo ""
	@echo "💡 Optional: to also set up pdomain-book-tools' own venv (for running its tests):"
	@echo "      (cd $(PEER_BOOK_TOOLS_PATH) && make setup)"

dev-local: ## [local-dev] Install pdomain-book-tools from ../pdomain-book-tools as editable in the venv
	$(call _require_peer_book_tools)
	UV_LINK_MODE=copy uv sync --group all-dev
	UV_LINK_MODE=copy uv pip install -e "$(PEER_BOOK_TOOLS)"
	@$(MAKE) --no-print-directory check-local-editable
	@echo "✅ Local editable pdomain-book-tools is active in the venv."

check-local-editable: ## [local-dev] Verify pdomain-book-tools resolves to ../pdomain-book-tools (not the pinned commit)
	$(call _require_peer_book_tools)
	@env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run python -c "import inspect, os, sys, importlib.metadata as md, pdomain_book_tools; \
module_file = os.path.realpath(inspect.getfile(pdomain_book_tools)); \
peer = os.path.realpath('$(PEER_BOOK_TOOLS)'); \
is_local = module_file.startswith(peer + os.sep) or module_file == peer; \
print('module_file=', module_file); \
print('expected_peer=', peer); \
print('dist_version=', md.version('pdomain-book-tools')); \
sys.exit(0 if is_local else 1)" || (echo "❌ pdomain-book-tools is not local/editable. Run: make dev-local" >&2; exit 1)
	@echo "✅ pdomain-book-tools resolves to local editable copy."

run-local: check-local-editable ## [local-dev] Run the trainer UI against the local editable workspace; pass ARGS="..."
	env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run pd-ocr-trainer-ui $(ARGS)

run-local-verbose: check-local-editable ## [local-dev] Run the trainer UI in verbose mode against the local editable workspace; pass ARGS="..."
	env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run pd-ocr-trainer-ui $(ARGS)

python-local: check-local-editable ## [local-dev] Run python against the local editable workspace; pass ARGS="..."
	env -u VIRTUAL_ENV UV_NO_SYNC=1 uv run python $(ARGS)

.DEFAULT_GOAL := help

endif
