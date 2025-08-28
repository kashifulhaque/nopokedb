# =========================
# NoPokeDB Makefile (portable macOS/Linux)
# =========================

# ---- Project config ----
PKG       := nopokedb
INIT_FILE := src/$(PKG)/__init__.py

# ---- sed -i portability (BSD/mac vs GNU) ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SED_INPLACE := sed -i ''
else
  SED_INPLACE := sed -i
endif

# ---- Tooling autodetect (uv/uvx/python) ----
HAS_UV  := $(shell command -v uv  >/dev/null 2>&1 && echo 1 || echo 0)
HAS_UVX := $(shell command -v uvx >/dev/null 2>&1 && echo 1 || echo 0)
PYEXE   := $(shell command -v python >/dev/null 2>&1 && echo python || (command -v python3 >/dev/null 2>&1 && echo python3 || echo python))

RUNPY    := $(if $(filter 1,$(HAS_UV)),uv run $(PYEXE),$(PYEXE))
PYTEST   := $(if $(filter 1,$(HAS_UV)),uv run pytest,pytest)
RUFF     := $(if $(filter 1,$(HAS_UV)),uv run ruff,ruff)
BUILDCMD := $(if $(filter 1,$(HAS_UVX)),uvx --from build pyproject-build,$(RUNPY) -m build)
TWINECMD := $(if $(filter 1,$(HAS_UVX)),uvx twine,$(if $(filter 1,$(HAS_UV)),uv run twine,twine))

# ---- Allow both styles: `make bump V=1.2.3` OR `make bump 1.2.3` ----
ARGV := $(word 2,$(MAKECMDGOALS))
ifeq ($(V),)
  ifneq ($(ARGV),)
    V := $(ARGV)
  endif
endif

# =========================
# Targets
# =========================
.PHONY: help
help: ## Show this help
	@echo "NoPokeDB – common tasks"
	@echo
	@echo "  make bump V=0.5.1        Bump version in pyproject + __init__, commit & tag"
	@echo "  make bump 0.5.1          (same as above)"
	@echo "  make publish             Build, check, and upload to PyPI (needs PYPI_API_KEY)"
	@echo "  make publish-test        Build, check, upload to TestPyPI (needs TEST_PYPI_API_KEY)"
	@echo "  make test                Run pytest"
	@echo "  make lint                Run ruff (if installed)"
	@echo "  make clean               Remove build artifacts"
	@echo "  make distclean           Clean + remove virtual artifacts"
	@echo

.PHONY: bump
bump: ## Bump version: make bump V=x.y.z OR make bump x.y.z
	@[ -n "$(V)" ] || (echo "Usage: make bump V=<new-version>  (or: make bump <new-version>)"; exit 1)
	@echo "Bumping version to $(V)"
	@$(SED_INPLACE) -E 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	@$(SED_INPLACE) -E 's/^__version__ = ".*"/__version__ = "$(V)"/' $(INIT_FILE)
	@git add pyproject.toml $(INIT_FILE)
	@git commit -m "Bump to v$(V)" || echo "nothing to commit"
	@git tag -f v$(V)
	@git push origin main --tags

.PHONY: publish
publish: ensure-tools clean ## Build wheels/sdist, check, and upload to PyPI
	@echo "Building sdist and wheel..."
	@$(BUILDCMD)
	@echo "Validating metadata..."
	@$(TWINECMD) check dist/*
	@echo "Uploading to PyPI..."
	@[ -n "$(PYPI_API_KEY)" ] || (echo "Set PYPI_API_KEY to your PyPI token"; exit 1)
	@$(TWINECMD) upload -u __token__ -p "$(PYPI_API_KEY)" dist/*

.PHONY: publish-test
publish-test: ensure-tools clean ## Build & upload to TestPyPI
	@echo "Building sdist and wheel..."
	@$(BUILDCMD)
	@echo "Validating metadata..."
	@$(TWINECMD) check dist/*
	@echo "Uploading to TestPyPI..."
	@[ -n "$(TEST_PYPI_API_KEY)" ] || (echo "Set TEST_PYPI_API_KEY to your TestPyPI token"; exit 1)
	@$(TWINECMD) upload -r testpypi -u __token__ -p "$(TEST_PYPI_API_KEY)" dist/*

.PHONY: test
test: ## Run pytest
	@$(PYTEST) -q

.PHONY: lint
lint: ## Run ruff if available
	@$(RUFF) check . || echo "ruff not found; install with 'uv tool install ruff' or 'pip install ruff'"

.PHONY: clean
clean: ## Remove build artifacts
	@rm -rf dist build *.egg-info

.PHONY: distclean
distclean: clean ## Deep clean (extend as needed)
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +

.PHONY: ensure-tools
ensure-tools: ## Ensure build/twine exist if not using uvx
ifeq ($(HAS_UVX),0)
ifeq ($(HAS_UV),0)
	@$(PYEXE) -m pip show build  >/dev/null 2>&1 || $(PYEXE) -m pip install --upgrade build
	@$(PYEXE) -m pip show twine  >/dev/null 2>&1 || $(PYEXE) -m pip install --upgrade twine
endif
endif

# Swallow the second word when using `make bump 0.5.1`
%:
	@:
