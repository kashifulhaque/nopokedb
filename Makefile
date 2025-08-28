# Makefile (portable macOS/Linux)

# ---- Config ----
PKG          := nopokedb
INIT_FILE    := src/$(PKG)/__init__.py

# Detect OS for sed -i portability
UNAME_S      := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SED_INPLACE := sed -i ''
else
  SED_INPLACE := sed -i
endif

.PHONY: bump
# Usage: make bump V=0.5.1
bump:
	@[ -n "$(V)" ] || (echo "Usage: make bump V=<new-version>"; exit 1)
	@echo "Bumping version to $(V)"
	@$(SED_INPLACE) -E 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	@$(SED_INPLACE) -E 's/^__version__ = ".*"/__version__ = "$(V)"/' $(INIT_FILE)
	@git add pyproject.toml $(INIT_FILE)
	@git commit -m "Bump to v$(V)" || echo "nothing to commit"
	@git tag -f v$(V)
	@git push origin main --tags

.PHONY: publish
publish:
	@python -m build
	@twine upload \
	  -u __token__ \
	  -p "$(PYPI_API_KEY)" \
	  dist/*

.PHONY: test
test:
	@pytest -q
