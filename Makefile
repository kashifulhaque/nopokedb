# Makefile

# read version from pyproject.toml
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.PHONY: bump
bump:
	@[ -n "$(V)" ] || (echo "Use V=0.1.2"; exit 1)
	sed -i 's/^version = ".*"/version = "$(V)"/' pyproject.toml
	sed -i 's/^__version__ = ".*"/__version__ = "$(V)"/' src/nopokedb/__init__.py
	git add .
	git commit -m "Bump to v$(V)"
	git tag v$(V)
	git push origin main --tags

.PHONY: publish
publish:
	python -m build
	twine upload \
	  -u __token__ \
	  -p $(PYPI_API_KEY) \
	  dist/*
