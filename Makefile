#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = proyecto-desarrollo-soluciones
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Run pre-commit checks on all files
.PHONY: check
check:
	pre-commit run --all-files --show-diff-on-failure

## Install pre-commit git hook locally
.PHONY: precommit-install
precommit-install:
	pre-commit install

## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check .
	isort --check --diff .
	black --check .
	mypy animal_classification

## Format source code with black
.PHONY: format
format:
	isort animal_classification
	black animal_classification

## Alias for format target
.PHONY: fmt
fmt: format




## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



## Run tests
.PHONY: test
test:
	pytest --cov=animal_classification

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
