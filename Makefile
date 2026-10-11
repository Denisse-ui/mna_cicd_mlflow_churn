#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mna_cicd_mlflow_churn
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 churn
	isort --check --diff churn
	black --check churn


## Format source code with black
.PHONY: format
format:
	isort churn
	black churn


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

TRAIN = $(PYTHON_INTERPRETER) -m churn.modeling.train

## Una corrida con los valores por defecto (baseline)
.PHONY: train
train:
	$(TRAIN)


## Las 8 configuraciones del experimento
.PHONY: experiments
experiments:
	$(TRAIN) --model logreg --C 0.01
	$(TRAIN) --model logreg --C 1.0
	$(TRAIN) --model logreg --C 10
	$(TRAIN) --model logreg --penalty l1
	$(TRAIN) --model logreg --class-weight none
	$(TRAIN) --model rf
	$(TRAIN) --model rf --max-depth 10
	$(TRAIN) --model rf --min-samples-leaf 5


## Abre la interfaz de MLflow en http://127.0.0.1:5001
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001


## Borra las corridas locales de MLflow (no toca el código)
.PHONY: clean-mlflow
clean-mlflow:
	rm -rf mlflow.db mlartifacts


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