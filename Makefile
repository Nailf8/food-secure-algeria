########################
# Editable configuration
########################

# General Variables
REQUIRED_PYTHON_VERSION          := 3.10
# Specifies the Python command based on OS and installation ("py", "python", "python3").
PYTHON_EXECUTABLE_CMD            := python

# Project structure configuration
WORKDIR                          := .
SRC                              := $(WORKDIR)/src

# Configuration for Conda virtual environment
CONDA_VIRTUAL_ENVIRONMENT_NAME   := FoodSecureAlgeria

# Configuration for Venv virtual environment
VENV_DIR                         := $(WORKDIR)/.venv

# Detect operating system and set configuration accordingly
ifeq ($(findstring MINGW64,'$(shell uname)'),MINGW64)
	PIP                    := $(VENV_DIR)/Scripts/pip
	ACTIVATE_CMD           := source $(VENV_DIR)/Scripts/activate
	DETECTED_OS            := MINGW64
else ifeq ('$(OS)','Windows_NT')
	PIP                    := $(VENV_DIR)/Scripts/pip
	ACTIVATE_CMD           := call $(VENV_DIR)/Scripts/activate.bat
	DETECTED_OS            := Windows
else
	PIP                    := $(VENV_DIR)/bin/pip
	ACTIVATE_CMD           := source $(VENV_DIR)/bin/activate
	DETECTED_OS            := UNIX
endif

# Detect current python version
SHELL                      := /bin/bash
PYTHON_VERSION             := $(shell $(PYTHON_EXECUTABLE_CMD) -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")


initconda:
	@echo "Creating a new virtual environment from scratch..."
    # (Re)create an empty virtual environment with a specific Python version
	conda create --name $(CONDA_VIRTUAL_ENVIRONMENT_NAME) python=$(REQUIRED_PYTHON_VERSION) --yes
    # Install dependencies
	conda run -n $(CONDA_VIRTUAL_ENVIRONMENT_NAME) \
		pip install -r $(WORKDIR)/requirements.txt
	@echo "##################################"
	@echo "#"
	@echo "# To activate this environment, use"
	@echo "#"
	@echo "#     $$ conda activate $(CONDA_VIRTUAL_ENVIRONMENT_NAME)"
	@echo "#"
	@echo "##################################"

initvenv: check_python_version
	@echo "Creating a new virtual environment from scratch..."
    # Create a virtual environment in a project folder
	$(PYTHON_EXECUTABLE_CMD) -m venv $(VENV_DIR)
    # Install project dependencies
	$(ACTIVATE_CMD) && \
		pip install -r $(WORKDIR)/requirements.txt
	@echo "##################################"
	@echo "#"
	@echo "# To activate this environment in $(DETECTED_OS), use"
	@echo "#"
	@echo "#     $$ $(ACTIVATE_CMD)"
	@echo "#"
	@echo "##################################"

run-app:
	python ${SRC}/app.py --model openai