SHELL = /bin/bash
PYTHON ?= python3.8
PYTEST ?= pytest
VENV ?= $(shell poetry env info -p)/lib/${PYTHON}/site-packages
LC_ALL = en_DK.UTF-8
LC_CTYPE = en_DK.UTF-8

default: install

install:
	poetry install

test:
	pytest --cov=squigglypy
