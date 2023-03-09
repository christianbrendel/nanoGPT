.PHONY: all format-code

all: format-code

format-code:
	pre-commit install
	isort .
	black .