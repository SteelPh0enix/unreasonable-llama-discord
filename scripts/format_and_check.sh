#!/bin/sh

set -xeo pipefail

poetry run ruff format $(ls ./unllamabot/*.py)
poetry run mypy --strict --pretty $(ls ./unllamabot/*.py)
poetry run ruff check --fix $(ls ./unllamabot/*.py)

poetry run ruff format $(ls ./tests/*.py)
poetry run mypy --strict --pretty $(ls ./tests/*.py)
poetry run ruff check --fix $(ls ./tests/*.py)
