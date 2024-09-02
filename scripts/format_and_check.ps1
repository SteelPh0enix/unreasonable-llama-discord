$python_files = Get-ChildItem -Path ./unllamabot/*.py
echo "Checked files: ${python_files}"
poetry run ruff format $python_files
poetry run mypy --strict --pretty $python_files
poetry run ruff check --fix $python_files

$python_files = Get-ChildItem -Path ./tests/*.py
echo "Checked files: ${python_files}"
poetry run ruff format $python_files
poetry run mypy --implicit-reexport --strict --pretty $python_files
poetry run ruff check --fix $python_files
