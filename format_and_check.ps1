$python_files = Get-ChildItem -Path ./*.py
echo "Checked files: ${python_files}"
poetry run ruff format $python_files
poetry run mypy --strict --pretty $python_files
poetry run ruff check --fix $python_files
