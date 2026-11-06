"""An import without the package name, kept here to show its error. Do not copy this style.

    uv run python src/gradebook/mistake_bare.py      # works by accident: the file's own folder is on sys.path
    uv run python -m gradebook.mistake_bare          # ModuleNotFoundError: No module named 'models'
"""

from models import Student  # bare import: "models" without "gradebook." in front

print("Imported", Student.__name__, "with a bare import")
