"""A relative import, kept here to show its error. Do not copy this style.

    uv run python -m gradebook.mistake_relative      # works: Python knows the parent package
    uv run python src/gradebook/mistake_relative.py  # ImportError: no known parent package
"""

from .models import Student  # relative import: the dot means "the package this file is in"

print("Imported", Student.__name__, "with a relative import")
