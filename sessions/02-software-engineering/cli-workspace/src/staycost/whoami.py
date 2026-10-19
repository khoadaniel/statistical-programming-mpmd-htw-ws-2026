"""A module that only reports its own name: run it and import it to see `__name__` change.

    uv run python -m staycost.whoami        # started directly -> __name__ is '__main__'
    uv run python -c "import staycost.whoami"  # imported        -> __name__ is 'staycost.whoami'
"""

print(f"whoami.py is running, and its __name__ is {__name__!r}")

if __name__ == "__main__":
    print("-> I was started directly, so the code under the guard runs.")
