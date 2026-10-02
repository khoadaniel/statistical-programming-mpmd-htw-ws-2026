# Object-oriented programming, errors and tests

This page covers the first block of Session 2. Session 1 introduced classes, objects, attributes and methods. Here we use them in practice: how classes are combined (composition and inheritance), how dataclasses remove boilerplate, how type hints and docstrings document a contract, how exceptions report broken contracts, and how automated tests with pytest check that the code keeps its promises. The running example is a class that validates and cleans the record of one Binding Tariff Information (BTI) decision of the course case study, `DecisionRecord`, in the [workspace](../workspace/README.md).

```mermaid
flowchart LR
  C["Class<br/>(contract: fields, methods)"] --> H["Type hints and<br/>docstrings<br/>(written contract)"]
  C --> E["Exceptions<br/>(contract broken)"]
  H --> T["Tests<br/>(contract checked)"]
  E --> T
  T --> R["Code others<br/>can rely on"]
```

## Object-oriented programming in practice

### Concept

A **class** bundles data (**attributes**) and behaviour (**methods**). Two ways of building larger classes from smaller ones:

- **Composition** ("has a"): an object holds other objects as attributes and delegates work to them. A `FeatureSet` *has* a list of features; the open-data client of the workspace *has* an HTTP client.
- **Inheritance** ("is a"): a **subclass** takes over all attributes and methods of its **base class** (or parent) and adds or replaces some. `DigitCount` *is a* `Feature`. Replacing a method of the parent is called **overriding**. Code that works with `Feature` objects works with every subclass: this is **polymorphism**.

A **dataclass** is a class whose main purpose is to hold data. The decorator `@dataclass` reads the annotated class attributes (the **fields**) and writes `__init__`, `__repr__` and `__eq__` automatically. The special method `__post_init__` runs right after `__init__` and is the place to check and clean the fields. `frozen=True` makes the objects read-only.

A small example by hand: `FeatureSet([WordCount(), DigitCount()])` applied to the description `"Plush toy, 30 cm"` asks each feature in turn: four words, two digits, so the result is `{"n_words": 4, "n_digits": 2}`.

```mermaid
classDiagram
  class Feature {
    <<abstract>>
    +str name
    +compute(text) int
  }
  class WordCount {
    +compute(text) int
  }
  class DigitCount {
    +compute(text) int
  }
  class FeatureSet {
    +list~Feature~ features
    +transform(text) dict
  }
  Feature <|-- WordCount : inherits
  Feature <|-- DigitCount : inherits
  FeatureSet o-- Feature : has many
```

### Why it matters

Composition keeps classes small and replaceable: the HTTP client inside the open-data client can be swapped for a fake one in tests. Inheritance lets many classes share one interface, so that the code that uses them does not need to know which one it has. Dataclasses make data-holding classes short enough to read at a glance. Used carelessly, deep inheritance trees become hard to follow; a common rule is *prefer composition, inherit only for a true "is a" relation*.

### How it works in Python

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class Feature(ABC):
    """Base class: a feature turns one text into one number."""

    name = "feature"

    @abstractmethod
    def compute(self, text: str) -> int:   # every subclass must implement this
        ...


class WordCount(Feature):                  # inheritance: WordCount is a Feature
    name = "n_words"

    def compute(self, text: str) -> int:   # overrides the abstract method
        return len(text.split())


class DigitCount(Feature):
    name = "n_digits"

    def compute(self, text: str) -> int:
        return sum(c in "0123456789" for c in text)


@dataclass
class FeatureSet:                          # composition: a FeatureSet has features
    features: list[Feature] = field(default_factory=list)

    def transform(self, text: str) -> dict[str, int]:
        return {f.name: f.compute(text) for f in self.features}   # polymorphism


features = FeatureSet([WordCount(), DigitCount()])
print(features.transform("Plush toy, 30 cm"))  # {'n_words': 4, 'n_digits': 2}
print(isinstance(WordCount(), Feature))        # True
try:
    Feature()                                  # an abstract class cannot be instantiated
except TypeError as err:
    print("TypeError:", str(err).split(" with")[0])   # TypeError: Can't instantiate abstract class Feature


@dataclass
class Decision:
    """A dataclass: __init__, __repr__ and __eq__ are generated from the fields."""

    bti_reference: str
    heading: str
    description: str = ""

    def __post_init__(self) -> None:           # runs after the generated __init__
        self.description = " ".join(self.description.split())


d = Decision("DE1", "9503", "  Plush   toy ")
print(d)                                       # Decision(bti_reference='DE1', heading='9503', description='Plush toy')
print(d == Decision("DE1", "9503", "Plush toy"))   # True: fields are compared
```

### In practice

- scikit-learn builds every model on the base class `BaseEstimator` and small "mixin" classes (`ClassifierMixin`, `TransformerMixin`). Because all models share `fit` and `predict`, a `Pipeline` can combine any of them: composition and inheritance together.
- Python's `dataclasses` module (PEP 557, Python 3.7) is widely used for configuration and record types; the library pydantic, used by FastAPI, extends the same idea with automatic type conversion and validation.

> [!WARNING]
> A mutable default such as `features: list = []` in a dataclass raises `ValueError` (in a normal function default it would be silently shared between all calls). Use `field(default_factory=list)`.

> [!TIP]
> Before you write a subclass, ask whether "B is an A" is true in the domain. A `DecisionTable` *has* decisions; it is not a kind of decision. When in doubt, use composition.

The workbooks [01-classes-and-objects.ipynb](../workbooks/01-classes-and-objects.ipynb) (composition, equivalence, copying) and [02-inheritance.ipynb](../workbooks/02-inheritance.ipynb) (*Think Python*) practise both patterns with exercises.

## Type hints and docstrings

### Concept

A function or class is a **contract**: given inputs of a stated kind, it returns an output with a stated meaning. The contract is written down in two places.

- **Type hints** state the types of parameters, return values and fields: `def n_words(text: str) -> int`. Common forms: `list[str]`, `dict[str, int]`, `tuple[int, int]`, `str | None` (text or nothing), `Callable[[float], None]`. Python does **not** check them when the code runs. Editors use them for completion and warnings; **type checkers** such as mypy or ty check them without running the code.
- A **docstring** is the string literal on the first line of a module, class or function. It explains what the code does, its arguments, its result, the exceptions it raises and an example. `help(function)` and editors display it; documentation generators build web pages from it. The workspace uses the *Google style* (`Args:`, `Returns:`, `Raises:`).

### Why it matters

Hints and docstrings tell a teammate (and an AI assistant) how to call the code without reading its body. Type checkers catch a class of bugs before any test runs, for example passing a whole DataFrame where one text is expected. In code review, a missing or wrong docstring is as much a defect as a wrong line.

### How it works in Python

```python
def share_with(texts: list[str], char: str = "%") -> float | None:
    """Share of texts that contain ``char``.

    Args:
        texts: descriptions of goods.
        char: the character to look for.

    Returns:
        A share between 0 and 1, or None if ``texts`` is empty.
    """
    if not texts:
        return None
    return sum(char in t for t in texts) / len(texts)


print(share_with(["100 % cotton", "Plush toy"]))   # 0.5
print(share_with([]))                          # None
print(share_with.__annotations__["return"])    # float | None
print(share_with.__doc__.splitlines()[0])      # Share of texts that contain ``char``.
try:
    share_with(["100 % cotton"], char=5)       # the hint says str; Python does not stop the call
except TypeError as err:
    print("TypeError:", err)                   # TypeError: 'in <string>' requires string as left operand, not int
```

Checking the hints statically, without running the code:

```bash
uvx ty check src/          # or: uvx mypy src/
```

### In practice

- pandas, NumPy and scikit-learn document every public function in the *NumPy docstring style*; their online API reference is generated from these docstrings.
- Dropbox added type hints to about four million lines of Python and checks them with mypy; the company described the migration in "Our journey to type checking 4 million lines of Python" (2019).

> [!NOTE]
> `from __future__ import annotations` at the top of a module (used in the workspace) stores hints as text and allows the newer syntax such as `str | None` on older Python versions.

> [!CAUTION]
> A hint is a promise, not a check. `heading: str` does not stop someone from passing the integer `901`, which has already lost the leading zero of `"0901"`. Values that come from outside (files, forms, APIs) must be checked explicitly, which is the job of the validation in `__post_init__`.

## Exceptions and error handling

### Concept

An **exception** is Python's signal that an operation cannot be completed. Built-in examples: `ValueError` (right type, wrong value), `TypeError` (wrong type), `KeyError` (missing key), `FileNotFoundError`. If nothing handles an exception, the program stops and prints a **traceback**: the chain of calls that led to the error. Read it from the bottom: the last line names the exception and its message.

- `raise ValueError("message")` signals a broken contract deliberately.
- `try: … except ValueError as err: …` **handles** an expected exception; `else:` runs if no exception occurred; `finally:` always runs (for example to close a file).
- Exceptions are classes and form a hierarchy. Defining your own subclasses, such as `ValidationError(BtiToolsError, ValueError)`, lets callers catch exactly what they can handle.
- `raise NewError(...) from err` keeps the original cause in the traceback.

Two styles: *Look before you leap* (LBYL) checks first (`if key in d:`); *Easier to ask forgiveness than permission* (EAFP) tries and handles the exception. Python code often uses EAFP for conversions such as `int(value)`.

```mermaid
flowchart TD
  I["Raw value"] --> V{"Valid?"}
  V -- yes --> C["Clean and store"]
  V -- no --> R["raise ValidationError(field, message)"]
  R --> H{"Caller can<br/>handle it?"}
  H -- "yes (skip record,<br/>answer HTTP 422)" --> L["Handle and log"]
  H -- no --> S["Stop with traceback"]
```

### Why it matters

A clear error at the place where the problem occurs saves hours compared with a wrong number found weeks later in a chart. Silent failures are worse than crashes because they produce plausible results. Catching every exception with a bare `except:` hides programming errors.

### How it works in Python

```python
from datetime import date, datetime


class BtiToolsError(Exception):
    """Base class of the package's errors."""


class ValidationError(BtiToolsError, ValueError):
    """A record breaks a rule; also a ValueError for callers that expect one."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


def parse_start_date(value: object) -> date:
    """Parse a date in the format of the EBTI export, day first: 10/05/2023."""
    try:
        parsed = datetime.strptime(str(value).strip(), "%d/%m/%Y").date()   # EAFP
    except ValueError as err:
        raise ValidationError("start_date", f"expected dd/mm/yyyy, got {value!r}") from err
    if not 1990 <= parsed.year <= 2035:
        raise ValidationError("start_date", f"implausible year {parsed.year}")
    return parsed


good, bad = [], []
for raw in ["10/05/2023", "2023-05-10", "31/02/2023", "15/06/2200", "01/12/2022"]:
    try:
        start = parse_start_date(raw)
    except ValidationError as err:               # handle only the error we expect
        bad.append((err.field, str(err)))
    else:
        good.append(start.isoformat())

print(good)        # ['2023-05-10', '2022-12-01']
print(bad[0])      # ('start_date', "start_date: expected dd/mm/yyyy, got '2023-05-10'")
print(bad[2])      # ('start_date', 'start_date: implausible year 2200')
```

### In practice

- In 2020, Public Health England failed to report about 16,000 positive COVID-19 test results because a data pipeline stored them in the old Excel format (`.xls`), whose limit of 65,536 rows silently cut the data. A pipeline that raised an error on truncation would have stopped instead of under-reporting.
- scikit-learn raises `ValueError: Input X contains NaN` instead of fitting a model on missing values; the message names the problem and the fix.

> [!WARNING]
> Never write `except: pass` or `except Exception: pass`. It hides typing mistakes (`NameError`) together with the error you meant to catch. Catch specific exceptions and do something meaningful: skip and count the record, log it, or re-raise with a clearer message.

## Automated tests with pytest

### Concept

A **unit test** is a small function that calls the code with a known input and asserts the expected output. **pytest** finds files named `test_*.py`, runs every function whose name starts with `test_`, and reports each as passed, failed or skipped.

- `assert expression` fails the test if the expression is false; pytest shows the values involved.
- `@pytest.mark.parametrize("x, expected", [...])` runs one test for many cases, each reported separately.
- A **fixture** (`@pytest.fixture`) prepares data for tests; a test receives it by naming it as a parameter. Fixtures in `conftest.py` are available to all test files. `tmp_path` is a built-in fixture giving a fresh temporary folder.
- `with pytest.raises(ValidationError):` checks that the code raises the expected exception.
- `@pytest.mark.skip` skips a test (the workspace uses it for exercises).

Good tests are **small** (one behaviour each), **independent** (no shared state, any order), **fast** and **deterministic** (no network, no randomness without a seed). They cover normal cases, **edge cases** (empty text, the first and the last chapter, a heading with a leading zero) and invalid input.

```mermaid
stateDiagram-v2
  [*] --> Red: write a test that fails
  Red --> Green: write the simplest code that passes
  Green --> Refactor: clean up, tests stay green
  Refactor --> Red: next behaviour
```

This cycle is called **test-driven development** (TDD). You do not have to follow it strictly, but writing the test first makes you decide what "correct" means before you write the code.

### Why it matters

Tests turn "it worked when I tried it" into a check that runs on every change, on every computer and, with continuous integration (block 3), on every pull request. They make it safe to change shared code, and they document the expected behaviour by example.

### How it works in Python

Save as `test_chapter.py` and run `uv run --with pytest pytest -q test_chapter.py`, or run the file directly with Python (the last two lines start pytest):

```python
import pytest


def chapter_of(heading: str) -> str:
    if not (isinstance(heading, str) and len(heading) == 4 and heading.isdigit()):
        raise ValueError(f"expected a four-digit heading, got {heading!r}")
    return heading[:2]


@pytest.fixture
def headings() -> list[str]:
    return ["9503", "3926", "0901", "6307", "8517"]


def test_chapters_of_fixture(headings):
    assert [chapter_of(h) for h in headings] == ["95", "39", "09", "63", "85"]


@pytest.mark.parametrize(("heading", "expected"), [("0101", "01"), ("0901", "09"), ("9706", "97")])
def test_boundaries(heading, expected):
    assert chapter_of(heading) == expected


@pytest.mark.parametrize("heading", ["950", "95031", "toys", 9503])
def test_malformed_heading_raises(heading):
    with pytest.raises(ValueError, match="four-digit"):
        chapter_of(heading)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))   # 8 passed
```

In the workspace, tests live in `tests/` and import the package: `from btitools import DecisionRecord`. Run them with `uv run pytest -q` in the workspace folder.

### In practice

- NumPy, pandas and scikit-learn use pytest; their test suites run in continuous integration on every pull request before a maintainer merges it.
- In 2013, a reproduction by Herndon, Ash and Pollin found a spreadsheet formula error in an influential economics paper (Reinhart and Rogoff 2010) that had excluded several countries from an average. Automated checks of the computation against hand-computed cases are designed to catch exactly this kind of error.

> [!CAUTION]
> A test that never fails is worthless. After writing a test, break the code on purpose (or change the expected value) and check that the test fails.

> [!TIP]
> When you fix a bug, first write a test that reproduces it. The test fails, you fix the code, the test passes, and the bug can never come back unnoticed (a **regression test**).

## Practice: a class that validates and cleans a decision record

Work in the [workspace](../workspace/README.md), exercises 1 and 2:

1. Read `src/btitools/records.py`: which fields does `DecisionRecord` have, which rules does `__post_init__` apply (reference present, two-letter country code, EU language code, plausible start date, non-empty description, four-digit heading), and what does `from_dict` add?
2. Run `uv run pytest -q tests/test_records.py` and read one parametrized test.
3. Complete the TODO: `end_date` is optional, but if present it must be a valid date that does not lie before `start_date`. Activate `test_end_date_rules`.
4. Add two tests of your own for rules that are not yet tested.

```python
import sys

sys.path.insert(0, "sessions/02-software-engineering/workspace/src")   # uv run in the workspace does this
from btitools import DecisionRecord, ValidationError

raw = {"bti_reference": " DE42 ", "issuing_country": "de", "language": "DE",
       "start_date": "10/05/2023", "description": "Plüschtier,  30 cm ", "heading": "9503"}
record = DecisionRecord.from_dict(raw)
print(record.bti_reference, record.issuing_country, record.language, record.start_date)
# DE42 DE de 2023-05-10
print(record.description, "|", record.chapter)   # Plüschtier, 30 cm | 95
try:
    DecisionRecord.from_dict({**raw, "heading": 901})
except ValidationError as err:
    print(err.field, "|", err)   # heading | heading: expected text, got int
```

## Check your understanding

1. The open-data client holds an `httpx.Client` as an attribute. Is this composition or inheritance, and why does it make testing easier?
2. What does `@dataclass` generate, and what is `__post_init__` for?
3. A function has the hint `heading: str`. What happens when it is called with the integer `901`?
4. Why does `ValidationError` inherit from both `BtiToolsError` and `ValueError`?
5. Name three kinds of cases a good set of tests for `parse_start_date` should contain.

## Further reading

- Downey, A. B. (2024). *Think Python*, 3rd edition, chapters 16–17 (classes and objects, inheritance). Free online, CC BY-NC-SA 4.0. https://allendowney.github.io/ThinkPython/
- The Carpentries Incubator (2024). *Intermediate Research Software Development in Python*, section 2 (testing) and section 3 (software design). CC-BY 4.0. https://carpentries-incubator.github.io/python-intermediate-development/
- Python Software Foundation (2026). *dataclasses — Data Classes*. https://docs.python.org/3/library/dataclasses.html
- pytest developers (2026). *Get started*; *How to parametrize fixtures and test functions*. https://docs.pytest.org/en/stable/getting-started.html
