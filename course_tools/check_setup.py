"""Check that a student's laptop is ready for the course.

Run it from the repository folder:

    uv run check_setup

It checks the tools (Git, uv), the Python environment, the course packages and the VS Code
settings, and prints one line per check with a fix for every problem. It changes nothing on the
computer. Exit code 0 means ready; 1 means at least one required check failed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = "GETTING_STARTED.md"

# (import name, distribution name, why the course needs it)
PACKAGES = [
    ("numpy", "numpy", "arrays, Session 1"),
    ("pandas", "pandas", "tables, Session 1"),
    ("pyarrow", "pyarrow", "reading the Parquet data"),
    ("matplotlib", "matplotlib", "charts"),
    ("seaborn", "seaborn", "statistical charts, Session 5"),
    ("scipy", "scipy", "statistical tests, Session 5"),
    ("statsmodels", "statsmodels", "regression, Session 6"),
    ("sklearn", "scikit-learn", "machine learning, Sessions 6 to 13"),
    ("xgboost", "xgboost", "gradient boosting, Session 10"),
    ("lightgbm", "lightgbm", "gradient boosting, Sessions 10 and 12"),
    ("catboost", "catboost", "gradient boosting, Session 10"),
    ("torch", "torch", "language models, Sessions 14 and 15"),
    ("sentence_transformers", "sentence-transformers", "embeddings, Session 14"),
    ("fastapi", "fastapi", "web APIs, Session 16"),
    ("ipykernel", "ipykernel", "running notebooks in VS Code"),
]
OPENMP_PACKAGES = {"xgboost", "lightgbm"}

OK, WARN, FAIL = "ok", "warn", "fail"
MARK = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}
results: list[str] = []


def report(status: str, what: str, detail: str = "", fix: str = "") -> None:
    results.append(status)
    line = f"{MARK[status]} {what}"
    if detail:
        line += f": {detail}"
    print(line)
    if fix and status != OK:
        print(f"     → {fix}")


def run(command: list[str]) -> str | None:
    """Return the first line of a command's output, or None if it cannot run."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    except OSError, subprocess.SubprocessError:
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else ""


def check_computer() -> None:
    system, machine = platform.system(), platform.machine().lower()
    name = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(system, system)
    if system == "Darwin" and machine == "x86_64":
        report(
            FAIL,
            "Computer",
            "Intel Mac",
            "Intel Macs are not supported; ask the lecturer for help.",
        )
    elif system == "Windows" and machine in ("arm64", "aarch64"):
        report(
            FAIL, "Computer", "Windows on ARM", "Windows on ARM is not supported; ask the lecturer."
        )
    else:
        report(OK, "Computer", f"{name} on {machine}")


def check_tools() -> None:
    git = run(["git", "--version"])
    if git is None:
        report(
            FAIL, "Git", "not found", f"Install Git ({GUIDE}, step 2), then open a new terminal."
        )
    else:
        report(OK, "Git", git.replace("git version ", ""))
        name = run(["git", "config", "--global", "user.name"])
        email = run(["git", "config", "--global", "user.email"])
        if name and email:
            report(OK, "Git identity", f"{name} <{email}>")
        else:
            report(
                WARN,
                "Git identity",
                "name or e-mail not set (needed from Session 2)",
                f"Run the two git config commands of {GUIDE}, step 2 (with your name and the "
                "e-mail of your GitHub account).",
            )

    uv = run(["uv", "--version"])
    if uv is None:
        report(FAIL, "uv", "not found", f"Install uv ({GUIDE}, step 3), then open a new terminal.")
    else:
        report(OK, "uv", uv.replace("uv ", ""))


def check_python() -> None:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] == (3, 14):
        report(OK, "Python", version)
    else:
        report(
            FAIL,
            "Python",
            f"{version}, the course needs 3.14",
            "Run uv sync in the repository folder.",
        )

    venv = REPO / ".venv"
    inside = Path(sys.prefix).resolve() == venv.resolve()
    if inside:
        report(OK, "Environment", ".venv of the course repository")
    else:
        report(
            FAIL,
            "Environment",
            f"running in {sys.prefix}",
            "Run the check with uv run check_setup from the repository folder.",
        )


def check_packages() -> None:
    for module, dist, why in PACKAGES:
        try:
            importlib.import_module(module)
        # Any error counts: ImportError, an OSError for a missing system library such as OpenMP, or a
        # broken package. The check must report it and go on, never stop with a traceback.
        except Exception as error:  # noqa: BLE001
            text = f"{type(error).__name__}: {error}"
            if module in OPENMP_PACKAGES and ("omp" in text.lower() or "vcomp" in text.lower()):
                report(
                    WARN,
                    f"Package {dist}",
                    "the OpenMP runtime is missing (needed from Session 10)",
                    "Install it before Session 10: see the Setup section of "
                    "sessions/10-tree-based-models/README.md.",
                )
            else:
                report(
                    FAIL,
                    f"Package {dist}",
                    f"cannot be imported ({why}): {text.splitlines()[0][:120]}",
                    "Run uv sync. If the error stays, post this output in the course forum.",
                )
            continue
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:  # e.g. xgboost-cpu, the Linux build
            version = getattr(sys.modules.get(module), "__version__", "installed")
        report(OK, f"Package {dist}", version)


def check_vscode() -> None:
    if (REPO / ".vscode" / "settings.json").exists():
        report(OK, "VS Code settings", ".vscode/settings.json found")
    else:
        report(WARN, "VS Code settings", "missing", "Run git pull, or clone the repository again.")
    if shutil.which("code") is None:
        report(
            WARN,
            "VS Code command",
            "'code' is not on the PATH (optional)",
            "Not needed: open the folder from VS Code with File → Open Folder…",
        )
    else:
        report(OK, "VS Code command", "code found")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print("Checking your setup for Statistical Programming (MPMD, HTW Berlin)")
    for title, check in [
        ("Computer", check_computer),
        ("Tools", check_tools),
        ("Python", check_python),
        ("Packages (this takes a few seconds)", check_packages),
        ("VS Code", check_vscode),
    ]:
        print(f"\n{title}")
        check()

    failed, warned = results.count(FAIL), results.count(WARN)
    print()
    if failed:
        print(
            f"❌ {failed} problem(s) to fix, {warned} warning(s). Follow the → lines above, then run "
            "uv run check_setup again. Still stuck? Post the full output in the course forum."
        )
        return 1
    if warned:
        print(f"✅ Ready to work, with {warned} warning(s) to sort out before they are needed.")
    else:
        print("✅ Everything is set up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
