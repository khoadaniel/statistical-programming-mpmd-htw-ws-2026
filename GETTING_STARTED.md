# Getting started

> [!IMPORTANT]
> **A green setup check is the prerequisite for this course. Please complete this guide before the first class.** At the end, this command must finish with a ✅ on its last line:
>
> ```bash
> uv run check_setup
> ```
>
> If you get stuck, post the output of the check in the course forum before Session 1, so that it can be fixed before class rather than during it.

You set up your laptop once. It takes about 20 minutes and about 5 GB of free disk space.

> [!IMPORTANT]
> **The course material changes during the semester.** Update your copy before you start working on something: see [Keeping up to date](#keeping-up-to-date).

Everyone uses the same tools: **VS Code** as the editor, **Git** to get the course material, and **uv**, which installs Python 3.14 and the exact package versions the course is tested with. You do not need to install Python, `pip` or Anaconda yourself. If you already have Python or Anaconda, you can keep it; uv does not touch it.

**Supported computers:** a Mac with Apple silicon (M1 or newer) and macOS 14 or newer; a Windows 10 or 11 PC with an Intel or AMD processor; Linux (x86-64 or ARM64) such as Ubuntu 22.04 or newer. Intel Macs and Windows PCs with an ARM processor are not supported: the machine-learning packages of the course no longer publish versions for them.

A **terminal** is where you type the commands below: on macOS the app *Terminal*; on Windows *PowerShell* (Start menu → type "PowerShell"); on Linux your terminal app. Type or paste one command at a time and press Enter.

| Step | What | Check |
|---|---|---|
| 1 | [Install VS Code](#step-1--install-vs-code) | VS Code starts |
| 2 | [Install Git](#step-2--install-git) | `git --version` |
| 2b | [Create a GitHub account and tell Git who you are](#create-a-github-account-and-tell-git-who-you-are) | your profile page opens |
| 2c | [Log in to GitHub from your laptop](#log-in-to-github-from-your-laptop) | `gh auth status` |
| 3 | [Install uv](#step-3--install-uv) | `uv --version` |
| 4 | [Get the course repository and create the environment](#step-4--get-the-course-repository-and-create-the-environment) | `uv sync` finishes |
| 5 | [Check your setup](#step-5--check-your-setup) | `uv run check_setup` ends with ✅ |
| 6 | [Open the repository in VS Code and run a notebook](#step-6--open-the-repository-in-vs-code-and-run-a-notebook) | the kernel reads `statistical-programming (3.14…)` |
| 7 | [Download the course data](#step-7--download-the-course-data) | the data check lists the files |

## Step 1 · Install VS Code

Download and install VS Code from **https://code.visualstudio.com/download**, the same on every system.

## Step 2 · Install Git

Git downloads and updates the course material.

<details open>
<summary><b>macOS</b></summary>

Install the package manager Homebrew from **https://brew.sh**: copy the command shown on that page into Terminal and press Enter (it asks for your password and takes a few minutes). At the end, Homebrew prints two or three commands under "Next steps" that add `brew` to your terminal; run them. Then install Git:

```bash
brew install git
```
</details>

<details>
<summary><b>Windows</b></summary>

Download Git from **https://git-scm.com/downloads/win** and run the installer; the default options are fine.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
sudo apt install git curl     # Ubuntu, Debian
sudo dnf install git curl     # Fedora
```

`curl` downloads the uv installer in step 3; many Linux systems do not include it.
</details>

Check that Git works (close and reopen the terminal first):

```bash
git --version
```

### Create a GitHub account and tell Git who you are

Your team works on GitHub from Session 2: a shared repository, branches, pull requests and reviews. Your own GitHub account is the evidence of your work in the final project ([how it is assessed](sessions/17-18-final-presentations/final-project.md#presentation-and-assessment)). So everyone needs a GitHub account:

1. Sign up at **https://github.com/signup** (free). Choose a username you are happy to show an employer: your GitHub profile is part of your portfolio. Use an e-mail address you will keep after your studies.
2. Optionally add your real name and a photo under **Settings → Public profile**, so that your team and the lecturer recognise you. Your profile is then at `https://github.com/<your-username>`; the lecturer's, for example, is [github.com/khoadaniel](https://github.com/khoadaniel).
3. Tell Git on your laptop who you are, with the **same e-mail as your GitHub account**, so that GitHub links your commits to your profile:

```bash
git config --global user.name "Your Name"
git config --global user.email you@example.com
```

Check it with `git config --global --list`. Have the account ready before Session 2 and share your username with your team.

These two commands only label your commits; they do not log you in. GitHub asks who you are the first time you send work to it (`git push`), and it does not accept your GitHub password there. The next step sets up the login once, so that `git push` works.

### Log in to GitHub from your laptop

The GitHub command-line tool `gh` logs you in through your web browser and then lets Git use that login. Install it:

<details open>
<summary><b>macOS</b></summary>

```bash
brew install gh
```
</details>

<details>
<summary><b>Windows</b> (PowerShell)</summary>

```powershell
winget install --id GitHub.cli
```

If `winget` is not found, download the installer from **https://cli.github.com** instead.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
sudo apt install gh     # Ubuntu, Debian
sudo dnf install gh     # Fedora
```
</details>

Close the terminal and open a new one, then log in:

```bash
gh auth login
```

It asks a few questions; choose with the arrow keys and Enter:

1. *Where do you use GitHub?* → **GitHub.com**
2. *What is your preferred protocol for Git operations?* → **HTTPS**
3. *Authenticate Git with your GitHub credentials?* → **Yes**
4. *How would you like to authenticate?* → **Login with a web browser**

It shows a one-time code. Press Enter, sign in to GitHub in the browser that opens, and type the code. Check the result:

```bash
gh auth status
```

It prints `Logged in to github.com account <your-username>`. You do this once per laptop; Git now pushes to GitHub without asking for a password. Source: [GitHub CLI manual](https://cli.github.com/manual/gh_auth_login).

## Step 3 · Install uv

<details open>
<summary><b>macOS and Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
</details>

<details>
<summary><b>Windows</b> (PowerShell)</summary>

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
</details>

**Close the terminal and open a new one**, then check that uv is found:

```bash
uv --version
```

It prints a version number such as `uv 0.13.0`. If it says "command not found", restart the computer once. Source of these commands: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## Step 4 · Get the course repository and create the environment

In the terminal, go to the folder where you keep your course work (for example your Documents folder), then download the repository and create the environment:

```bash
cd Documents
git clone https://github.com/khoadaniel/statistical-programming-mpmd-htw-ws-2026.git
cd statistical-programming-mpmd-htw-ws-2026
uv sync
```

`uv sync` downloads Python 3.14 and every package of the course into a folder `.venv` inside the repository (about 2.5 GB, a few minutes). This `.venv` is **the** course environment: every notebook and script in this repository runs in it. Every later command in this guide is typed in this repository folder.

## Step 5 · Check your setup

```bash
uv run check_setup
```

The check looks at your computer, Git and your Git identity, uv, Python and the course environment, the main course packages, and the VS Code settings of the repository. It prints one line per check:

- ✅ the check passed;
- ⚠️ something to fix before it is needed (for example your Git name and e-mail before Session 2); you can start Session 1;
- ❌ a problem to fix now. The line below it, starting with →, says how.

Fix what it reports and run it again until the last line starts with ✅. The check changes nothing on your computer, so you can run it as often as you like, for example after every `git pull`. Its code is in [`course_tools/check_setup.py`](course_tools/check_setup.py).

## Step 6 · Open the repository in VS Code and run a notebook

1. Start VS Code, choose **File → Open Folder…** and select the folder `statistical-programming-mpmd-htw-ws-2026`. If VS Code asks whether you trust the authors, choose **Yes, I trust the authors**.
2. VS Code offers to install the **recommended extensions** (Python, Jupyter, Ruff, Markdown Mermaid for the diagrams) in a pop-up at the bottom right; choose **Install**. If you missed it: open the Extensions view (the four squares on the left), type `@recommended` and install the four listed under *Workspace Recommendations*.
3. Open `sessions/01-careers-and-python/workbooks/02-lecture-notebooks-python-and-numpy.ipynb` and run the first **code** cell (▶ to its left, or Shift+Enter).
4. Look at the top right of the notebook: the kernel must read **statistical-programming (3.14…)**, the course environment in `.venv`. VS Code usually picks it automatically. If it shows something else, or asks you to select a kernel, choose the course environment yourself (see below). You do this once; VS Code remembers it for the folder.

**Choosing the course environment.** Click the kernel name at the top right (or **Select Kernel**) → **Python Environments…**. The list shows every Python on your computer, so it may look like this:

```text
+ Create Python Environment                                   Recommended   ← do not choose
★ statistical-programming (3.14.7)   .venv/bin/python                       ← choose this one
```

Choose the entry named **statistical-programming** whose path is **`.venv/bin/python`** (on Windows `.venv\Scripts\python.exe`). The version numbers on your computer may differ. Do not choose *Create Python Environment*, even though VS Code marks it *Recommended*: it builds a new environment without the course packages. The other entries are Pythons installed elsewhere on your computer, without the course packages, even when their version is also 3.14.

The repository contains the VS Code settings for this ([`.vscode/settings.json`](.vscode/settings.json), [`.vscode/extensions.json`](.vscode/extensions.json)). Terminals opened inside VS Code use the `.venv` automatically, so `python` and `pytest` there are the course versions.

## Step 7 · Download the course data

The notebooks read two datasets that are not in the repository. Download `airbnb-v1.zip` and `ebti-v1.zip` from the course drive and unzip them inside `case-study/data/`, as described in [Getting the data](case-study/README.md#getting-the-data), including the command that checks the result.

You are ready for Session 1.

## Keeping up to date

Notebooks are corrected, exercises are added, and sometimes new packages are needed. Working on an old copy means old mistakes and missing files. Pull the latest version **before you start working on something**, in the repository folder:

```bash
git pull
uv sync
uv run check_setup
```

Do it at the start of each session, before an exercise and before you open a notebook at home. `git pull` fetches the new material; `uv sync` updates the environment if the course packages changed; `uv run check_setup` confirms that everything still works. Keep your own work in copies (for example `my-notes.ipynb`) or in your team repository, so that `git pull` does not conflict with it ([Troubleshooting](#troubleshooting)).

## Later in the course

- **Everything is in one environment.** `uv sync` installs all packages of the course at once, including PyTorch for the language-model sessions; there is nothing to add later. A few optional notebooks in Sessions 12 and 16 need other package versions and run in a temporary environment of their own; their session README gives the command.
- **Databases** (Session 3 onwards) use SQLite, which is part of Python: there is nothing to install. The deployment session (16) needs [Docker Desktop](https://www.docker.com/products/docker-desktop/); its README gives the commands.
- **Language models** (Sessions 14–15) work with a local model through [Ollama](https://ollama.com/) or with any OpenAI-compatible API; see the session READMEs.
- **Workspaces** (Sessions 2, 15, 16) are small projects with their own environment: `cd sessions/<session>/workspace`, then `uv sync` and `uv run pytest`. Session 2 also has a tiny [CLI workspace](sessions/02-software-engineering/cli-workspace/README.md), the command `staycost`, to try how a script becomes a command.

## Troubleshooting

Run `uv run check_setup` first: most problems show up there, each with a line starting with → that says how to fix it. Every command below is typed in a terminal **inside the repository folder** (`cd` into `statistical-programming-mpmd-htw-ws-2026` first).

| What you see | What it means | What to do |
|---|---|---|
| `uv: command not found` (Windows: `uv is not recognized`) | The terminal was opened before uv was installed, so it does not know uv yet. | Close the terminal and open a new one. If that does not help, restart the computer. |
| `uv run check_setup` says the command is not found | You are not in the repository folder, or your copy is older than this guide. | `cd` into the repository folder, run `git pull` and `uv sync`, then run the check again. |
| When you run a notebook cell, VS Code asks you to **select a kernel**, or the top right of the notebook shows something other than `statistical-programming (3.14…)` | The notebook is not using the course environment. | Choose **statistical-programming** with the path `.venv/bin/python`, as in [Step 6](#step-6--open-the-repository-in-vs-code-and-run-a-notebook). |
| In that list (**Select Kernel → Python Environments…**) there is no entry **statistical-programming** with the path `.venv/bin/python` | The course environment does not exist yet, or VS Code started before it was created. | 1. In a terminal in the repository folder run `uv sync`. 2. In VS Code press Ctrl+Shift+P (Cmd+Shift+P on a Mac), type **Reload Window** and press Enter. 3. Pick the kernel again; **statistical-programming** is now in the list. Also check that you opened the repository folder itself with **File → Open Folder…**, not a single notebook file. |
| `ModuleNotFoundError: No module named '…'` in a notebook | The notebook uses a different Python (see the kernel rows above), or your environment is older than the course material. | Check that the kernel is **statistical-programming**; then run `git pull` and `uv sync`. |
| Windows: `DLL load failed` or `[WinError 126]` when importing a package such as `torch` | The Microsoft Visual C++ runtime that some packages need is missing. | Download and run the [Microsoft Visual C++ Redistributable (X64)](https://aka.ms/vc14/vc_redist.x64.exe), then restart VS Code. |
| Windows: "running scripts is disabled on this system" in the VS Code terminal | PowerShell blocks scripts by default. | In PowerShell run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, confirm with `Y`, open a new terminal. |
| `git push` asks for a username and password, or says `Password authentication is not supported` | Git is not logged in to GitHub, and GitHub does not accept your account password for Git. | Press Ctrl+C, then follow [Log in to GitHub from your laptop](#log-in-to-github-from-your-laptop) and push again. |
| `git pull` says your local changes would be overwritten | You edited a file of the course material. | Save your work under a new name (for example `my-notes.ipynb`), then run `git restore <file>` for the course file and pull again. |
| Anything else | | Post in the course forum: the full error message and the complete output of `uv run check_setup`. |

XGBoost or LightGBM errors about `libomp`, `libgomp` or `vcomp140.dll` belong to Session 10; its [setup notes](sessions/10-tree-based-models/README.md#setup) explain the fix.
