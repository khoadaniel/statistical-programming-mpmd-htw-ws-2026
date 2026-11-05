# Git workspace · Session 2: two pull requests, with and without a conflict

A team activity for pairs. You share one fork of the course repository, change these small files on your own branches and open pull requests. First you change different files: both pull requests merge without a conflict. Then you change the same line: the second pull request has a conflict, and you resolve it.

```
git-workspace/
├── greeter.py   class Greeter: greet() returns "Hello, Ada!"   (both of you, the same line, in step 3)
├── counter.py   class Counter: increment() counts up            (Ada, in step 2)
├── listing.py   class Listing: describe() returns one line       (Ben, in step 2)
└── main.py      uses the three classes                          (run it after every step)
```

The files have no dependencies: `python3 main.py` is enough, no uv needed.

**Roles.** Decide who is **Ada**, the owner of the fork, and who is **Ben**, the collaborator. Below, `<ada>` stands for Ada's GitHub name. Lines starting with `$` are what you type; the lines under them are what Git prints. They come from a real run; your commit ids differ.

## Step 1 · Fork, invite, clone

1. **Ada** opens [the course repository](https://github.com/khoadaniel/statistical-programming-mpmd-htw-ws-2026), clicks **Fork**, then **Create fork**. The fork is `github.com/<ada>/statistical-programming-mpmd-htw-ws-2026`.
2. **Ada**, in the fork: **Settings**, **Collaborators**, **Add people**, then Ben's GitHub name.
3. **Ben** accepts the invitation, from the e-mail of GitHub or at github.com/notifications. Without it, Ben's `git push` is rejected.
4. **Both** clone Ada's fork (not the course repository) and run the program:

```bash
$ git clone https://github.com/<ada>/statistical-programming-mpmd-htw-ws-2026.git
Cloning into 'statistical-programming-mpmd-htw-ws-2026'...
$ cd statistical-programming-mpmd-htw-ws-2026/sessions/02-software-engineering/git-workspace
$ python3 main.py
Hello, Ada!
Counter: 2
Kreuzberg loft: 80 EUR per night
```

## Step 2 · Different files: no conflict

**Ada** adds a method `reset` to `counter.py`, on her own branch:

```python
    def reset(self) -> None:
        """Set the count back to zero."""
        self.value = 0
```

```bash
$ git switch -c ada/counter
Switched to a new branch 'ada/counter'
$ git add counter.py
$ git commit -m "Add reset to Counter"
[ada/counter 1a2b3c4] Add reset to Counter
 1 file changed, 4 insertions(+)
$ git push -u origin ada/counter
 * [new branch]      ada/counter -> ada/counter
branch 'ada/counter' set up to track 'origin/ada/counter'.
```

**Ben**, at the same time, adds a method `total_price` to `listing.py`:

```python
    def total_price(self, nights: int) -> int:
        """Return the price of a stay of this many nights."""
        return self.price_per_night * nights
```

```bash
$ git switch -c ben/listing
Switched to a new branch 'ben/listing'
$ git add listing.py
$ git commit -m "Add total_price to Listing"
[ben/listing 5d6e7f8] Add total_price to Listing
 1 file changed, 4 insertions(+)
$ git push -u origin ben/listing
 * [new branch]      ben/listing -> ben/listing
branch 'ben/listing' set up to track 'origin/ben/listing'.
```

GitHub also prints lines starting with `remote:` and a link to open a pull request.

**On GitHub, in Ada's fork.** Each of you opens a pull request: **Pull requests**, **New pull request**, compare your branch.

> ⚠️ For a fork, GitHub first proposes the course repository as the base. Set **base repository** to `<ada>/statistical-programming-mpmd-htw-ws-2026` and **base** to `main`. Your pull requests stay in your fork.

Review each other's pull request, then **Merge pull request**: first #1, then #2. **Predict:** will #2 conflict? It does not: the two branches changed different files, so nothing overlaps.

**Both** bring the two merges to your laptop:

```bash
$ git switch main
Switched to branch 'main'
Your branch is up to date with 'origin/main'.
$ git pull
Updating 1a2b3c4..9f8e7d6
Fast-forward
 sessions/02-software-engineering/git-workspace/counter.py | 4 ++++
 sessions/02-software-engineering/git-workspace/listing.py | 4 ++++
 2 files changed, 8 insertions(+)
```

"Up to date" only compares with your last fetch; `git pull` fetches the two merges and moves your `main` forward.

## Step 3 · The same line: a conflict

Both of you start a new branch from the updated `main` and change **line 12** of `greeter.py`, `return f"Hello, {self.name}!"`.

**Ada:**

```bash
$ git switch -c ada/hallo
# greeter.py, line 12: Hello becomes Hallo
$ git diff -U0
-        return f"Hello, {self.name}!"
+        return f"Hallo, {self.name}!"
$ git commit -am "Greet in German"
$ git push -u origin ada/hallo
```

**Ben:**

```bash
$ git switch -c ben/xin-chao
# greeter.py, line 12: Hello becomes Xin chào
$ git diff -U0
-        return f"Hello, {self.name}!"
+        return f"Xin chào, {self.name}!"
$ git commit -am "Greet in Vietnamese"
$ git push -u origin ben/xin-chao
```

Both open a pull request, as in step 2. **Predict:** which one will conflict? Merge Ada's #3 first. Ben's #4 now shows **"This branch has conflicts that must be resolved"**: the same line has two different versions, and only people can decide which is right. Nothing is lost; Ben's commit is still on his branch.

## Step 4 · Resolve the conflict and update the pull request

**Ben** brings the new `main` into his branch:

```bash
$ git switch ben/xin-chao
$ git fetch origin
$ git merge origin/main
Auto-merging sessions/02-software-engineering/git-workspace/greeter.py
CONFLICT (content): Merge conflict in sessions/02-software-engineering/git-workspace/greeter.py
Automatic merge failed; fix conflicts and then commit the result.
$ git status --short
UU greeter.py
```

`greeter.py` now holds both versions between conflict markers (lines 12 to 16):

```text
<<<<<<< HEAD
        return f"Xin chào, {self.name}!"
=======
        return f"Hallo, {self.name}!"
>>>>>>> origin/main
```

`HEAD` is Ben's branch; `origin/main` is Ada's change, already on `main`. Agree on one line, for example `return f"Hallo and xin chào, {self.name}!"`, and delete the three marker lines. Then:

```bash
$ git add greeter.py
$ git commit --no-edit
[ben/xin-chao 3c4d5e6] Merge remote-tracking branch 'origin/main' into ben/xin-chao
$ git push
```

`--no-edit` keeps the message Git proposes. The push updates pull request #4: the conflict message is gone. Merge it on GitHub, then **both**:

```bash
$ git switch main
$ git pull
$ python3 main.py
Hallo and xin chào, Ada!
Counter: 2
Kreuzberg loft: 80 EUR per night
```

## What to remember

- Changes in different files, or in different regions of one file, merge without a conflict.
- Changes to the same line conflict in the pull request that is merged second.
- Fix it on the branch of that pull request: `git fetch`, `git merge origin/main`, decide, delete the markers, `git add`, `git commit`, `git push`. The same pull request updates itself; you never open a new one.
- Run `git log --oneline --graph --all` at the end and find the two merge commits of step 4.
