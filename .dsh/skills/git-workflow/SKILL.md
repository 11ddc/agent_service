---
name: git-workflow
description: Consistent and safe Git practice — Conventional Commits message format, small atomic commits, branch hygiene, rebase vs merge decisions, and a safe undo/rollback playbook. Use whenever preparing a commit, writing a commit message, cleaning up history, resolving conflicts, reverting or resetting changes, or reviewing branch history.
whenToUse: Any commit, history, branch, conflict, or rollback task in a Git repository.
---

# Git Workflow

Follow a predictable, reviewable workflow. History should read like a changelog: each commit is one logical change with a clear message.

## Commit messages — Conventional Commits

```
<type>(<optional scope>): <imperative summary, ≤ ~50 chars>

<optional body: what and why, one idea per bullet>
```

| Type | When |
|---|---|
| `feat` | New user-visible or API feature |
| `fix` | Bug fix |
| `refactor` | Behavior-preserving restructuring |
| `perf` | Performance improvement |
| `test` | Add/fix tests |
| `docs` | Documentation only |
| `style` | Formatting, whitespace, lint-only |
| `build` / `ci` | Build system / CI config |
| `chore` | Maintenance, deps, tooling |
| `revert` | Reverts a previous commit |

- Summary in **imperative mood**: "add pagination to list endpoint", not "added" / "adds".
- Body explains **why**, not what the diff already shows; reference issue IDs where relevant.
- Breaking change: add `!` after type/scope (`feat!: drop v1 endpoint`) and explain in the body.
- Scope is optional but consistent (`feat(api): ...`).

## Atomicity & hygiene
- One logical change per commit. Split unrelated edits (`git add -p`).
- Commit **working, buildable** states; no half-finished refactors.
- Before committing: `git status` and `git diff` to review exactly what will be committed — never `git add -A` blindly.
- Never commit secrets, generated artifacts, or local config (keep them in `.gitignore`).

## Branches & integration
- Short-lived topic branches off a stable base (trunk-based or feature-branch model, whichever the repo uses — match the team).
- **Rebase** to integrate upstream changes on a **local, unpublished** branch: keeps history linear.
- **Merge commit** for integrating shared/published branches where preserving topology matters.
- Keep the branch up to date: `git fetch` then `git rebase <base>` (or `git pull --rebase`) — avoid `git pull` default merges that create noise.
- Typical pre-push pass: `git rebase -i <base>` to squash fixup commits and tidy messages — **only before the branch is shared**.

## Conflicts
- Read both sides before editing; do not blindly keep yours or theirs.
- Theirs/yours semantics differ by command (`merge --theirs` = their branch; `rebase --theirs` = actually *your* commits being replayed). When unsure, edit the file manually.
- Stuck? `git merge --abort` / `git rebase --abort`, then retry. Enable `rerere` (`git config rerere.enabled true`) to remember resolutions.

## Undo / rollback playbook

| Situation | Command | Safe on shared history? |
|---|---|---|
| Last commit message wrong | `git commit --amend` | Only if unpushed |
| Forgot a file in last commit | `git add <file> && git commit --amend --no-edit` | Only if unpushed |
| Discard working-tree changes to a file | `git restore <file>` | Yes (destroys local edits) |
| Unstage a file | `git restore --staged <file>` | Yes |
| Undo the last commit, keep changes staged | `git reset --soft HEAD~1` | Only if unpushed |
| Undo last N commits, keep working tree | `git reset HEAD~N` | Only if unpushed |
| Remove a commit that was **pushed/shared** | `git revert <sha>` (creates an inverse commit) | Yes — the right tool |
| Revert a merge | `git revert -m 1 <merge-sha>` | Yes — do not `reset` a shared merge |
| I deleted/mangled something | `git reflog` to find the old sha, then reset/branch from it | Recovery only |

**Golden rules**
- Never `reset`/force-push history that others may have fetched. Prefer `revert`.
- `revert` when the commit is public; `reset`/`amend` only for your own unpushed commits.
- After any history rewrite others share, coordinate — it requires everyone to re-base.
- When in doubt, copy the branch (`git branch backup`) before destructive operations.

## Inspecting history
- `git log --oneline --graph --decorate` — topology at a glance.
- `git blame <file>` — find who/when/why a line changed.
- `git show <sha>` — full diff of one commit.
- `git diff <base>...<head>` — changes on a branch as a whole (use `...`, the merge-base form).

## Final checks before push/PR
- Branch contains only intended commits (`git log --oneline <base>..`).
- Commit messages follow Conventional Commits (CI may enforce with commitlint).
- No merge conflict markers, no debug leftovers, no secrets, tests pass.
