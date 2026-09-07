---
name: code-review
description: Structured, checklist-driven code review covering correctness, security, performance, reliability, API contracts, tests, and maintainability. Use when asked to review a pull request, diff, or branch; when self-reviewing changes before commit; or when another agent asks for a review pass.
whenToUse: Any review of code changes — before submitting a PR/MR, when evaluating a diff or branch range, or as a final self-review gate.
---

# Code Review

Act as a careful senior reviewer. You are reviewing **the change**, not the author. Read the diff **in its context** — a hunk alone rarely tells the whole story.

## Workflow

1. **Get the exact change set.** For a branch: `git diff <base>...<head>`; for review of staged work: `git diff --cached`; for a range of commits: `git log --oneline <base>..<head>`.
2. **Read the touched files**, not just hunks. Open each modified file to see surrounding functions, callers, and the module's conventions.
3. **Walk the checklist below**, collecting findings by severity. Do not rubber-stamp; do not nitpick trivia either.
4. **Trace the hard paths:** what happens on empty input, max input, missing keys, failure of every external call, and concurrent access?
5. **Write findings** in the comment format below, grouped by severity. Then give an overall verdict (approve / approve-with-nits / needs-changes).

## Severity levels

| Level | Meaning | Blocks merge? |
|---|---|---|
| **Blocker** | Wrong behavior, security hole, data loss/corruption, crash on a reachable path | Yes |
| **Major** | Real bug in edge cases, missing error handling, broken contract, untested risky path | Ideally yes |
| **Minor** | Suboptimal but safe; can ship and fix later | No |
| **Nit** | Style, naming, docs, formatting | No |

## Checklist

### Correctness
- Logic matches the stated intent; off-by-one, inverted conditions, wrong operator.
- Boundary values: empty, null/None, single element, max size, first/last.
- Error paths: what does every `except`/early-return do? Are errors swallowed silently?
- Concurrency: shared state, races, locks, async scheduling assumptions.
- Resource lifecycle: files, sockets, DB connections, locks are closed on **all** paths (including exceptions).

### Security
- Injection surfaces (SQL, shell, path, template, HTML, deserialization) — untrusted input is validated/parameterized.
- Authorization is checked on every entry point, not just the UI.
- Secrets never logged, hard-coded, or committed; dependency versions not pinned to vulnerable ranges.
- Sensitive data not leaked in responses, logs, or stack traces.

### Performance
- Obvious complexity traps: N+1 queries, repeated recomputation in loops, O(n²) on hot paths, blocking calls under async/event loops.
- Work proportional to need: no eager full scans when a filter exists.

### Reliability
- External calls have timeouts and retry/backoff where it matters; retries are **idempotent**.
- Failure is loud where it must be (observability), graceful where it can be.

### API & contracts
- Breaking changes to interfaces/schemas flagged, with migration or versioning considered.
- Validation of inputs happens at the boundary; error semantics are consistent.

### Tests
- The new behavior is actually exercised — happy path **and** the tricky branches this change touches.
- Assertions check behavior, not implementation trivia; failure messages would be informative.
- No test that can pass for the wrong reason (missing assert, always-true condition, skipped silently).

### Maintainability
- Naming reflects meaning; no dead code or commented-out blocks; no large unexplained duplication.
- Complexity is justified and documented; comments say *why*, code says *what*.

## Comment format

For each finding write: `[severity] path:line — what's wrong · why it matters · concrete suggestion` with a short example when helpful.

**Rule:** if you flag something, be ready to justify it and suggest a fix. If you don't understand why a line exists, ask rather than assume it is wrong.

## Etiquette & rules
- Review the code, not the person; praise what is genuinely good.
- Verify claims — do not report a bug you have not traced.
- Before you call something a blocker, be certain; otherwise say "Major — please confirm".
- End with a one-line overall verdict and, when merge-blocking findings exist, the single most important fix to start with.
