---
name: debugging-playbook
description: Systematic debugging methodology — reproduce reliably, isolate the failing component, form and test hypotheses, instrument with logging/debuggers, bisect regressions, and verify the fix with a regression test. Use when a program misbehaves, a test fails unexpectedly, a bug is intermittent or hard to pin down, or when investigating a crash, hang, or regression.
whenToUse: Any investigation of unexpected behavior, failures, crashes, performance regressions, or flaky/intermittent issues.
---

# Debugging Playbook

Debugging is **root-cause investigation**, not guessing. Resist the urge to patch the symptom; each fix must explain the failure and must not recur.

## Mindset rules
1. **One change at a time.** Multiple edits at once destroy the evidence.
2. **Write down the hypothesis** before touching code. If you cannot state it, you are not ready to change anything.
3. **Do not guess-fix.** A fix without a verified cause is a new bug waiting to happen.
4. **Trust, but verify** — including your own assumptions about inputs, environment, and order of operations.

## The five phases

### 1. Reproduce
- Get a **reliable, minimal** reproduction. Reduce inputs/state until the bug appears with the smallest surface.
- If you cannot reproduce at will, you cannot confirm a fix — keep digging on the environment/conditions first (see Intermittent bugs).
- Record: exact command/input, versions, environment, and the expected vs actual result.

### 2. Read the evidence
- Stack trace: read **top-down** for the throw site, then bottom-up for who called it. Find the frame *you* wrote.
- Logs: look at the lines *before* the error — the cause usually precedes the symptom.
- Diff the behavior change: what was the last known-good state?

### 3. Hypothesize & isolate
- Form 2–3 ranked hypotheses; design one experiment to discriminate between them (a targeted print, a minimal repro, a boundary probe).
- **Bisect the space**: binary search inputs, code paths, or commits (see below) instead of reading line by line.
- Isolate layers: is it the data, the logic, the dependency, or the environment? Test each boundary.

### 4. Instrument
- Add logging at the boundaries (function entry/exit with key values), not inside every line.
- Python: `breakpoint()` / `pdb` for interactive inspection; `python -X dev` for extra warnings; `python -X faulthandler` to dump tracebacks on hang/crash; `PYTHONASYNCIODEBUG=1` for async deadlocks.
- Watch for silent failure: bare `except: pass`, ignored return values, swallowed exceptions.
- Add temporary assertions that encode your expectation — a failed assert is data.

### 5. Fix & verify
- Fix the **root cause** at the correct layer, minimally.
- Re-run the reproduction: it must be gone.
- **Add a regression test** that fails on the old code and passes on the new — this is what makes the fix permanent.
- Re-run the surrounding suite; check you did not fix one path by breaking another.

## Regression hunting with git bisect

When a bug appeared "sometime recently":

```
git bisect start
git bisect bad            # current broken state
git bisect good <sha>     # last known-good commit
# repeat: mark each tested commit good/bad
git bisect reset
```

Binary-search the history instead of reading every commit. Then read the single found commit to understand the change that broke it.

## Intermittent / flaky bugs
- Suspects, in order: **time** (races, TTLs, clocks, timezone), **state/order** (shared globals, dict/set iteration, test pollution), **environment** (locale, cwd, env vars, paths, resource limits), **external dependencies** (network, services, rate limits).
- Run the case many times with different seeds/order (`pytest -p randomly`, `--count=N`) to raise frequency.
- Add rich logging around the suspect region and capture it on failure — you cannot debug what you do not observe.
- For concurrency: check shared mutable state, missing locks, blocking calls inside async/event-loop contexts, and shutdown ordering.

## "Works on my machine" checklist
- Clean checkout vs dirty tree; untracked/stale files; version drift of dependencies (`pip freeze` diff).
- Python version, virtualenv vs system Python, `PYTHONPATH`, installed vs editable package.
- Case sensitivity, path separators, line endings, locale/encoding — classic Windows/Unix deltas.
- Same config/secrets present? Debug vs production settings?

## Postmortem habit
Finish with one short note: symptom → root cause → fix → regression test. It turns today's hour into next month's minute, and it is what separates fixing from firefighting.

## Anti-patterns to avoid
- Print-debugging the whole file top to bottom with no hypothesis.
- Fixing the symptom (retry harder, swallow the error) while the cause stays.
- Rewriting the module "to be safe" instead of changing one line.
- Declaring victory without re-running the original reproduction.
