---
name: tdd-pytest
description: Test-driven development with pytest — the red/green/refactor loop, test design and naming, fixtures, parametrization, monkeypatching, mocking at boundaries, and meaningful coverage. Use when implementing new Python behavior, fixing a bug (write the failing test first), or improving or extending an existing Python test suite.
whenToUse: Writing or changing Python code where tests apply — features, bug fixes, refactors, or a weak test suite that needs strengthening.
---

# TDD with pytest

Drive code from tests. The test is the first user of your API — write it before the implementation and watch it fail for the right reason.

## The loop

1. **Red** — write one failing test that expresses the next behavior. Run it; confirm it fails because the behavior is missing (not because of a typo).
2. **Green** — write the minimal code to pass. Resist gold-plating.
3. **Refactor** — clean up while the test keeps you safe.

Keep the loop small: minutes, not hours. Run the suite frequently; never leave it red overnight.

## Test structure

- One behavior per test. Name it `test_<behavior>_when_<condition>` — the name is the spec: `test_raises_when_token_expired`.
- Arrange → Act → Assert, separated by blank lines; keep Assert last and specific.
- Assert real outcomes, not implementation details: prefer `assert result.price == 10` over checking which internal method was called.
- Make failures informative: use pytest's `assert` rewriting; add a message only when it adds context.
- Prefer plain asserts over `assertTrue`/`assertEqual` helpers.

## pytest essentials

| Need | Tool |
|---|---|
| One-off temp dir | `tmp_path` fixture (per-test) |
| Patch an attribute/method | `monkeypatch.setattr(obj, "name", fake)` (auto-undo) |
| Raise on a dependency call | `monkeypatch.setattr` + a function that raises |
| Many input cases | `@pytest.mark.parametrize("input,expected", [...])` |
| Expect an exception | `with pytest.raises(ValueError, match="token"):` |
| Float comparisons | `pytest.approx(0.1 + 0.2)` |
| Skip / expected failure | `@pytest.mark.skipif(...)`, `@pytest.mark.xfail(...)` |
| Shared setup per scope | `@pytest.fixture(scope="module" / "session")` in `conftest.py` |
| Test categories | `@pytest.mark.slow`, then run `-m "not slow"` |

## Fixtures & isolation rules
- Tests must be **deterministic**: no wall-clock dependence, no network, no real I/O, no reliance on execution order.
- Replace time with injection (`datetime.now` monkeypatched, `freezegun`-style fake clocks); seed random number generators.
- Use `tmp_path` instead of writing to the repo or system temp.
- DB/file state belongs to fixtures with clear teardown — never leak state between tests.
- A test that depends on another test's side effects is a bug in the suite.

## Mocking guidance
- **Prefer real objects + dependency injection** over mocks; mock at **boundaries** (HTTP clients, clocks, external services), not internals you own.
- Over-mocking makes tests tautological — they verify the mock, not your code. If a test needs many mocks, the design is coupling too tightly.
- With `monkeypatch` you never need `mock.patch` cleanup; use it consistently.

## Coverage — meaningful, not maximal
- Run: `pytest --cov=<package> --cov-report=term-missing` (needs `pytest-cov`).
- Track **branch** coverage for conditionals: `--cov-branch`.
- Coverage is a floor, not a goal: 100% of lines with zero assertions protects nothing. Prefer strong tests on risky logic over padding easy modules.
- Set CI thresholds honestly (e.g. `--cov-fail-under=80`) and raise them deliberately.

## Bug fixes, TDD style
1. Reproduce: write a test that fails with the bug present (`test_...`).
2. Confirm it fails for the **actual** bug (message/behavior matches the report).
3. Fix the code minimally; test goes green.
4. Keep the regression test — it documents the bug forever. Never delete it to make CI pass.

## Refactoring with a safety net
- Refactor only when the relevant tests are green; after each structural step, re-run.
- If the suite is thin, write characterization tests (record current behavior) before touching legacy code.

## Suite habits
- Fast suite runs in seconds — keep slow tests behind `@pytest.mark.slow`.
- Run the focused test first (`pytest path/to/test.py::test_name -x`), the file next, then the full suite before finishing.
- Watch for tests that pass for the wrong reason (missing assert, `xfail` that no longer fails, test skipping silently via bad `skipif`).
