# SERF Coding Standards

> **The standard: write the simplest code that satisfies the specific requirement in front of you, and nothing else. A researcher should be able to read any SERF module top to bottom, in one sitting, and understand exactly what it does.**

SERF is a research project that we want people to adopt, read, and build on. Research code that nobody can read gets reimplemented instead of cited. Every line we add is a line a reader has to get through before they reach the idea. Treat reader attention as the scarcest resource in the project.

This document covers _how we decide what to write_. For mechanical conventions — formatter, import style, logging, config — see [CLAUDE.md](../CLAUDE.md), which remains authoritative on those.

---

## The Rule

**Implement the requirement. Not the requirement plus the three things you imagine might be asked for next.**

When you find yourself writing code for a case nobody asked about, stop and delete it. If the case turns out to matter, we will add it then, with a test that shows why. Speculative generality is the main way research codebases become unreadable, and it is not free: every branch is a branch the reader must hold in their head and the maintainer must keep correct.

### What this Rules Out

**Defensive checks for conditions that cannot happen.** If a function is only ever called with a non-empty block, do not check for an empty block. If the pipeline guarantees a column exists, do not check whether it exists. Let it fail loudly at the real boundary instead of failing quietly in ten places.

```python
# No — three branches, two of which never execute
def resolve(block: EntityBlock | None) -> BlockResolution:
    if block is None:
        return BlockResolution(...)
    if not block.entities:
        return BlockResolution(...)
    if len(block.entities) == 1:
        return BlockResolution(...)
    ...

# Yes — the caller guarantees a non-empty block; singletons are handled by the caller,
# where the skip reason belongs
def resolve(block: EntityBlock) -> BlockResolution:
    ...
```

**Configuration for things that have one value.** A parameter that is never passed anything but its default is not a parameter, it is a constant with extra steps. Add the knob when the second caller appears.

**Abstraction with one implementation.** No base classes, protocols, registries, or factories until there are at least two concrete things to abstract over. One `Blocker` class is a class. One `AbstractBlocker` with one `SemanticBlocker` subclass is a class and a riddle.

**Don't create unecessary clases**. Try to use `dicts` for configuration, or anywhere else you can.

**Wrappers that only forward.** If a function's body is a single call to another function with the same arguments, delete it and call the other function.

`try`**/**`except ImportError`**.** Assume dependencies are installed. If they are not, the crash is correct and the traceback is the error message.

**Limit comments intended for yourself.** Do not write many lines of comments for one property when one will do for a human.

### What is Not Ruled Out

Minimalism is about removing incidental complexity, never about removing correctness. These stay:

- **The identifier conservation logic.** It is intricate because the problem is intricate. See [ID_INVARIANTS.md](ID_INVARIANTS.md). Every branch there earns its place and is covered by a test that names the case it protects.
- **Error recovery around LLM calls.** The network fails and models return malformed output. That is a real case, not a hypothetical one.
- **Docstrings on public functions.** NumPy style, with types. Reading is the point.
- **Tests.** Minimal code is not untested code. A small module with a thorough test file is exactly what we want.

---

## Functions

**Aim for one screen.** If a function does not fit in about forty lines, it is usually doing two things. The fix is almost always to name the second thing, not to add comments explaining where it starts.

**One reason to exist.** A function that loads data _and_ transforms it _and_ writes it has three reasons to change. Split it — _except in Spark dataflows_, where the opposite rule applies (see below).

**Return early.** Guard clauses at the top beat nested conditionals in the middle. Prefer three early returns over three levels of indentation.

**Name for what it produces, not how.** `resolve_block` beats `process_block_via_llm_call`. The reader wants to know what they get, not how you got it.

## Spark is different, deliberately

Spark dataflows are the one place where we prefer one long linear function over several small ones. A dataflow broken into `load_x`, `compute_y`, `join_z` forces the reader to jump around to reconstruct a sequence that was linear to begin with, and it obscures the shuffle boundaries that actually determine performance.

Write the dataflow top to bottom in one function. Assume columns exist. Assume paths exist. Do not handle edge cases nobody asked about. This is the rule in [assets/PYSPARK.md](../assets/PYSPARK.md) and it wins over the general "small functions" guidance whenever the two conflict.

## Comments

**Comment the constraint, not the code.** The code says what it does. A comment earns its place only by saying something the code cannot: a non-obvious invariant, a protocol requirement, a counterintuitive reason for an ordering.

```python
# No — restates the line below it
# Increment the iteration counter
iteration += 1

# Yes — states a constraint the reader cannot see
# The master must be the lowest input id: downstream evaluation joins on it
# and Abzu's cross-iteration UUID validation assumes this ordering.
master_id = min(entity_ids)
```

Never write a comment addressed to the reviewer rather than the reader. No "changed this to fix X", no "as requested", no "this is now correct". Those are pull-request conversation, and they are noise from the moment the PR merges.

## Types

Annotate everything. Modern builtin generics (`list[str]`, `dict[str, int]`), `Optional` from `typing` for optional parameters. Annotations are for the reader first: a signature that states its types is documentation that cannot go stale.

There is currently no type checker wired into the project. Annotations are still required; they are just not machine-verified right now.

## Tests

**Test the behavior, not the implementation.** A test that breaks when you rename a private method is a test that costs more than it protects.

**Name the case in the test name.** `test_recovers_entity_dropped_by_llm` tells a reader what invariant is at stake. `test_unmap_block_2` does not.

**Every invariant in** [ID_INVARIANTS.md](ID_INVARIANTS.md) **has a test that would fail if the invariant broke.** This is the one area where we are deliberately thorough rather than minimal.

**Mock the LLM in unit tests.** Real API calls belong in integration tests, marked, and skipped when no key is present.

pytest style — module-level functions, fixtures for setup. No test classes.

## The CLI

Every capability is reachable from `serf`. If you build something that cannot be run from the command line, it cannot be reproduced, and if it cannot be reproduced it cannot go in the paper.

Logic lives in `serf.<module>`. The CLI in `serf.cli` reads arguments, calls one function, and prints the result. When you catch yourself writing business logic inside a Click command, move it.

## Before you commit

```bash
uv run ruff check --fix src tests && uv run ruff format src tests
uv run pytest tests/
```

Fix both without being asked.

---

## The review question

For every module, function, and branch you add, one question:

> **Would a researcher reading this for the first time understand why it is here?**

If the answer is no, the code is wrong even if it works. Delete it, simplify it, or explain the constraint in one line.
