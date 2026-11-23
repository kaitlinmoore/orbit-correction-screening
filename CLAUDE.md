# CLAUDE.md: orbit-correction-screening
Standard: Research Project Standard v2

## What this is
A PhysicsNeMo model that learns the error of SGP4 against a high-fidelity numerical propagator, tested on whether corrected predictions screen LEO conjunctions better than uncorrected ones and classical filters.

## Principles
- AI may write any part of this project. I decide what I write myself; don't assign work to me.
- Every component gets clear comments and docstrings, a write-up in learning/ (gitignored), and a practice notebook in practice/ where useful.
- Work is exploratory. Instruments can change freely; log each change in docs/changelog.md with its reason. Leave the held-out split unexamined during development. Freeze only before a confirmatory run (docs/frozen_config.md).
- Before a change that could affect earlier results, frozen configs or held-out data, tell me what it touches and the tradeoffs, then let me decide.
- Propose a plan and wait for my approval before large changes, restructures or deletions.
- Honest framing: separate what I implemented from library behavior, disclose AI assistance, state limits, report nulls.
- Tests for deterministic logic, against fixtures in tests/fixtures/.

## Repo conventions
- Data stays out of git. The README records data sources with URLs, versions, access dates and licenses.
- Space-Track data is never committed or redistributed (user agreement).
- Secrets in .env; .env.example lists key names only.

## Environment
WSL2 (Ubuntu), because PhysicsNeMo supports Windows only through WSL. Claude Code runs inside WSL. The repo lives at C:\dev\GitHub\orbit-correction-screening (/mnt/c/dev/GitHub/orbit-correction-screening from WSL); the uv environment lives at $HOME/.venvs/orbit-correction-screening, set by UV_PROJECT_ENVIRONMENT in .envrc.
uv, Python 3.12 (matches lunar-thermal-surrogate; numpy 2.5 and scipy 1.18 need it). PyTorch must be built for CUDA 12.8 or newer (RTX 5080, Blackwell). Pinned versions: pending, taken from lunar-thermal-surrogate's recorded environment.
Key deps: nvidia-physicsnemo, torch, sgp4, astropy, numpy, numba, scipy. Truth propagator: Orekit through orekit-jpype (jdk4py extra) with the orekit-data bundle pinned by git commit, in a data-generation dependency group; the training process never starts the JVM. brahe for the cross-check.

## Project-specific
- Design document: orbit_correction_screening_planning_package.md at the repo root (gitignored). Its locked decisions (section 2) stay locked unless I reopen them.
- Frames and time scales: SGP4 outputs TEME; convert explicitly to the project's frame and test the conversion. Keep UTC, TT and UT1 straight.
- Element sets are SGP4 mean elements; initializing the numerical propagator from them is a logged, deliberate conversion.
- Splits are by object and by time. Test objects never appear in training; test epochs come after training epochs.
- Contracts between components live in src/orbit_correction/types.py and are checked by tests/test_contracts.py. Change them only when the task says so; otherwise raise it.
- Report errors by altitude band and solar-activity bin. Recall of truth close approaches is the headline screening metric; missed conjunctions are the costly error.

## Working from a handoff
Tasks on this project come as handoff files in `handoffs/`, which exists only on my machine and is gitignored. State doc: `claude/orchestrator_state.md` in the claude.ai Project "Orbit Correction Screening".
- If your prompt names or attaches a handoff, use the orchestrated-handoffs skill in worker mode. If the skill isn't available: read the handoff first, then the files it points to; do only its scope; list decisions under open questions in its Report instead of making them; work on the branch it names; fill in its Report (or put it in your final message if you can't write the file).
- Never commit, quote or link anything from `handoffs/`. Commit messages, PR descriptions and branch names never mention handoffs or task IDs.
- Don't edit other handoffs or `handoffs/INDEX.md`.
