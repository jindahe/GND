# AGENTS.md

## Purpose

This file defines the working contract for coding agents in this repository.
The project studies mutual-information and conditional-mutual-information scaling
for dephased 2D toric code states using tensor-network methods.

## Scope

Agents are expected to work on:

- Numerical simulation code for `H(m)`, `CMI`, and related observables.
- Tensor-network contraction backends (exact TN and bMPS).
- Data generation scripts (`.csv`) and plotting scripts.
- Documentation for physics assumptions, algorithms, and experiment status.

## Repository Map

- `CMI_calculation.py`: main CMI estimator and geometry pipeline.
- `bMPS_contraction.py`: boundary-MPS contraction backend.
- `H_calculation.py`: entropy `H(m)` estimator.
- `plot_CMI_vs_p.py`, `plot_CMI_bMPS.py`: plotting utilities.
- `Syndrome_pure_error.py`: syndrome/pure-error checks.
- `PROJECT.md`: project-level technical summary.
- `PROBLEM.md`: known issues, bottlenecks, and current status.
- `PLAN.md`: prioritized roadmap and implementation checklist.
- `PHYSICS.md`: physics formulas and interpretation notes.
- `paper.md`: paper excerpts and appendix references.

## Execution Rules

- Prefer minimal, targeted edits that preserve current experiment behavior.
- Do not change physics conventions silently (geometry, parity definitions, sign conventions).
- When changing numerics, document expected impact on:
  - runtime per sample,
  - approximation error (vs exact TN when available),
  - Monte Carlo variance.
- Keep interfaces stable unless explicitly asked to refactor.
- Avoid adding compatibility layers unless requested.

## Validation Requirements

Before finishing substantial code changes, run relevant checks when feasible:

- Small-size sanity runs (for example small `L`, small sample count).
- If contraction logic changed, compare against exact TN on tractable sizes.
- If randomness is involved, use fixed seeds for regression checks.

If full validation is not feasible, clearly state what was not run and why.

## Communication Protocol

- Keep responses concise and direct.
- For complex changes, present plan first, then execute.
- If a design choice is ambiguous and can affect scientific conclusions,
  ask for confirmation before proceeding.
- Proactively suggest better paths when they reduce risk or compute cost.

## Editing Boundaries

- Do not revert unrelated local changes.
- Do not perform destructive git/file operations unless explicitly requested.
- Keep new files and comments concise and purposeful.

## Coding Style

- Follow existing style in each file.
- Use clear function names tied to physics meaning.
- Add short comments only where tensor index logic or parity logic is non-obvious.
- Prefer deterministic outputs for scripts intended for comparison.

## Experiment Notes Convention

When updating docs (`PROJECT.md`, `PROBLEM.md`, `PLAN.md`), include:

- exact parameter set (`L`, `p`, `r`, `num_samples`, `max_bond`),
- hardware/runtime context if timing is reported,
- whether values are exact TN, bMPS approximation, or MC estimate.

## Hand-off Checklist

At task completion, report:

- files changed,
- behavioral impact,
- validation performed,
- remaining risks or follow-up items.
