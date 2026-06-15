# AGENTS.md

Stable rules for agents working in this repository.

## Purpose

This repo contains the cleaned generative-decoder workflow and an archived
syndrome-only mutual-information (MI) scaling pipeline for toric-code
experiments. The archived syndrome-only pipeline now lives under
`syndrome_only_mi/`; top-level `decoding/` and `scripts/` no longer carry
syndrome-only compatibility wrappers.

Keep this file stable. Put active run ids, seed blocks, current commands, and
unfinished decisions in `docs/NEXT_STEPS.md`. Put completed syndrome-only run
reports in `syndrome_only_mi/docs/agent_outputs/scaling_runs/`.

## Read First

Before substantial work, read only the smallest relevant set:

- `README.md`: overview and common workflows.
- `docs/NEXT_STEPS.md`: current unfinished plan.
- `syndrome_only_mi/docs/SEED_POLICY.md`: seed inclusion/replacement/failure policy.
- `syndrome_only_mi/docs/STABILITY_CHECKLIST.md`: stability gates.
- `syndrome_only_mi/docs/MI_FIT_SUMMARY.md` and
  `syndrome_only_mi/docs/MI_FIT_ANALYSIS.md`: archived fit state.
- Latest relevant syndrome-only report in
  `syndrome_only_mi/docs/agent_outputs/scaling_runs/`.

If docs disagree, prefer machine-readable artifacts and the most recent run
report, then update docs rather than silently choosing one version.

## Repository Map

- `module/`: code objects, GF(2), error models, autoregressive models.
- `decoding/`: code generation, legacy decoder evaluation, and baselines.
- `gnd/`: current GND datasets, training, decoder evaluation, cut-MI, and
  `n_d^min(L)` tooling.
- `syndrome_only_mi/`: archived syndrome-only datasets, MI training/evaluation,
  scaling analysis, audits, scripts, and historical fit docs.
- `scripts/`: GPU wrapper and current experiment helpers.
- `docs/`: current GND plans and notes.
- `net/`: current generated datasets/checkpoints/results/plots. Do not commit heavy
  artifacts here.
- `logs/`: local run logs, ignored by Git.

## Experiment Workflow

1. Read `docs/NEXT_STEPS.md`.
2. Run required audits before formal MI work.
3. Use a fresh output directory/run id unless explicitly told otherwise.
4. Pass important configuration through CLI flags or environment variables.
5. Preserve JSON records next to generated artifacts.
6. Copy conclusions into `docs/`; do not rely on shell history or logs.
7. After completion, update `docs/NEXT_STEPS.md` so it contains only unfinished
   work.

Prefer existing scripts, CLIs, and record formats over ad hoc tooling.

## GND Outline MI Invariants

`outline.md` is the source of truth for current GND MI work. It asks for MI
under the true distribution

```text
p(beta, gamma) = p(beta_1, beta_2, gamma_1, gamma_2)
```

and then for neural models that learn the same distribution for
`n_d^min(L)` analysis.

The required cuts are variable cuts in the GND representation:

- `middle`: `I(beta : gamma)`.
- `quarter`: `I(beta_1 : beta_2, gamma)`, with `beta_1` on side A and
  `(beta_2, gamma)` on side B.
- `three_quarter`: `I(beta, gamma_1 : gamma_2)`, with `(beta, gamma_1)` on
  side A and `gamma_2` on side B.

Do not reinterpret these as physical real-space cuts of qubits, edges,
plaquettes, or regions. Do not reinterpret them as the archived syndrome-only
AB/BA spatial cuts. Do not condition on `syndrome = 0` unless the user
explicitly asks for a separate diagnostic with that condition. For true
distribution baselines, the target object is `p(beta, gamma)` at the selected
physical error rate and error model.

## Protocol-Fixed MI Comparisons

Treat each full training recipe as a named protocol track. A protocol includes
architecture, dataset size, optimizer, learning rate, batch size, weight decay,
gradient clipping, warmup, LR scheduler, early stopping, epoch budget, MI
evaluation settings, seeds, partition, error model, physical error rate, and
code identifiers.

For scaling fits, compare MI values across `L` only within the same protocol
track. If any training detail changes, record the result as a diagnostic row
under a new protocol track until same-protocol anchors show it can be promoted.
Do not silently mix endpoint-only optimizer, architecture, data-size, or
scheduler changes into the recommended fit.

When a new protocol appears promising at one `L`, first test it on a nearby
anchor `L` before promotion. If the anchor matches the current protocol within
train-seed spread and has no objective failures, the new protocol can be
considered for an explicit endpoint policy. If the anchor shifts materially,
build a complete same-protocol subset across the relevant `L` values instead
of mixing protocols.

For one fixed `L`, choose among multiple MI aggregates by this priority:

- Prefer the row from the currently declared recommended protocol.
- Exclude only seeds with objective saved-JSON training failures, following
  `syndrome_only_mi/docs/SEED_POLICY.md`; do not exclude clean high-MI or
  low-MI seeds by value.
- Prefer clean same-protocol 8-seed aggregates over pilots, replacements, or
  mixed-protocol diagnostics.
- Treat lower bootstrap error as secondary to train-seed stability and
  objective training health.
- Keep alternative architectures, data sizes, replacement-seed aggregates, and
  optimizer variants as diagnostic unless the fit docs explicitly promote a
  new protocol policy.

## Required Checks

Before formal syndrome-only MI work:

```bash
syndrome_only_mi/scripts/run_mi_agent_audits.sh
```

Required marker:

```text
MI_AGENT_AUDITS_PASSED
```

For GPU/CUDA checks:

```bash
scripts/run_codex_gpu.sh "scripts/check_gpu_env.sh"
```

`scripts/run_codex_gpu.sh` starts nested `codex exec`. Use it for short
foreground checks, not long detached training. Do not run
`scripts/run_codex_gpu.sh ... &`, and do not put it inside `tmux`/`screen`.
For long GPU jobs, verify GPU/CUDA first, then run the actual project script
directly in a persistent shell session such as `tmux`.

For code changes, run the smallest relevant smoke tests. For syndrome-only MI
logic changes, always run `syndrome_only_mi/scripts/run_mi_agent_audits.sh`.
For GND outline MI logic changes, test the `gnd.*` path being changed; do not
substitute syndrome-only audits for GND cut-MI validation.

## Syndrome-Only MI Invariants

The workflow estimates:

```text
I_q(A;B) = H_q(A) + H_q(B) - H_q(A,B)
```

Required assumptions:

- Matching `AB` and `BA` datasets/checkpoints are required.
- `AB` prefix estimates `H(A)`.
- `BA` prefix estimates `H(B)`.
- Full-sequence likelihood estimates `H(A,B)`.
- Partition metadata, error model, physical error rate, code identifiers, and
  seeds must match across paired artifacts.

When changing this logic, update tests/audits for ordering, entropy
decomposition, and artifact compatibility.

## Artifacts

Typical run root:

- `<run_root>/datasets/*.pt`
- `<run_root>/models/*.pt`
- `<run_root>/models/records/*.json`
- `<run_root>/results/*.json`
- `<run_root>/L*_tseed*/mi_vs_L.{json,csv,png}`
- `<run_root>/L*_tseed*/sweep_manifest.json`

Do not commit heavy `net/` artifacts, checkpoints, datasets, plots, or logs.
Lightweight docs/reports should remain visible for review.

## Long Training Runs

Do not continuously monitor after successful launch. Required launch checks:

- audits and GPU/CUDA checks passed;
- run id, output dir, command, and log path verified;
- process/session started;
- canonical log exists and shows the project training script, not a nested
  Codex prompt;
- intended run directory exists if artifacts have started.

Detached launch policy:

- Prefer `tmux` for long GPU jobs.
- In `tmux`, run the actual project script directly, for example:

```bash
tmux new-session -d -s <session> \
  'cd /path/to/repo &&
   env BASE_ROOT=syndrome_only_mi/net/mi_scaling/<run_id> ... \
     syndrome_only_mi/scripts/run_made_mi_ntrain400k.sh \
   > logs/<run_id>.log 2>&1'
```

- Do not use bare `&` for long jobs; the process may die when the agent shell
  exits.
- Do not use `tmux ... scripts/run_codex_gpu.sh "..."`; it can start a nested
  Codex agent instead of training.
- Before relaunching, check existing session/process/log/run directory to avoid
  duplicate writers.
- Preserve failed launch logs under diagnostic names, then free
  `logs/<run_id>.log` for the real run.
- Do not delete or overwrite partial artifacts unless the user approves.

After launch checks, wait for completion before extraction, aggregation,
reporting, fit updates, or next-step planning. Poll only when the user asks or
there is a clear failure signal.

## Seed And Failure Policy

Follow `syndrome_only_mi/docs/SEED_POLICY.md`.

A seed is a training failure only if saved training JSON shows objective
late-epoch failure, e.g. late train/validation NLL `>= 1e3`. Do not infer
failure from MI value alone. Clean high-MI or low-MI seeds must be kept.
Replacement seeds must follow a predeclared order, not observed MI.

For each fixed-`L` aggregate, report:

- seed list and count;
- mean MI;
- sample standard deviation across train seeds;
- `cv = seed_std / mean`;
- min/max MI;
- mean bootstrap std;
- excluded seeds and objective reasons.

Use sample standard deviation for `seed_std`.

## Fit Records

Recommended fit inputs are rows in `syndrome_only_mi/docs/MI_FIT_POINTS.csv` with
`include_in_recommended_fit=yes`.

Rules:

- Keep historical and diagnostic rows.
- Keep only one recommended row per `L`.
- Diagnostic architecture, dataset-size, replacement, and sensitivity rows
  default to `include_in_recommended_fit=no`.
- Any changed recommended row requires updates to
  `syndrome_only_mi/docs/MI_FIT_SUMMARY.md` and
  `syndrome_only_mi/docs/MI_FIT_ANALYSIS.md`.
- Do not silently substitute diagnostic rows into the recommended fit.

Validate the CSV after edits.

## Documentation After Runs

After each completed formal or diagnostic run:

1. Add a report under `syndrome_only_mi/docs/agent_outputs/scaling_runs/`.
2. Add diagnostic/recommended rows to `syndrome_only_mi/docs/MI_FIT_POINTS.csv`
   if relevant.
3. Update `syndrome_only_mi/docs/MI_FIT_SUMMARY.md` if fit record or
   interpretation changes.
4. Update `syndrome_only_mi/docs/MI_FIT_ANALYSIS.md` if sensitivity/interpretation changes.
5. Update `docs/NEXT_STEPS.md`.

Run reports should include command, artifact paths, per-seed MI/bootstrap std,
AB/BA best epoch, AB/BA test NLL, late-NLL max/failure flags, aggregate mean,
seed std, cv, min/max, mean bootstrap std, interpretation, and next decision.

## Stability Guidance

Use `syndrome_only_mi/docs/STABILITY_CHECKLIST.md` for detailed policy.

- Evaluation is stable when bootstrap std is much smaller than train-seed
  spread.
- A usable baseline needs enough independent train seeds, no objective training
  failures, and no obvious mean drift after adding seeds.
- Treat architecture, dataset-size, and replacement-seed comparisons as
  diagnostics unless fit docs explicitly promote them.
- Mark provisional endpoints as provisional until stability is documented.

## Git And Editing

- Worktree may be dirty. Do not revert user/previous-agent changes.
- Use `apply_patch` for manual edits.
- Prefer `rg` and `rg --files`.
- Keep generated experiment outputs under a new run id unless told otherwise.
- Before finalizing edits, run:

```bash
git diff --check
```

Run the current planned experiment only after reading `docs/NEXT_STEPS.md`.
Keep exact commands and run ids in `docs/NEXT_STEPS.md` or run reports, not in
this stable guide.
