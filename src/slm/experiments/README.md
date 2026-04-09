# SLM Project Milestones and Timeline

## Current Goal
Develop and validate a strong training recipe for a dense small language model on a fixed dataset and tokenizer, with the stretch goal of approaching SmolLM-class performance under similar small-scale compute.

---

## Milestone 1 — Fix the compute regime
Start from a fixed compute budget range, then use that range to decide the sensible dataset size and model size.

### Output
- fixed compute budget range
- chosen token-to-parameter regime
- candidate model sizes
- candidate training token counts

### Done when
I can clearly answer:
- what compute range I am targeting
- what model sizes fit that compute
- what dataset size / token budget makes sense for that regime

---

## Milestone 2 — Build the dense baseline recipe
Within the fixed compute regime, optimize the highest-impact recipe choices for a dense model.

### Main knobs
- batch size / gradient accumulation
- learning rate
- scheduler
- optimizer

### Output
- one clean dense baseline config
- stable training behavior
- sensible learning curves
- justified defaults for core training settings

### Done when
I have one dense baseline run that I trust technically and can use as the reference recipe.

---

## Milestone 3 — Evaluate the dense baseline
Train and evaluate the dense model properly against the chosen benchmark framing.

### Output
- validation loss
- bits per byte / perplexity
- benchmark comparison against SmolLM / OPT-class expectations
- honest conclusion on competitiveness

### Done when
I have one benchmark-worthy result with a clear comparison story.

---

## Milestone 4 — Stabilize the recipe
Turn the dense baseline recipe into something robust rather than something that only works in one narrow setting.

### Focus
- remove one-off hacks
- keep only settings that reliably help
- identify which hyperparameters are sensitive and which are robust

### Output
- cleaner final recipe
- smaller set of trusted defaults
- understanding of stability-sensitive knobs

### Done when
The recipe feels reliable enough to reuse without rethinking everything from scratch.

---

## Milestone 5 — Make the recipe attention-variant compatible
Use the stabilized dense recipe as the base framework for later attention experiments.

### Goal
Not fully attention-agnostic, but reusable with minimal retuning across variants.

### Output
- shared recipe framework
- list of settings that transfer well
- list of settings that likely need small variant-specific adjustment

### Done when
I can test another attention variant without rebuilding the whole training setup.

---

## Milestone 6 — Systematic sweeps and relationship finding
After the dense baseline is stable, run controlled sweeps to understand how recipe choices and attention variants interact.

### Focus
- a few important hyperparameters
- a few important attention variants
- controlled, comparable experiments

### Output
- systematic sweep results
- trends and correlations
- understanding of which recipe choices transfer across variants

### Done when
I can explain not just what worked, but how the important knobs relate to performance and stability.

---

# Timeline

## Phase 1: April 9–12
### Lock the compute framing
- finalize compute budget range
- decide the candidate model sizes
- decide the candidate token budgets
- define the exact benchmark target and metrics

### End result
A fully locked experimental scope for the dense baseline stage.

---

## Phase 2: April 13–17
### Use scaling to choose the final dense regime
- finish enough of the scaling experiment to guide decisions
- choose final dense model size
- choose final token budget
- narrow the recipe candidates

### End result
A clear decision on what final dense training run should look like.

---

## Phase 3: April 18–22
### Build and run the first serious dense baseline
- train one clean dense baseline
- monitor stability
- inspect learning curves
- identify major recipe weaknesses

### End result
One dense baseline result I trust technically.

---

## Phase 4: April 23–26
### Recipe iteration and stabilization
- test a small number of high-impact recipe changes
- compare outcomes
- choose the best dense recipe
- identify robust defaults

### End result
A stabilized dense training recipe.

---

## Phase 5: April 27–30
### Final benchmark run
- train the benchmark-worthy dense model
- collect final metrics
- compare against SmolLM / OPT-class expectations
- summarize the result clearly

### End result
A finished version 1 benchmark result.

---

## Phase 6: Next phase after April
### Extend into attention-variant experiments
- reuse the stabilized dense recipe
- test selected attention variants
- perform controlled sweeps
- study which recipe choices transfer

### End result
The project moves from baseline-building into more systematic research.

---

# What counts as success

## Minimum success
- fixed compute framing
- one stable dense training recipe
- one full trained model
- usable evaluation metrics

## Good success
- justified dense recipe
- benchmark-worthy model
- fair comparison against SmolLM / OPT-class targets

## Excellent success
- all of the above
- recipe stabilization insights
- clean write-up
- strong foundation for attention-variant experiments

---

# Non-goals for the current phase
These are intentionally out of scope for now:
- MoE
- residual attention as the main project target
- multimodal integration
- broad data-mixture studies
- full scaling-law reproduction
- trying to match frontier 1B SOTA immediately

These can come after the dense baseline recipe is proven.

---

# Simple summary
This project has two layers:

## Layer 1 — Current focus
Fix a compute regime, choose the model/data scale, build a strong dense training recipe, and benchmark it.

## Layer 2 — Next phase
Stabilize the recipe enough that it can support systematic attention-variant experiments.