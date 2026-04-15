training log with wandb artifact will log simplistic config snapshot and wandb run idx if it is sweep wandb sweep id



## Lock the fixed-compute framing

### 1. Finalize the compute budget for the dense baseline stage
Choose one fixed compute budget that is realistic for the training setup and large enough to avoid the architecture collapse observed at the low end of the scaling sweep.

### 2. Use a small scaling-law sweep to estimate the token-to-parameter tradeoff
Run a small frontier-style study across a few compute budgets to identify a sensible \(D:N\) relationship for this setup, rather than trying to fully re-derive a scaling law from scratch.

### 3. Use the estimated \(D:N\) relationship to set the final dense baseline allocation
Apply the rule from Step 2 to the chosen fixed compute budget to determine:
- the target parameter count \(N\)
- the target token budget \(D\)

### 4. Resolve the final candidate model shape
Convert the target parameter count into a concrete dense architecture by selecting the final depth and width configuration that best matches the target while remaining valid for the implementation.

### 5. Lock the final training token budget
Set the dataset size and token budget from the chosen \(D:N\) allocation so the model and data distribution are consistent with the fixed-compute target.

### 6. Define the benchmark target and evaluation metrics
Set the benchmark goal for the dense baseline stage so it can later be compared against SmolLM-style baselines, using metrics that are fair, reproducible, and easy to interpret.

## End result

A fully locked experimental scope for the dense baseline stage, including:
- one fixed compute budget
- one final dense architecture
- one final token budget
- one benchmark target
- one evaluation protocol

for future referecne 
Stoachstic Lanczos Quadrature methods 



# TODO

## Training Stability
- [ ] Add continuous training / resume from checkpoint
  - [ ] Save periodic checkpoints during training
  - [ ] Restore model, optimizer, scheduler, scaler, and train state
  - [ ] Resume cleanly after Slurm interruption or timeout
  - [ ] Keep both `last` and `best` checkpoints

## Infrastructure
- [ ] Fix DVC pull / DVC setup on M3
  - [ ] Ensure packages install into the real `slm_ven` Conda environment
  - [ ] Install `dvc[s3]` inside the working env
  - [ ] Verify `dvc remote list`
  - [ ] Verify `dvc pull`
  - [ ] Verify S3 auth / credentials on M3
  - [ ] Clean up old user-site installs after current jobs finish

## Analysis / Diagnostics
- [ ] Add Hessian analysis tool
  - [ ] Decide whether to track top eigenvalue, trace, or spectral density approximation
  - [ ] Make it usable during training or as a post-training diagnostic
  - [ ] Use it to inspect training stability and curvature

## Scaling Law Workflow
- [ ] Optimize scaling-law pipeline
  - [ ] Clean up compute budget → model size → token budget workflow
  - [ ] Reduce duplicate candidate architectures
  - [ ] Make sweeps easier to generate and compare
  - [ ] Improve reporting of theoretical vs actual training setup

## Hyperparameter Tuning
- [ ] Optimize learning rate, batch size, scheduler, and optimizer
  - [ ] Tune learning rate range
  - [ ] Tune effective batch size / gradient accumulation
  - [ ] Compare scheduler choices
  - [ ] Tune optimizer settings
    - [ ] AdamW betas
    - [ ] epsilon
    - [ ] weight decay
  - [ ] Check stability across different model sizes