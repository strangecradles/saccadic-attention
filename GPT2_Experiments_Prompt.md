# GPT-2 Saccadic Attention: Multi-Hop & Layer Ablation Experiments

## Overview

We have two key findings from our tiny model (2.7M params, trained from scratch):
1. Multi-hop: 100% accuracy on N-hop tasks with N saccades (N=2,3,4)
2. Context scaling: Does NOT generalize beyond training length

And one key finding from GPT-2 (124M params, pretrained):
1. Context scaling: DOES generalize — 51% accuracy at 4096 context (4x training length)

The hypothesis: GPT-2's pretrained semantic representations enable context-length generalization
that the tiny model's shallow representations cannot. The saccadic MECHANISM is the same —
what differs is the quality of the peripheral map.

## Experiment Set 1: GPT-2 Multi-Hop

Does the N-hop/N-saccade diagonal pattern from the tiny model reproduce on GPT-2?

### Setup
- Base model: GPT-2 124M, frozen
- Saccadic layers: layers 6-11 replaced (following our earlier v1 setup)
- Use the BEST hyperparameters discovered from tiny model experiments:
  - block_size=8 (this was the breakthrough — fine peripheral resolution)
  - window_size=64
  - lr=2e-3 (scale down for GPT-2 — try 5e-4 and 1e-3)
  - supervised_warmup_weight=2.0, warmup_steps=300
  - classification head (predict each digit independently as 10-class)
- Context length: 2048 (clamped position embeddings to 1023)
- Training budget: 10 minutes per run on GH200 (GPT-2 is bigger, needs more time than tiny model)

### Multi-hop passkey task
Same as tiny model: N clues scattered at random positions, each providing one digit of an N-digit code.

### Runs needed
```
N_HOPS=2, N_SACCADES=2   # does the diagonal hold?
N_HOPS=2, N_SACCADES=3   
N_HOPS=3, N_SACCADES=3   # does the diagonal hold?
N_HOPS=3, N_SACCADES=4   
N_HOPS=4, N_SACCADES=4   # does the diagonal hold?
N_HOPS=4, N_SACCADES=5
N_HOPS=5, N_SACCADES=5   # does the diagonal hold?
```

### Implementation
Create `run_gpt2_multihop.py` that:
1. Loads GPT-2 124M from HuggingFace, freezes all params
2. Replaces attention in layers 6-11 with saccadic attention (block_size=8, window=64)
3. Adds classification head for digit prediction
4. Adds supervised warmup targeting passkey positions
5. For each (N_HOPS, N_SACCADES) config: train for 10 min, evaluate on 200 test samples
6. Log: accuracy, avg_clue_distance, training loss trajectory
7. Save results to gpt2_multihop_results.json

CRITICAL: Use the position embedding clamp fix (position_ids clamped to max 1023).

## Experiment Set 2: GPT-2 Context Scaling

Does GPT-2 saccadic attention generalize across context lengths? Re-run with best hyperparameters.

### Setup
- Train ONE model at 2048 context length (10 min budget)
- Evaluate that SINGLE model at: 1024, 2048, 4096, 8192
- Use block_size=8 at all eval lengths (controller sees more blocks at longer contexts)
- Peripheral encoder position embeddings must support up to 8192/8 = 1024 blocks

### Implementation
Create `run_gpt2_context_scaling.py` that:
1. Trains saccadic GPT-2 at 2048 context (single-hop passkey, 10 min)
2. Evaluates the SAME trained model at 1024, 2048, 4096, 8192
3. Reports accuracy and fixation_distance at each length
4. Save results to gpt2_context_scaling_results.json

For 8192 context: the standard attention in layers 0-5 will be expensive. 
Options: (a) just try it and see if GH200's 96GB handles it, 
(b) if OOM, replace layers 0-5 standard attention with local window attention (window=512)

## Experiment Set 3: Layer Ablation — How Many Pretrained Layers Are Needed?

This answers: how much pretrained semantic representation does the saccadic controller need
to generalize across context lengths?

### Setup
Test different splits of standard vs saccadic layers:
- Config A: 0 standard + 12 saccadic (ALL saccadic — like tiny model but with GPT-2 weights)
- Config B: 2 standard + 10 saccadic (layers 0-1 standard, 2-11 saccadic)  
- Config C: 4 standard + 8 saccadic (layers 0-3 standard, 4-11 saccadic)
- Config D: 6 standard + 6 saccadic (layers 0-5 standard, 6-11 saccadic) — our original setup
- Config E: 8 standard + 4 saccadic (layers 0-7 standard, 8-11 saccadic)
- Config F: 10 standard + 2 saccadic (layers 0-9 standard, 10-11 saccadic)

For each config:
1. Train at 2048 context (10 min)
2. Evaluate at 2048 AND 4096 (zero-shot context generalization test)

### What we expect
- Config A (all saccadic): high accuracy at 2048, LOW at 4096 (no pretrained representations — like tiny model)
- Config D (6+6): high at 2048, MODERATE at 4096 (this is what we already saw — 51%)
- Config F (10+2): maybe moderate at 2048 (only 2 saccadic layers), but GOOD generalization to 4096

The sweet spot — minimum pretrained layers needed for context generalization — is the finding.

### Implementation
Create `run_layer_ablation.py` that:
1. For each config (A through F):
   a. Load GPT-2, freeze all params
   b. Replace specified layers with saccadic attention
   c. Train at 2048 for 10 min
   d. Evaluate at 2048 and 4096
2. Save: layer_ablation_results.json with accuracy at both lengths for each config

## Execution Order

Run these SEQUENTIALLY on the GH200 (no GPU contention):

1. GPT-2 multi-hop (7 configs × 10 min = ~70 min)
2. GPT-2 context scaling (1 train + 4 eval = ~15 min)  
3. Layer ablation (6 configs × 10 min = ~60 min)

Total: ~2.5 hours on GH200 (~$6.25)

## What Success Looks Like

### Best case (publishable as main conference paper):
- Multi-hop: N-hop/N-saccade diagonal holds on GPT-2 (100% or near-100% for N=2,3,4)
- Context scaling: GPT-2 saccadic model trained at 2048 achieves >50% at 4096, >0% at 8192
- Layer ablation: clear threshold — X pretrained layers are sufficient for generalization
- Combined with tiny model results: complete story from mechanism validation to deployment

### Minimum viable (publishable as workshop paper):
- Multi-hop diagonal holds for N=2,3 on GPT-2
- Context scaling shows ANY generalization beyond training length
- Layer ablation shows ANY relationship between pretrained depth and generalization
