# Saccadic Attention

## What this project is

A novel attention mechanism for transformers inspired by human saccadic eye movements and foveated vision. Instead of attending to all tokens uniformly (quadratic cost) or using static sparse patterns, the model learns a sequential fixation policy that dynamically decides where in the context to apply high-resolution attention, while maintaining O(n) peripheral awareness of the full context.

## Architecture (three components)

1. **Peripheral Encoder**: Learned weighted pooling over blocks of tokens → compressed global map. O(n).
2. **Foveal Processor**: Standard multi-head attention over a small window (k tokens) centered at the fixation point. O(k²).
3. **Saccadic Controller**: Cross-attention between accumulated state and peripheral map → selects next fixation point via Gumbel-softmax.

Forward pass is iterative: peripheral_map → for t in range(num_saccades): choose fixation → process window → update state → output.

## Base model

GPT-2 124M from HuggingFace. Original attention replaced in selected layers. All original GPT-2 params frozen; only saccadic components are trainable.

## Key files

- `src/peripheral_encoder.py` — Component 1
- `src/foveal_processor.py` — Component 2
- `src/saccadic_controller.py` — Component 3
- `src/saccadic_attention.py` — Combined module with iterative loop
- `src/gpt2_saccadic.py` — GPT-2 integration
- `src/data.py` — Passkey retrieval dataset + PG19 loader
- `train.py` — Training script
- `evaluate.py` — Benchmarking

## Implementation rules

- PyTorch only (no JAX). Use HuggingFace transformers for GPT-2.
- Use einops for tensor reshaping where it improves clarity.
- Every module must have shape comments on all tensor operations.
- Write tests for each component before integrating.
- Commit after each logical step with a descriptive message.
- All hyperparameters go in YAML configs, not hardcoded.

## Current defaults

- num_saccades: 5
- window_size (k): 128
- block_size (B): 32
- Gumbel temperature: 1.0 → 0.1 (annealed)
- Initial state: mean of peripheral map
- Optimizer: AdamW, lr=1e-4, cosine schedule

## What NOT to do

- Do NOT pretrain from scratch. We fine-tune from pretrained GPT-2.
- Do NOT use Flash Attention or custom CUDA kernels. Keep it pure PyTorch for clarity.
- Do NOT make the foveal processor anything other than standard multi-head attention. The novelty is in the CONTROLLER, not the attention computation itself.
- Do NOT skip tests. Each component must be tested independently before integration.
