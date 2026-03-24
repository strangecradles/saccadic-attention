# Overnight Experiment: Saccadic Attention on Modern Small Models with RULER-Style Evaluation

## Summary

Run saccadic attention on Qwen2.5-1.5B using our best architecture (bottlenecked additive),
evaluated on RULER-style tasks at multiple context lengths. This is the experiment that
produces the paper's key comparison figure: small model + saccadic attention vs. small model
alone at increasing context lengths.

## The Architecture: Best Design from All Experiments

Based on everything we've learned tonight:

```
┌─────────────────────────────────────────────────────────┐
│ Qwen2.5-1.5B (ALL layers FROZEN, standard attention)    │
│ ~28 transformer layers, hidden_dim=1536                 │
│ Processes full context with standard attention           │
│ Provides rich semantic representations                   │
└──────────────────────────┬──────────────────────────────┘
                           │ Layer outputs (1536-dim)
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Projection: 1536 → 128 (trainable, ~200K params)        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ SACCADIC MODULE (all trainable, ~2-3M params total)     │
│                                                          │
│ Peripheral Encoder:                                      │
│   - Input: projected Qwen outputs (128-dim)             │
│   - Block size: 8 tokens per block                       │
│   - Features: mean + std + max_pool per block           │
│   - Project concatenated features → 128-dim              │
│                                                          │
│ For each saccade (num_saccades configurable):           │
│   1. Controller: cross-attention(state, peripheral_map) │
│      → Gumbel-softmax → selects fixation block          │
│   2. Foveal processor: full attention over 64-token     │
│      window at fixation point                            │
│   3. Accumulated foveal context: previous windows        │
│      stay in attention (working memory grows)            │
│   4. Global peripheral map update: cross-attention      │
│      from all blocks to foveal output + residual        │
│                                                          │
│ Supervised warmup: auxiliary loss on fixation targeting   │
│   warmup_weight=2.0, warmup_steps=300, annealed         │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Projection: 128 → 1536 (trainable, ~200K params)        │
│ + Task-specific head (depends on evaluation task)        │
└─────────────────────────────────────────────────────────┘
```

### Key hyperparameters (from tiny model experiments):
- block_size = 8 (CRITICAL — this was the T15 breakthrough)
- window_size = 64
- num_saccades = 3 (default, adjust per task)
- lr = 2e-3 for saccadic params (may need tuning for Qwen, start with 1e-3)
- supervised_warmup_weight = 2.0
- supervised_warmup_steps = 300
- All Qwen params FROZEN, only saccadic module + projections trainable

### Why Qwen2.5-1.5B:
- 1.5B params, modern training (2024/2025), vastly better representations than GPT-2
- Hidden dim 1536, ~28 layers
- Native 32K+ context support (but we test beyond that with saccadic)
- Apache 2.0 license
- Well-studied, published RULER numbers exist for comparison
- Fits on GH200 (96GB) with room to spare (~3GB in fp16)
- HuggingFace: `Qwen/Qwen2.5-1.5B`

## Experiment Set 1: RULER-Style Tasks (implement ourselves)

RULER's official pipeline is Docker-based and complex. Instead, implement the core tasks
directly. RULER has 4 categories with 13 tasks. We implement the 7 most relevant:

### Category 1: Retrieval (saccadic should EXCEL)

**Task 1a: Single Needle (S-NIAH)**
- Insert one key-value pair in Paul Graham-style essay text
- Format: "The special magic {city} number is: {7-digit number}"
- Query at end: "What is the special magic {city} number?"
- Metric: exact match of 7-digit number
- This is basically our passkey test but with RULER's exact format

**Task 1b: Multi-Key Needle (MK-NIAH)**
- Insert 1 target needle + 3 distractor needles (different cities, different numbers)
- Query asks for ONE specific city's number
- Tests: can the controller find the RIGHT needle among distractors?

**Task 1c: Multi-Value Needle (MV-NIAH)**
- Insert 4 needles that ALL match the query key
- Query asks for ALL values associated with one city
- Tests: can multi-saccade find ALL relevant needles?
- Use num_saccades=4+ for this task

### Category 2: Multi-Hop Tracing (saccadic should EXCEL)

**Task 2a: Variable Tracking (VT)**
- Insert a chain of variable bindings: "X1 = Y1", "Y1 = Z1", "Z1 = W1"
- Scattered across the context with filler between them
- Query: "What is the value of X1?"
- Answer requires following the chain: X1 → Y1 → Z1 → W1
- Test with chain lengths 2, 3, 4 (matching num_saccades)
- This is exactly our multi-hop passkey test in RULER's format

### Category 3: Aggregation (saccadic will STRUGGLE — show honestly)

**Task 3a: Common Words Extraction (CWE)**
- Context filled with words sampled from a Zipf distribution
- Query: "What are the 10 most common words in the text?"
- This requires scanning the ENTIRE context — saccadic can't do this
- Include as honest limitation

**Task 3b: Frequency Estimation (FWE)**
- Same as CWE but with different alpha parameter
- Harder variant — less separation between frequent and rare words

### Category 4: Question Answering (genuinely uncertain)

**Task 4a: Single-hop QA (SQuAD-in-haystack)**
- Take a SQuAD question-passage-answer triple
- Embed the passage in a long context of distractor passages
- Query with the SQuAD question
- Tests: can saccadic attention find the relevant passage AND extract the answer?
- This is the REAL test of whether saccadic attention works beyond synthetic tasks

## Experiment Set 2: Context Length Scaling

For EACH task above, evaluate at context lengths: 4K, 8K, 16K, 32K

For the saccadic model:
- Early Qwen layers (0-13): use LOCAL window attention (window=512 tokens)
  to avoid OOM at long contexts. These layers handle local context processing.
- Late Qwen layers (14-27): same local window attention
- Saccadic module on top: handles long-range dependencies
- This means the Qwen backbone cost is O(n × 512) = linear
- Saccadic cost is O(S × 64²) = constant = 12,288 ops regardless of context

For the baseline (vanilla Qwen):
- Run Qwen2.5-1.5B with standard full attention at each context length
- No saccadic module, no modifications
- This gives us the "without saccadic" comparison line

## Implementation Plan

### File 1: `saccadic_qwen.py` — The model

```python
# Pseudocode structure

class SaccadicAdapter(nn.Module):
    """Bottlenecked saccadic attention module that sits on top of any pretrained LM."""
    
    def __init__(self, base_dim=1536, saccadic_dim=128, block_size=8, 
                 window_size=64, num_saccades=3, max_blocks=4096):
        # Projection layers
        self.proj_down = nn.Linear(base_dim, saccadic_dim)  # 1536 → 128
        self.proj_up = nn.Linear(saccadic_dim, base_dim)     # 128 → 1536
        
        # Peripheral encoder
        self.peripheral_encoder = PeripheralEncoder(
            dim=saccadic_dim, block_size=block_size)  # mean+std+max_pool
        
        # Saccadic layers
        self.controller = SaccadicController(dim=saccadic_dim)
        self.foveal_processor = FovealProcessor(
            dim=saccadic_dim, n_heads=4, window_size=window_size)
        
        # Global peripheral map update (cross-attention)
        self.peripheral_updater = nn.MultiheadAttention(
            saccadic_dim, num_heads=4, batch_first=True)
        
        # Classification/extraction head (task-specific)
        self.output_head = None  # set per task
    
    def forward(self, base_model_output, num_saccades=3):
        # Project down
        h = self.proj_down(base_model_output)  # (B, seq_len, 128)
        
        # Build peripheral map
        peripheral_map = self.peripheral_encoder(h)  # (B, num_blocks, 128)
        
        state = initial_state(h)
        accumulated_windows = []
        
        for t in range(num_saccades):
            # Controller picks fixation
            fixation = self.controller(peripheral_map, state)
            
            # Foveal processing with accumulated context
            window = extract_window(h, fixation, self.window_size)
            accumulated_windows.append(window)
            all_context = torch.cat(accumulated_windows, dim=1)
            foveal_out = self.foveal_processor(all_context)
            
            # Update state
            state = foveal_out[:, -self.window_size:, :].mean(dim=1)
            
            # Global peripheral map update
            peripheral_map = peripheral_map + self.peripheral_updater(
                peripheral_map, foveal_out, foveal_out)[0]
        
        # Project back up and output
        output = self.proj_up(state)
        return self.output_head(output)


class SaccadicQwen(nn.Module):
    """Qwen2.5-1.5B with saccadic adapter on top."""
    
    def __init__(self):
        self.qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
        
        # Freeze ALL Qwen parameters
        for param in self.qwen.parameters():
            param.requires_grad = False
        
        # Add saccadic adapter
        self.saccadic = SaccadicAdapter(
            base_dim=self.qwen.config.hidden_size,  # 1536
            saccadic_dim=128,
            block_size=8,
            window_size=64,
            num_saccades=3
        )
    
    def forward(self, input_ids, num_saccades=3):
        # Get Qwen's output (frozen, rich representations)
        with torch.no_grad():
            outputs = self.qwen(input_ids, output_hidden_states=True)
            # Use last hidden state
            hidden = outputs.hidden_states[-1]
        
        # Saccadic processing
        return self.saccadic(hidden, num_saccades)
```

### File 2: `ruler_tasks.py` — RULER task implementations

Each task generates (input_tokens, query, answer) triples:
- S-NIAH: single needle in Paul Graham essays
- MK-NIAH: one target + 3 distractor needles
- MV-NIAH: 4 needles matching same key
- VT: variable tracking chains (2-hop, 3-hop, 4-hop)
- CWE: common words extraction
- SQuAD-in-haystack: real QA in long context

Each task has a generate(context_length, n_samples) function that produces
evaluation data at the specified context length.

### File 3: `run_ruler_overnight.py` — Orchestration

```python
# For each task:
#   For each context length [4K, 8K, 16K, 32K]:
#     1. Train saccadic adapter on that task at that length (until convergence)
#     2. Evaluate saccadic model (200 test samples)
#     3. Evaluate baseline Qwen (same 200 test samples, no saccadic)
#     4. Log: task, context_length, saccadic_accuracy, baseline_accuracy, 
#             saccadic_time, baseline_time, fixation_distances
#
# Save all results to ruler_results.json
# Print summary table at the end
```

### Training protocol per task:
- Train saccadic adapter from scratch for each (task, context_length) config
- Convergence: stop when val accuracy doesn't improve for 1000 steps, or 30 min max
- Adam optimizer, lr=1e-3, cosine decay
- Supervised warmup for fixation targeting: weight=2.0, steps=300
- Batch size: 4 (adjust if OOM)
- Evaluation: 200 test samples per config

### For the retrieval/multi-hop tasks:
- Task head: classification (predict answer tokens independently)
- num_saccades: 3 for single-hop, N for N-hop tasks

### For the aggregation tasks:
- Task head: generate top-K words
- num_saccades: 10 (maximum coverage — still won't be enough, but honest attempt)
- We EXPECT poor performance here — that's the point

### For QA tasks:
- Task head: extractive span selection (start/end position within foveal window)
- num_saccades: 3 (find relevant passage, read it, extract answer)

## Execution Plan

1. Install Qwen: `pip install transformers accelerate --break-system-packages`
2. Download model: will auto-download on first use from HuggingFace
3. Run sequentially on GH200:
   - S-NIAH at 4K, 8K, 16K, 32K (~2 hours)
   - MK-NIAH at 4K, 8K, 16K, 32K (~2 hours)  
   - VT (2,3,4-hop) at 4K, 8K, 16K (~2 hours)
   - CWE at 4K, 8K (~1 hour — we know this will be bad)
   - Baseline Qwen evaluation on all above (~1 hour)
   
   Total: ~8 hours overnight

4. Save all results, commit to GitHub
5. Generate summary table and comparison plots

## What Success Looks Like

### The headline figure:
```
Accuracy vs Context Length

100% ─  ●───●───●───●  Saccadic Qwen (retrieval tasks)
        │             
 80% ─  │   ●───●     Baseline Qwen (retrieval tasks)
        │       │ ╲    
 60% ─  │       │   ╲   
        │       │    ╲  Baseline degrades
 40% ─  │       │     ╲
        │       │      ●
 20% ─  │       │       
        │       │       
  0% ───┴───────┴───────
       4K    8K   16K  32K   Context Length
```

If saccadic Qwen maintains high accuracy where baseline Qwen degrades,
that's the paper's central result.

### The honest limitation figure:
```
Accuracy on Aggregation Tasks

100% ─  
        ●───●───●       Baseline Qwen (sees everything)
 80% ─  │             
        │             
 60% ─  │             
        │             
 40% ─  │             
        ●───●───●       Saccadic Qwen (can't see everything)
 20% ─  │       │       
        │       │       
  0% ───┴───────┴───────
       4K    8K   16K   Context Length
```

Showing where saccadic attention fails is as important as showing where it succeeds.

## Critical Notes

- Use `torch.float16` for Qwen inference to save VRAM
- For contexts > 8K, Qwen's standard attention will be expensive.
  If OOM, use `torch.utils.checkpoint` for gradient checkpointing on Qwen,
  or run baseline eval in smaller batches (batch_size=1)
- For contexts > 16K, baseline Qwen may OOM even for evaluation.
  That's actually a GOOD result for us — "baseline can't even run at this length,
  saccadic model handles it fine"
- Save checkpoints after each task/length combo so we can resume if interrupted
- Log VRAM usage and wall-clock time — the efficiency comparison is a secondary result
- Paul Graham essays for haystack: download from RULER's GitHub or generate
  similar quality filler text

## After Overnight Run

With the results:
1. Generate comparison table (saccadic vs baseline per task per length)
2. Plot accuracy vs context length curves
3. Plot FLOPs comparison (constant saccadic vs quadratic full attention)  
4. Write the paper's results section
5. Update IDEAS.md with findings
