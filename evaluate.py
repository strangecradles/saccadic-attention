"""Evaluation script: passkey retrieval, perplexity, ablations, fixation visualization."""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from src.data import PasskeyRetrievalDataset, PG19Dataset
from src.gpt2_saccadic import GPT2Saccadic
from train import collate_fn


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> GPT2Saccadic:
    """Load a trained saccadic GPT-2 model from checkpoint."""
    sc = config['saccadic']
    model = GPT2Saccadic(
        model_name=config['model']['name'],
        saccadic_layers=config['model']['saccadic_layers'],
        num_saccades=sc['num_saccades'],
        window_size=sc['window_size'],
        block_size=sc['block_size'],
        gumbel_temperature=config['gumbel']['temp_end'],  # use final temperature
        mask_fixated=sc['mask_fixated'],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Load only saccadic block weights (rest are pretrained GPT-2)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


# ---------- Passkey Retrieval Evaluation ----------

def evaluate_passkey(
    model: GPT2Saccadic,
    context_length: int,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    num_samples: int = 500,
    batch_size: int = 4,
) -> dict:
    """Evaluate passkey retrieval at a specific context length.

    Returns:
        dict with accuracy, fixation distances, example fixation patterns
    """
    dataset = PasskeyRetrievalDataset(
        num_samples=num_samples,
        context_length=context_length,
        tokenizer=tokenizer,
        seed=999,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    correct = 0
    total = 0
    fixation_distances = []  # distance from fixation to passkey position
    example_fixations = []   # save some examples for visualization

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Passkey @{context_length}'):
            input_ids = batch['input_ids'].to(device)
            passkeys = batch['passkey']
            passkey_positions = batch['passkey_position']

            outputs = model(input_ids)
            logits = outputs['logits']

            # Collect fixation info
            for layer_idx, info in outputs['fixation_info'].items():
                for fp in info['fixation_points']:
                    for i in range(input_ids.shape[0]):
                        dist = abs(fp[i].item() - passkey_positions[i])
                        fixation_distances.append(dist)

            # Check accuracy
            for i in range(input_ids.shape[0]):
                labels = batch['labels'][i]
                answer_positions = (labels != -100).nonzero(as_tuple=True)[0]
                if len(answer_positions) == 0:
                    continue
                pred_ids = logits[i, answer_positions - 1].argmax(dim=-1)
                pred_text = tokenizer.decode(pred_ids).strip()
                if passkeys[i] in pred_text:
                    correct += 1
                total += 1

            # Save first few examples for visualization
            if len(example_fixations) < 5:
                for layer_idx, info in outputs['fixation_info'].items():
                    example_fixations.append({
                        'layer': layer_idx,
                        'fixation_points': [fp[0].item() for fp in info['fixation_points']],
                        'passkey_position': passkey_positions[0],
                        'context_length': context_length,
                    })

    return {
        'accuracy': correct / max(total, 1),
        'total': total,
        'correct': correct,
        'mean_fixation_distance': np.mean(fixation_distances) if fixation_distances else 0,
        'example_fixations': example_fixations[:10],
    }


def evaluate_passkey_sweep(
    model: GPT2Saccadic,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    context_lengths: list[int],
    num_samples: int = 500,
) -> dict:
    """Run passkey retrieval across multiple context lengths."""
    results = {}
    for ctx_len in context_lengths:
        print(f'\nEvaluating passkey retrieval at context length {ctx_len}...')
        results[ctx_len] = evaluate_passkey(
            model, ctx_len, tokenizer, device, num_samples=num_samples,
        )
        print(f'  Accuracy: {results[ctx_len]["accuracy"]:.4f}')
    return results


# ---------- Perplexity Evaluation ----------

def evaluate_perplexity(
    model: GPT2Saccadic,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    context_length: int = 1024,
    max_books: int = 5,
    batch_size: int = 2,
) -> float:
    """Evaluate language modeling perplexity on PG19."""
    dataset = PG19Dataset(
        split='test',
        context_length=context_length,
        tokenizer=tokenizer,
        max_books=max_books,
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='PG19 perplexity'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, labels=labels)
            # Count non-padding tokens
            num_tokens = (labels[:, 1:] != -100).sum().item()
            total_loss += outputs['loss'].item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    return perplexity


# ---------- Fixation Visualization ----------

def visualize_fixations(
    model: GPT2Saccadic,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    output_dir: str,
    context_length: int = 2048,
):
    """Generate fixation heatmaps showing where the model looks."""
    os.makedirs(output_dir, exist_ok=True)

    dataset = PasskeyRetrievalDataset(
        num_samples=10,
        context_length=context_length,
        tokenizer=tokenizer,
        seed=777,
    )

    model.eval()
    with torch.no_grad():
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            passkey_pos = sample['passkey_position']

            outputs = model(input_ids)

            # Collect fixation data across layers and saccades
            fig, axes = plt.subplots(
                len(outputs['fixation_info']), 1,
                figsize=(14, 2 * len(outputs['fixation_info'])),
                squeeze=False,
            )

            for ax_idx, (layer_idx, info) in enumerate(
                sorted(outputs['fixation_info'].items())
            ):
                ax = axes[ax_idx, 0]

                # Plot fixation logits as heatmap
                logits = torch.stack(info['fixation_logits'])  # (num_saccades, 1, num_blocks)
                probs = torch.softmax(logits[:, 0, :], dim=-1).cpu().numpy()

                ax.imshow(probs, aspect='auto', cmap='hot', interpolation='nearest')
                ax.set_ylabel(f'Layer {layer_idx}')
                ax.set_xlabel('Block index')
                ax.set_yticks(range(probs.shape[0]))
                ax.set_yticklabels([f'Saccade {t}' for t in range(probs.shape[0])])

                # Mark fixation points
                for t, fp in enumerate(info['fixation_points']):
                    block_idx = fp[0].item() // model.saccadic_blocks[str(layer_idx)].saccadic_attn.block_size
                    ax.plot(block_idx, t, 'c*', markersize=10)

                # Mark passkey position
                block_size = model.saccadic_blocks[str(layer_idx)].saccadic_attn.block_size
                passkey_block = passkey_pos // block_size
                ax.axvline(x=passkey_block, color='lime', linestyle='--', alpha=0.7, label='Passkey')

            fig.suptitle(f'Fixation patterns (example {idx}, passkey="{sample["passkey"]}")')
            axes[0, 0].legend(loc='upper right')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'fixation_example_{idx}.png'), dpi=150)
            plt.close(fig)

    print(f'Fixation visualizations saved to {output_dir}/')


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Evaluate Saccadic Attention GPT-2')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--task', type=str, default='all',
                        choices=['passkey', 'perplexity', 'fixations', 'all'])
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])

    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, config, device)
    print(f'Model loaded. Trainable params: {model.get_trainable_params():,}')

    results = {}

    if args.task in ('passkey', 'all'):
        print('\n=== Passkey Retrieval Evaluation ===')
        passkey_results = evaluate_passkey_sweep(
            model, tokenizer, device,
            context_lengths=config['eval']['context_lengths'],
            num_samples=config['data']['num_eval_samples'],
        )
        results['passkey'] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'example_fixations'}
            for k, v in passkey_results.items()
        }
        print('\nPasskey Results:')
        for ctx_len, res in sorted(passkey_results.items()):
            print(f'  {ctx_len:>6} tokens: {res["accuracy"]:.4f} accuracy')

    if args.task in ('perplexity', 'all'):
        print('\n=== PG19 Perplexity Evaluation ===')
        ppl = evaluate_perplexity(model, tokenizer, device)
        results['perplexity'] = ppl
        print(f'PG19 Perplexity: {ppl:.2f}')

    if args.task in ('fixations', 'all'):
        print('\n=== Fixation Visualization ===')
        viz_dir = os.path.join(args.output_dir, 'fixations')
        visualize_fixations(model, tokenizer, device, viz_dir)

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()
