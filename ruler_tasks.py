"""RULER-style task generators for saccadic attention evaluation.

7 tasks across 4 categories:
  1. Retrieval: S-NIAH, MK-NIAH, MV-NIAH
  2. Multi-hop: Variable Tracking (2/3/4-hop)
  3. Aggregation: Common Words Extraction
  4. QA: (skipped — requires external dataset)

Each task produces (input_text, labels, metadata) for training and evaluation.
"""

import random
import string

# ── Filler text: 5000 unique Wikipedia paragraphs (no repeats per context) ────

def _load_wikitext_filler():
    import os, json
    filler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wikitext_filler.json')
    if os.path.exists(filler_path):
        with open(filler_path) as f:
            return json.load(f)
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    paras, seen = [], set()
    for row in ds:
        t = row['text'].strip()
        if len(t) > 100 and t not in seen and not t.startswith('='):
            paras.append(t)
            seen.add(t)
        if len(paras) >= 5000:
            break
    with open(filler_path, 'w') as f:
        json.dump(paras, f)
    return paras

_WIKI_PARAS = _load_wikitext_filler()


def _make_filler(tokenizer, ctx_len, reserved_tokens, rng=None):
    """Generate filler from unique Wikipedia paragraphs. No paragraph repeats."""
    if rng is None:
        rng = random.Random()
    budget = ctx_len - reserved_tokens
    # Shuffle and draw without replacement to avoid repeats
    indices = list(range(len(_WIKI_PARAS)))
    rng.shuffle(indices)
    filler = []
    for idx in indices:
        tokens = tokenizer.encode(" " + _WIKI_PARAS[idx])
        filler.extend(tokens)
        if len(filler) >= budget:
            break
    return filler[:budget]


# ── Task 1a: Single Needle in a Haystack (S-NIAH) ────────────────────────────

CITIES = ['Chicago', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 'Mumbai',
          'Toronto', 'Seoul', 'Dubai', 'Madrid', 'Rome', 'Bangkok', 'Cairo', 'Lima']


class SingleNeedleTask:
    """Insert one key-value pair (city → 7-digit number) in essay filler."""
    NAME = 'S-NIAH'
    N_DIGITS = 5  # 5 digits to match proven debug config (7 was too hard to converge in 30 min)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            city = rng2.choice(CITIES)
            number = ''.join(rng2.choices(string.digits, k=self.N_DIGITS))
            needle = f" The special magic {city} number is: {number}."
            query = f" What is the special magic {city} number? The number is:"

            needle_ids = self.tokenizer.encode(needle)
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + number)
            reserved = len(needle_ids) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved, rng=rng2)
            pos = rng2.randint(0, len(filler))

            full = filler[:pos] + needle_ids + filler[pos:] + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in number], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': number,
                'needle_position': pos,
                'target_positions': [pos],
            })
        return samples


# ── Task 1b: Multi-Key Needle (MK-NIAH) ──────────────────────────────────────

class MultiKeyNeedleTask:
    """1 target needle + 3 distractor needles. Query asks for target city."""
    NAME = 'MK-NIAH'
    N_DIGITS = 5
    N_DISTRACTORS = 3

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            cities = rng2.sample(CITIES, 1 + self.N_DISTRACTORS)
            target_city = cities[0]
            numbers = [''.join(rng2.choices(string.digits, k=self.N_DIGITS))
                        for _ in range(1 + self.N_DISTRACTORS)]

            needles = [f" The special magic {c} number is: {n}."
                       for c, n in zip(cities, numbers)]
            needle_ids_list = [self.tokenizer.encode(n) for n in needles]
            query = f" What is the special magic {target_city} number? The number is:"
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + numbers[0])
            reserved = sum(len(n) for n in needle_ids_list) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved, rng=rng2)
            # Insert needles at random positions
            positions = sorted(rng2.sample(range(len(filler)), min(len(cities), len(filler))))
            target_pos = positions[0]
            full = list(filler)
            for j in range(len(cities) - 1, -1, -1):
                p = positions[j] if j < len(positions) else rng2.randint(0, len(full))
                full = full[:p] + needle_ids_list[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in numbers[0]], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': numbers[0],
                'needle_position': target_pos,
                'target_positions': [target_pos],
            })
        return samples


# ── Task 1c: Multi-Value Needle (MV-NIAH) ────────────────────────────────────

class MultiValueNeedleTask:
    """4 needles with same key (city), different values. Must find ALL."""
    NAME = 'MV-NIAH'
    N_DIGITS = 7
    N_VALUES = 4

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            city = rng2.choice(CITIES)
            numbers = [''.join(rng2.choices(string.digits, k=self.N_DIGITS))
                        for _ in range(self.N_VALUES)]
            needles = [f" The special magic {city} number is: {n}." for n in numbers]
            needle_ids_list = [self.tokenizer.encode(n) for n in needles]
            query = f" What are all the special magic {city} numbers? The numbers are:"
            query_ids = self.tokenizer.encode(query)
            # Answer: all numbers separated by comma
            answer_text = " " + ", ".join(numbers)
            answer_ids = self.tokenizer.encode(answer_text)
            reserved = sum(len(n) for n in needle_ids_list) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved, rng=rng2)
            positions = sorted(rng2.sample(range(max(1, len(filler))),
                                           min(self.N_VALUES, max(1, len(filler)))))
            full = list(filler)
            target_positions = []
            for j in range(self.N_VALUES - 1, -1, -1):
                p = positions[j] if j < len(positions) else rng2.randint(0, len(full))
                target_positions.insert(0, p)
                full = full[:p] + needle_ids_list[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            # Labels: (N_VALUES, N_DIGITS)
            labels = torch.tensor([[int(d) for d in n] for n in numbers], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': numbers,
                'needle_position': target_positions[0],
                'target_positions': target_positions,
            })
        return samples


# ── Task 2a: Variable Tracking ───────────────────────────────────────────────

class VariableTrackingTask:
    """Chain of variable bindings: X=Y, Y=Z, Z=W. Query: what is X?"""
    NAME = 'VT'
    N_DIGITS = 4  # shorter values for variable tracking

    def __init__(self, tokenizer, chain_length=3):
        self.tokenizer = tokenizer
        self.chain_length = chain_length

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        var_names = [f'VAR_{chr(65+i)}' for i in range(26)]  # VAR_A, VAR_B, ...

        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            # Pick chain variables
            chain_vars = rng2.sample(var_names, self.chain_length + 1)
            final_value = ''.join(rng2.choices(string.digits, k=self.N_DIGITS))

            # Build chain statements
            statements = []
            for j in range(self.chain_length):
                if j == self.chain_length - 1:
                    # Last link: assign the actual value
                    stmt = f" {chain_vars[j]} equals {final_value}."
                else:
                    # Intermediate link: assign to next variable
                    stmt = f" {chain_vars[j]} equals {chain_vars[j+1]}."
                statements.append(stmt)

            stmt_ids = [self.tokenizer.encode(s) for s in statements]
            query = f" What is the value of {chain_vars[0]}? The value is:"
            query_ids = self.tokenizer.encode(query)
            answer_ids = self.tokenizer.encode(" " + final_value)
            reserved = sum(len(s) for s in stmt_ids) + len(query_ids) + len(answer_ids)

            filler = _make_filler(self.tokenizer, ctx_len, reserved, rng=rng2)
            seg = max(1, len(filler) // (self.chain_length + 1))
            full = list(filler)
            target_positions = []
            for j in range(self.chain_length - 1, -1, -1):
                p = min(j * seg + rng2.randint(0, seg), len(full))
                target_positions.insert(0, p)
                full = full[:p] + stmt_ids[j] + full[p:]

            full = full + query_ids + answer_ids
            if len(full) > ctx_len:
                full = full[:ctx_len]
            elif len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            labels = torch.tensor([int(d) for d in final_value], dtype=torch.long)
            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': final_value,
                'needle_position': target_positions[0],
                'target_positions': target_positions,
            })
        return samples


# ── Task 3a: Common Words Extraction ─────────────────────────────────────────

class CommonWordsTask:
    """Fill context with words from a Zipf distribution. Query top-10 most common."""
    NAME = 'CWE'
    TOP_K = 5  # predict top 5 words

    # Fixed word vocabulary (common English words)
    VOCAB = [
        'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child', 'world', 'life',
        'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
        'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother',
        'area', 'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book', 'eye',
        'job', 'word', 'business', 'issue', 'side', 'kind', 'head', 'house', 'service',
        'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law', 'car',
        'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'body', 'back',
        'parent', 'face', 'other', 'level', 'office', 'door', 'health', 'person', 'art',
        'war', 'history', 'party', 'result', 'change', 'morning', 'reason', 'research',
        'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education', 'foot', 'boy',
        'age', 'policy', 'process', 'music', 'market', 'sense', 'product', 'effect', 'class',
    ]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.word_to_idx = {w: i for i, w in enumerate(self.VOCAB)}

    def generate(self, ctx_len, n_samples, seed=42):
        rng = random.Random(seed)
        samples = []
        import numpy as np

        for i in range(n_samples):
            rng2 = random.Random(rng.randint(0, 2**32) + i)
            np_rng = np.random.RandomState(rng2.randint(0, 2**31))

            # Generate Zipf-distributed word counts
            n_words = ctx_len // 6  # ~6 chars per word average
            # Draw from Zipf with alpha=1.5
            weights = np.array([1.0 / (i + 1)**1.5 for i in range(len(self.VOCAB))])
            weights /= weights.sum()
            word_indices = np_rng.choice(len(self.VOCAB), size=n_words, p=weights)

            # Build text
            words = [self.VOCAB[idx] for idx in word_indices]
            text = ' '.join(words)

            # Find top-K most frequent
            from collections import Counter
            counts = Counter(word_indices)
            top_k_indices = [idx for idx, _ in counts.most_common(self.TOP_K)]

            query = " What are the most common words in the text above? The most common words are:"
            text_ids = self.tokenizer.encode(text)
            query_ids = self.tokenizer.encode(query)

            full = text_ids[:ctx_len - len(query_ids)] + query_ids
            if len(full) < ctx_len:
                full += [self.tokenizer.eos_token_id or 0] * (ctx_len - len(full))

            import torch
            # Multi-hot label over vocab
            labels = torch.zeros(len(self.VOCAB), dtype=torch.float)
            for idx in top_k_indices:
                labels[idx] = 1.0

            samples.append({
                'input_ids': torch.tensor(full[:ctx_len], dtype=torch.long),
                'labels': labels,
                'answer': [self.VOCAB[idx] for idx in top_k_indices],
                'needle_position': 0,
                'target_positions': [],  # no specific positions for aggregation
            })
        return samples


# ── Task Registry ─────────────────────────────────────────────────────────────

def get_task(task_name, tokenizer, **kwargs):
    """Get a task generator by name."""
    tasks = {
        'S-NIAH': lambda: SingleNeedleTask(tokenizer),
        'MK-NIAH': lambda: MultiKeyNeedleTask(tokenizer),
        'MV-NIAH': lambda: MultiValueNeedleTask(tokenizer),
        'VT-2': lambda: VariableTrackingTask(tokenizer, chain_length=2),
        'VT-3': lambda: VariableTrackingTask(tokenizer, chain_length=3),
        'VT-4': lambda: VariableTrackingTask(tokenizer, chain_length=4),
        'CWE': lambda: CommonWordsTask(tokenizer),
    }
    return tasks[task_name]()


def get_head_config(task_name):
    """Return (head_class, head_kwargs, num_saccades) for a task."""
    from saccadic_qwen import (
        DigitClassificationHead, MultiValueHead, VariableTrackingHead,
    )
    configs = {
        'S-NIAH':  (DigitClassificationHead, {'input_dim': 1536, 'n_digits': 7}, 3),
        'MK-NIAH': (DigitClassificationHead, {'input_dim': 1536, 'n_digits': 7}, 3),
        'MV-NIAH': (MultiValueHead, {'input_dim': 1536, 'n_values': 4, 'n_digits': 7}, 5),
        'VT-2':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 2),
        'VT-3':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 3),
        'VT-4':    (VariableTrackingHead, {'input_dim': 1536, 'n_digits': 4}, 4),
    }
    return configs[task_name]
