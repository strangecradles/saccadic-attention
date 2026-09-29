"""Generate docs/figures/results.png from the raw files in results/."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures" / "results.png"


def load_passkey():
    with open(RESULTS / "results.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    ids = [int(r["experiment_id"]) for r in rows]
    accs = [float(r["passkey_accuracy"]) for r in rows]
    return ids, accs


def load_multihop():
    with open(RESULTS / "gpt2_multihop_results.json") as f:
        rows = json.load(f)
    labels = [f"{r['n_hops']}h/{r['n_saccades']}s" for r in rows]
    accs = [r["accuracy"] for r in rows]
    return labels, accs


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ids, accs = load_passkey()
    ax1.bar([str(i) for i in ids], accs, color="#4878cf")
    ax1.set_xlabel("Experiment ID (results.tsv)")
    ax1.set_ylabel("Passkey accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title("Passkey retrieval (GPT-2, 4096 ctx)")

    labels, accs = load_multihop()
    ax2.bar(labels, accs, color="#6acc65")
    ax2.set_xlabel("Hops / saccades")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Multi-hop reasoning (GPT-2)")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
