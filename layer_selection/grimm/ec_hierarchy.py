import numpy as np
import pandas as pd
from collections import Counter
import torch
import os
import json
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

RANDOM_STATE_SEED = 42


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def get_dataset(data_path):
    """
    Load GRIMM splits from a directory containing TSV files.

    Returns:
        dict of DataFrames keyed by split name
    """
    splits = {}
    for name in ['train', 'validation', 'test', 'test2']:
        filepath = os.path.join(data_path, f'{name}.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t')
            df.columns = df.columns.str.strip()
            df.rename(columns={'EC number': 'EC_number'}, inplace=True)
            df['EC_number'] = df['EC_number'].astype(str).str.strip()
            df['Sequence'] = df['Sequence'].astype(str).str.strip()
            splits[name] = df
            print(f"  {name}: {len(df)} sequences, {df['EC_number'].nunique()} unique ECs")
        else:
            print(f"  {name}: not found at {filepath}")

    return splits


# ══════════════════════════════════════════════════════════════
# EC HIERARCHY PARSING
# ══════════════════════════════════════════════════════════════

def parse_ec_at_level(ec_str, level):
    """
    Parse an EC string to a given hierarchy level.
    For multi-functional enzymes (semicolon-delimited), uses the FIRST annotation.

    Args:
        ec_str: e.g. '3.6.5.n1', '1.1.1.10;1.1.1.162'
        level: 1-4

    Returns:
        str or None
    """
    ec = str(ec_str).split(';')[0].strip()
    parts = ec.split('.')

    if len(parts) < level:
        return None

    for i in range(level):
        if parts[i] == '-':
            return None

    return '.'.join(parts[:level])


# ══════════════════════════════════════════════════════════════
# REPRESENTATION EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_all_layer_representations_meanpool(model, tokenizer, sequence, device):
    """
    Extract MEAN-POOLED representations from ALL layers.
    Returns one vector per layer per protein.
    Used for: layer selection probing.

    Special tokens (CLS/EOS) excluded from pooling.

    Returns:
        np.array of shape (n_layers, hidden_dim)
    """
    inputs = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True,
                       max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # ESM-2 tokenization: [CLS] aa1 aa2 ... aaN [EOS]
    # We want positions 1 to len(sequence) inclusive (skip CLS at 0, EOS at end)
    seq_len = len(sequence)

    layer_reprs = []
    for layer_hidden_state in outputs.hidden_states:
        # layer_hidden_state: (1, total_tokens, hidden_dim)
        # Extract residue positions only (exclude CLS at pos 0 and EOS at end)
        residue_reprs = layer_hidden_state[0, 1:seq_len + 1, :]  # (seq_len, hidden_dim)
        layer_reprs.append(residue_reprs.mean(dim=0).cpu().numpy())

    return np.array(layer_reprs)  # (n_layers, hidden_dim)


def get_single_layer_per_residue(model, tokenizer, sequence, layer_idx, device):
    """
    Extract PER-RESIDUE representations from a SINGLE layer.
    Returns one vector per amino acid position.
    Used for: SAE training, residue-level feature discovery.

    Special tokens (CLS/EOS) excluded.

    Returns:
        np.array of shape (seq_len, hidden_dim)
    """
    inputs = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True,
                       max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    seq_len = len(sequence)

    # hidden_states[0] = embedding, hidden_states[i+1] = output of layer i
    # So hidden_states[layer_idx] gives the layer we want
    # (layer_idx=0 → embedding, layer_idx=5 → layer 5 output)
    hidden = outputs.hidden_states[layer_idx]  # (1, total_tokens, hidden_dim)

    # Exclude CLS (position 0) and EOS (position seq_len+1)
    residue_reprs = hidden[0, 1:seq_len + 1, :]  # (seq_len, hidden_dim)

    return residue_reprs.cpu().numpy()


# ══════════════════════════════════════════════════════════════
# PROBING MODE — DATA PREPARATION
# ══════════════════════════════════════════════════════════════

def prepare_probing_data(model, tokenizer, df, num_samples, min_class_size,
                         device, max_seq_len=512):
    """
    Extract mean-pooled representations for all layers + parse EC labels.
    Used for layer selection via linear probing.

    Returns:
        all_reprs: np.array (n_proteins, n_layers, hidden_dim)
        all_labels: dict with 'level_1'..'level_4', each a list of strings
    """
    # ── Parse EC labels, count frequencies ──
    print("Parsing EC hierarchy and filtering...")

    candidate_rows = []
    for idx, row in df.iterrows():
        if len(candidate_rows) >= num_samples:
            break

        seq = str(row['Sequence']).strip()
        ec = str(row['EC_number']).strip()

        if len(seq) > max_seq_len or len(seq) < 10:
            continue

        labels = {}
        valid = True
        for level in range(1, 5):
            parsed = parse_ec_at_level(ec, level)
            if parsed is None:
                valid = False
                break
            labels[f'level_{level}'] = parsed

        if not valid:
            continue

        candidate_rows.append({'sequence': seq, 'ec_full': ec, **labels})

    candidates = pd.DataFrame(candidate_rows)
    print(f"  Proteins with valid 4-level EC: {len(candidates)}")

    valid_classes = {}
    for level in range(1, 5):
        level_name = f'level_{level}'
        counts = Counter(candidates[level_name])
        valid = {l for l, c in counts.items() if c >= min_class_size}
        valid_classes[level_name] = valid
        print(f"  {level_name}: {len(valid)} classes with ≥{min_class_size} samples")

    # ── Extract representations ──
    print("\nExtracting mean-pooled representations from all layers...")
    model.eval()

    all_layer_reprs = []
    ec_labels = {f'level_{i}': [] for i in range(1, 5)}
    kept = 0

    for idx, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Proteins"):
        has_valid = any(
            row[f'level_{level}'] in valid_classes[f'level_{level}']
            for level in range(1, 5)
        )
        if not has_valid:
            continue

        reprs = get_all_layer_representations_meanpool(model, tokenizer, row['sequence'], device)
        all_layer_reprs.append(reprs)

        for level in range(1, 5):
            ec_labels[f'level_{level}'].append(row[f'level_{level}'])

        kept += 1

    print(f"\nTotal proteins kept: {kept}")
    return np.array(all_layer_reprs), ec_labels


# ══════════════════════════════════════════════════════════════
# EXTRACTION MODE — PER-RESIDUE FOR SAE TRAINING
# ══════════════════════════════════════════════════════════════

def extract_residue_representations(model, tokenizer, df, layer_idx, device,
                                    num_samples=50000, max_seq_len=512,
                                    save_path=None):
    """
    Extract per-residue representations for SAE training.

    Each protein contributes seq_len vectors of dimension hidden_dim.
    All residue vectors are concatenated into a single (N_total_residues, hidden_dim) array.

    Also saves metadata: which residues belong to which protein, EC labels, etc.

    Args:
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        df: DataFrame with Sequence and EC_number columns
        layer_idx: which layer to extract from
        device: cuda/cpu
        num_samples: max proteins to process
        max_seq_len: skip sequences longer than this
        save_path: directory to save outputs

    Returns:
        all_residue_reprs: np.array (N_total_residues, hidden_dim)
        metadata: list of dicts with protein-level info and residue ranges
    """
    print(f"\nExtracting per-residue representations from Layer {layer_idx}")
    print(f"  Max proteins: {num_samples}, Max seq len: {max_seq_len}")

    model.eval()

    all_residue_reprs = []
    metadata = []
    total_residues = 0
    proteins_processed = 0

    for idx, row in tqdm(df.iterrows(), total=min(len(df), num_samples), desc="Extracting"):
        if proteins_processed >= num_samples:
            break

        seq = str(row['Sequence']).strip()
        ec = str(row['EC_number']).strip()

        if len(seq) > max_seq_len or len(seq) < 10:
            continue

        # Parse EC at all levels
        ec_labels = {}
        for level in range(1, 5):
            ec_labels[f'ec_level_{level}'] = parse_ec_at_level(ec, level)

        # Extract per-residue representations
        residue_reprs = get_single_layer_per_residue(
            model, tokenizer, seq, layer_idx, device
        )  # (seq_len, hidden_dim)

        # Track residue ranges for this protein
        start_idx = total_residues
        end_idx = total_residues + residue_reprs.shape[0]

        metadata.append({
            'protein_idx': proteins_processed,
            'entry': row.get('Entry', f'protein_{proteins_processed}'),
            'ec_full': ec,
            **ec_labels,
            'seq_len': len(seq),
            'residue_start': start_idx,
            'residue_end': end_idx,
        })

        all_residue_reprs.append(residue_reprs)
        total_residues += residue_reprs.shape[0]
        proteins_processed += 1

    # Concatenate all residues
    all_residue_reprs = np.concatenate(all_residue_reprs, axis=0)

    print(f"\n  Proteins processed: {proteins_processed}")
    print(f"  Total residues: {total_residues:,}")
    print(f"  Representation shape: {all_residue_reprs.shape}")
    print(f"  Mean residues per protein: {total_residues / proteins_processed:.1f}")

    # Save
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save residue representations as numpy
        reprs_file = save_dir / f"residue_reprs_layer{layer_idx}.npy"
        np.save(reprs_file, all_residue_reprs)
        print(f"  Saved representations: {reprs_file} ({all_residue_reprs.nbytes / 1e9:.2f} GB)")

        # Also save a mean-pooled version (for compatibility with probing)
        meanpool_reprs = []
        for meta in metadata:
            start, end = meta['residue_start'], meta['residue_end']
            meanpool_reprs.append(all_residue_reprs[start:end].mean(axis=0))
        meanpool_reprs = np.array(meanpool_reprs)
        meanpool_file = save_dir / f"meanpool_reprs_layer{layer_idx}.npy"
        np.save(meanpool_file, meanpool_reprs)
        print(f"  Saved mean-pooled: {meanpool_file}")

        # Save metadata
        meta_file = save_dir / f"metadata_layer{layer_idx}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {meta_file}")

        # Save protein-level labels for convenience
        labels_df = pd.DataFrame(metadata)
        labels_file = save_dir / f"labels_layer{layer_idx}.csv"
        labels_df.to_csv(labels_file, index=False)
        print(f"  Saved labels: {labels_file}")

    return all_residue_reprs, metadata


# ══════════════════════════════════════════════════════════════
# PROBING
# ══════════════════════════════════════════════════════════════

def probe_layer(layer_num, all_reprs, labels, test_size=0.2):
    """Train a linear probe on a specific layer's mean-pooled representations."""
    X = all_reprs[:, layer_num, :]
    y = np.array(labels)

    counts = Counter(y)
    valid = [l for l, c in counts.items() if c >= 2]
    mask = np.isin(y, valid)
    X_f, y_f = X[mask], y[mask]

    if len(X_f) < 10:
        return 0.0, 0.0

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_f, y_f, test_size=test_size, random_state=RANDOM_STATE_SEED, stratify=y_f)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_f, y_f, test_size=test_size, random_state=RANDOM_STATE_SEED)

    probe = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE_SEED)
    probe.fit(X_tr, y_tr)

    y_pred = probe.predict(X_te)
    return accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average='weighted')


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_ec_hierarchy_results(best_result, all_layer_results, n_layers, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    titles = ['EC Level 1 (Main Class)',
              'EC Level 2 (Subclass)',
              'EC Level 3 (Sub-subclass)',
              'EC Level 4 (Specific Enzyme)']
    colors = ['#58a6ff', '#4caf50', '#f39c12', '#e74c3c']

    for idx in range(4):
        ax = axes[idx]
        key = f'level_{idx + 1}'
        accs = all_layer_results[key]
        layers = list(range(len(accs)))

        ax.plot(layers, accs, 'o-', linewidth=2, markersize=8,
                color=colors[idx], markeredgecolor='white', markeredgewidth=1)

        best_layer = best_result[key]['layer']
        best_acc = best_result[key]['accuracy']

        ax.axvline(best_layer, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(best_layer, best_acc, 'r*', markersize=15, zorder=5)
        ax.annotate(f'Layer {best_layer}\n{best_acc:.3f}',
                    xy=(best_layer, best_acc),
                    xytext=(best_layer + 0.5, best_acc - 0.05),
                    fontsize=10, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(titles[idx], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layers)

    fig.suptitle('ESM-2: Layer-wise EC Classification Accuracy (GRIMM Dataset)\n'
                 '(Logistic regression on mean-pooled representations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'ec_hierarchy_all_levels.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ESM-2 Layer Selection & Representation Extraction (GRIMM dataset)"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="probe",
                        choices=["probe", "extract"],
                        help="'probe' for layer selection, 'extract' for SAE training data")

    # Model
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="HuggingFace model name")

    # Data
    parser.add_argument("--data_path", type=str, default="../../data/grimm",
                        help="Path to GRIMM split directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Which split to use (train, test, etc.)")
    parser.add_argument("--dataset_size", type=int, default=185419,
                        help="Max proteins to process")
    parser.add_argument("--min_class_size", type=int, default=2,
                        help="Min proteins per EC class (probe mode)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Skip sequences longer than this")

    # Extract mode
    parser.add_argument("--layer", type=int, default=5,
                        help="Layer to extract (extract mode only)")

    # Output
    parser.add_argument("--save_dir", type=str, default="../results/")
    parser.add_argument("--extract_dir", type=str, default="../data/residue_representations",
                        help="Where to save extracted representations (extract mode)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (auto-detected if not set)")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ──
    print(f"Loading model: {args.model_name}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size

    print(f"  Layers: {n_layers} (0=embedding, 1-{n_layers - 1}=transformer)")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Device: {args.device}")

    # ── Load data ──
    print(f"\nLoading GRIMM dataset from: {args.data_path}")
    splits = get_dataset(args.data_path)

    if args.split not in splits:
        print(f"ERROR: {args.split} not found. Available: {list(splits.keys())}")
        return

    df = splits[args.split]
    print(f"Using split '{args.split}': {len(df)} sequences")

    # ══════════════════════════════════════════════════════════
    # MODE: PROBE — Layer selection via linear probing
    # ══════════════════════════════════════════════════════════
    if args.mode == "probe":
        print(f"\n{'=' * 60}")
        print("MODE: LAYER SELECTION PROBING (mean-pooled)")
        print(f"{'=' * 60}")

        all_reprs, all_labels = prepare_probing_data(
            model, tokenizer, df,
            num_samples=args.dataset_size,
            min_class_size=args.min_class_size,
            device=args.device,
            max_seq_len=args.max_seq_len,
        )

        if len(all_reprs) == 0:
            print("ERROR: No proteins passed filtering.")
            return

        # Probe each layer for each EC level
        best_result = {}
        all_layer_results = {}

        for level_idx in range(4):
            level_name = f'level_{level_idx + 1}'
            labels = all_labels[level_name]
            n_classes = len(set(labels))

            print(f"\n{'─' * 50}")
            print(f"EC Level {level_idx + 1} ({n_classes} classes, {len(labels)} proteins)")
            print(f"{'─' * 50}")

            layer_accs = []
            best_acc, best_layer = 0, 0

            for layer in range(n_layers):
                acc, f1 = probe_layer(layer, all_reprs, labels)
                layer_accs.append(acc)
                print(f"  Layer {layer}: accuracy={acc:.4f}, f1={f1:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_layer = layer

            best_result[level_name] = {'layer': best_layer, 'accuracy': best_acc}
            all_layer_results[level_name] = layer_accs
            print(f"  → Best: Layer {best_layer} ({best_acc:.4f})")

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY — Best layer per EC level")
        print(f"{'=' * 60}")
        for level_idx in range(4):
            key = f'level_{level_idx + 1}'
            info = best_result[key]
            print(f"  EC Level {level_idx + 1}: Layer {info['layer']} → {info['accuracy']:.4f}")

        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        plot_ec_hierarchy_results(best_result, all_layer_results, n_layers, args.save_dir)

        save_data = {
            "mode": "probe",
            "dataset": "GRIMM",
            "model": args.model_name,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_proteins": len(all_reprs),
            "best_result": {k: {"layer": int(v["layer"]), "accuracy": float(v["accuracy"])}
                            for k, v in best_result.items()},
            "all_layer_results": {k: [float(a) for a in v]
                                  for k, v in all_layer_results.items()},
        }
        json_path = os.path.join(args.save_dir, "layer_selection_results.json")
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to: {json_path}")

    # ══════════════════════════════════════════════════════════
    # MODE: EXTRACT — Per-residue representations for SAE
    # ══════════════════════════════════════════════════════════
    elif args.mode == "extract":
        print(f"\n{'=' * 60}")
        print(f"MODE: PER-RESIDUE EXTRACTION (Layer {args.layer})")
        print(f"{'=' * 60}")
        print(f"  Output: one {hidden_dim}-dim vector per amino acid")
        print(f"  Use case: SAE training → residue-level feature discovery")

        residue_reprs, metadata = extract_residue_representations(
            model, tokenizer, df,
            layer_idx=args.layer,
            device=args.device,
            num_samples=args.dataset_size,
            max_seq_len=args.max_seq_len,
            save_path=args.extract_dir,
        )

        # Print summary statistics
        print(f"\n{'=' * 60}")
        print("EXTRACTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Model: {args.model_name}")
        print(f"  Layer: {args.layer}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Proteins: {len(metadata)}")
        print(f"  Total residues: {residue_reprs.shape[0]:,}")
        print(f"  Shape: {residue_reprs.shape}")
        print(f"  Size: {residue_reprs.nbytes / 1e9:.2f} GB")
        print(f"\n  Files saved to: {args.extract_dir}")
        print(f"    residue_reprs_layer{args.layer}.npy  — per-residue for SAE training")
        print(f"    meanpool_reprs_layer{args.layer}.npy  — mean-pooled for probing")
        print(f"    metadata_layer{args.layer}.json       — protein info + residue ranges")
        print(f"    labels_layer{args.layer}.csv          — protein labels")

        # EC class distribution
        ec1_counts = Counter(m.get('ec_level_1', 'unknown') for m in metadata)
        print(f"\n  EC Level 1 distribution:")
        for ec, count in sorted(ec1_counts.items(), key=lambda x: -x[1]):
            print(f"    {ec}: {count}")

        print(f"\n  Next step: Train SAE on residue_reprs_layer{args.layer}.npy")
        print(f"    from train_sae import AnthropicSAE")
        print(f"    acts = np.load('residue_reprs_layer{args.layer}.npy')")
        print(f"    acts = torch.from_numpy(acts).float()")
        print(f"    # Train SAE with d_model={hidden_dim}, dict_size=8192")


if __name__ == "__main__":
    main()