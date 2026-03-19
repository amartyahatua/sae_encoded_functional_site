import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


RANDOM_STATE_SEED = 42


# ============================================================
# SAE MODEL
# ============================================================

class TopKSAE(nn.Module):
    def __init__(self, d_model, dict_size, top_k=256):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.top_k = top_k
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

    def encode(self, x):
        pre_acts = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_acts, self.top_k, dim=-1)
        topk_values = F.relu(topk_values)
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, topk_indices, topk_values)
        return sparse_acts

    def forward(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    D_MODEL = 320
    LAYER_IDX = 5

    # SAE to analyze (best config from training)
    DICT_SIZE = 16384
    TOP_K = 256

    # Data limits
    MAX_PROTEINS = 50000    # Analyze up to this many proteins
    BATCH_SIZE = 2048

    # Paths
    SAE_DIR = "../../mi_sae_esm/models/sae_topk"
    REPRS_PATH = "../../mi_layer_esm/data/residue_representations/residue_reprs_layer5.npy"
    METADATA_PATH = "../../mi_layer_esm/data/residue_representations/metadata_layer5.json"
    RESULTS_DIR = "../results/feature_analysis"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATA LOADING
# ============================================================

def load_sae(sae_dir, dict_size, top_k, layer_idx, device):
    """Load trained TopK SAE."""
    ckpt_name = f"esm2_8M_layer{layer_idx}_dict{dict_size}_topk{top_k}.pt"
    path = Path(sae_dir) / ckpt_name

    if not path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    sae = TopKSAE(
        d_model=checkpoint['d_model'],
        dict_size=checkpoint['dict_size'],
        top_k=checkpoint['top_k'],
    ).to(device)
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()

    print(f"  Loaded SAE: {path}")
    print(f"  d_model={checkpoint['d_model']}, dict={checkpoint['dict_size']}, K={checkpoint['top_k']}")
    return sae


def load_data(reprs_path, metadata_path, max_proteins):
    """Load per-residue representations and metadata."""
    print(f"\n  Loading metadata: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if max_proteins < len(metadata):
        metadata = metadata[:max_proteins]

    # Figure out how many residues we need
    last_protein = metadata[-1]
    max_residue_idx = last_protein['residue_end']

    print(f"  Proteins: {len(metadata)}")
    print(f"  Residues needed: {max_residue_idx:,}")

    # Memory-map and load only what we need
    print(f"  Loading representations: {reprs_path}")
    mmap = np.load(reprs_path, mmap_mode='r')
    reprs = np.array(mmap[:max_residue_idx])

    print(f"  Loaded: {reprs.shape}, {reprs.nbytes / 1e9:.2f} GB")
    return reprs, metadata


# ============================================================
# FEATURE ACTIVATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_feature_activations(sae, reprs, metadata, device, batch_size=2048):
    """
    Run SAE encoder on all residues and collect per-protein feature statistics.

    Returns:
        protein_features: list of dicts, one per protein, each containing:
            - 'feature_activations': dict mapping feature_id → mean activation
            - 'feature_residue_acts': dict mapping feature_id → list of (position, activation)
            - 'ec_labels': dict of EC at each level
        feature_protein_map: dict mapping feature_id → list of (protein_idx, mean_activation)
        feature_ec_map: dict mapping feature_id → Counter of EC labels (level 1-4)
    """
    print("\nExtracting feature activations...")

    # Per-feature global tracking
    feature_protein_map = defaultdict(list)           # feature → [(protein_idx, mean_act)]
    feature_ec_counts = {f'level_{l}': defaultdict(lambda: defaultdict(float))
                         for l in range(1, 5)}        # level → feature → {ec: total_act}
    feature_total_act = defaultdict(float)             # feature → total activation
    feature_residue_count = defaultdict(int)           # feature → how many residues activate it

    protein_features = []

    for pidx, meta in enumerate(tqdm(metadata, desc="Proteins")):
        start = meta['residue_start']
        end = meta['residue_end']
        seq_len = end - start

        if seq_len == 0:
            continue

        # Get this protein's residue representations
        protein_reprs = torch.from_numpy(reprs[start:end]).float().to(device)

        # Encode in batches if protein is very long
        all_acts = []
        for i in range(0, seq_len, batch_size):
            batch = protein_reprs[i:i + batch_size]
            acts = sae.encode(batch)  # (batch_len, dict_size)
            all_acts.append(acts.cpu())

        all_acts = torch.cat(all_acts, dim=0)  # (seq_len, dict_size)

        # Per-feature statistics for this protein
        # Which features are active (non-zero) for any residue?
        active_mask = (all_acts > 0)  # (seq_len, dict_size)
        active_features = active_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()

        protein_feat_data = {
            'protein_idx': pidx,
            'entry': meta.get('entry', f'protein_{pidx}'),
            'ec_full': meta.get('ec_full', ''),
            'ec_labels': {f'level_{l}': meta.get(f'ec_level_{l}', None)
                          for l in range(1, 5)},
            'seq_len': seq_len,
            'n_active_features': len(active_features),
            'top_features': [],
        }

        # For each active feature, compute stats
        for feat_id in active_features:
            feat_acts = all_acts[:, feat_id]  # (seq_len,)
            nonzero_mask = feat_acts > 0
            n_active_residues = nonzero_mask.sum().item()

            if n_active_residues == 0:
                continue

            mean_act = feat_acts[nonzero_mask].mean().item()
            max_act = feat_acts.max().item()
            total_act = feat_acts.sum().item()

            # Track globally
            feature_protein_map[feat_id].append((pidx, mean_act))
            feature_total_act[feat_id] += total_act
            feature_residue_count[feat_id] += n_active_residues

            # Track EC association
            for level in range(1, 5):
                ec_label = meta.get(f'ec_level_{level}', None)
                if ec_label is not None:
                    feature_ec_counts[f'level_{level}'][feat_id][ec_label] += total_act

            # Top activating residues for this feature in this protein
            top_residue_indices = feat_acts.topk(min(5, n_active_residues)).indices.tolist()
            top_residues = [
                {'pos': pos, 'activation': feat_acts[pos].item()}
                for pos in top_residue_indices
            ]

            protein_feat_data['top_features'].append({
                'feature_id': feat_id,
                'mean_activation': round(mean_act, 4),
                'max_activation': round(max_act, 4),
                'n_active_residues': n_active_residues,
                'frac_active': round(n_active_residues / seq_len, 4),
                'top_residues': top_residues,
            })

        # Sort by mean activation
        protein_feat_data['top_features'].sort(key=lambda x: -x['mean_activation'])
        protein_features.append(protein_feat_data)

    return protein_features, feature_protein_map, feature_ec_counts, feature_total_act, feature_residue_count


# ============================================================
# EC SPECIFICITY ANALYSIS
# ============================================================

def compute_ec_specificity(feature_ec_counts, feature_total_act, feature_residue_count,
                           dict_size, min_proteins=10):
    """
    For each feature, compute how specific it is to particular EC classes.

    Metrics:
      - Entropy: How spread out across EC classes? (low = specific)
      - Top EC fraction: What fraction of activation goes to the top EC?
      - Gini coefficient: Inequality of activation across ECs

    Returns:
        feature_specificity: dict mapping feature_id → specificity metrics per level
    """
    print("\nComputing EC specificity...")

    feature_specificity = {}

    for feat_id in range(dict_size):
        feat_spec = {'feature_id': feat_id}

        for level in range(1, 5):
            level_name = f'level_{level}'
            ec_acts = feature_ec_counts[level_name].get(feat_id, {})

            if len(ec_acts) == 0:
                feat_spec[level_name] = {
                    'n_ecs': 0, 'entropy': 0, 'top_ec': None,
                    'top_ec_frac': 0, 'gini': 0,
                }
                continue

            # Normalize to distribution
            total = sum(ec_acts.values())
            if total == 0:
                continue

            probs = {ec: act / total for ec, act in ec_acts.items()}

            # Entropy
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())
            max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # Top EC
            top_ec = max(probs, key=probs.get)
            top_ec_frac = probs[top_ec]

            # Gini coefficient
            sorted_probs = sorted(probs.values())
            n = len(sorted_probs)
            if n > 1:
                cum = np.cumsum(sorted_probs)
                gini = 1 - 2 * sum(cum) / (n * sum(sorted_probs)) + 1 / n
            else:
                gini = 0

            feat_spec[level_name] = {
                'n_ecs': len(ec_acts),
                'entropy': round(float(entropy), 4),
                'normalized_entropy': round(float(normalized_entropy), 4),
                'top_ec': top_ec,
                'top_ec_frac': round(float(top_ec_frac), 4),
                'gini': round(float(gini), 4),
            }

        feat_spec['total_activation'] = round(feature_total_act.get(feat_id, 0), 2)
        feat_spec['total_residues'] = feature_residue_count.get(feat_id, 0)
        feature_specificity[feat_id] = feat_spec

    return feature_specificity


# ============================================================
# FEATURE TAXONOMY
# ============================================================

def classify_features(feature_specificity, dict_size, ec_level='level_4'):
    """
    Classify features into three categories based on EC specificity.

    Categories:
      - GENERAL: Low specificity (normalized entropy > 0.8)
        Fires across many EC classes. Likely encodes structural/chemical
        properties shared across enzymes (e.g., alpha helix, hydrophobic core).

      - DISCRIMINATIVE: High specificity (top_ec_frac > 0.5, normalized entropy < 0.5)
        Fires primarily for specific EC classes. Likely encodes functional
        properties (e.g., catalytic mechanism, substrate binding).

      - MIXED: Moderate specificity (everything else)
        May encode properties that correlate with function but aren't directly
        causal (e.g., organism-specific bias, sequence length effects).

      - DEAD: Never activates. Not learned.

    Returns:
        taxonomy: dict with category → list of feature_ids
        summary: dict with counts and examples
    """
    print(f"\nClassifying features (based on {ec_level})...")

    taxonomy = {
        'GENERAL': [],
        'DISCRIMINATIVE': [],
        'MIXED': [],
        'DEAD': [],
    }

    for feat_id in range(dict_size):
        spec = feature_specificity.get(feat_id, {})
        level_spec = spec.get(ec_level, {})

        n_ecs = level_spec.get('n_ecs', 0)
        total_act = spec.get('total_activation', 0)

        if total_act == 0 or n_ecs == 0:
            taxonomy['DEAD'].append(feat_id)
            continue

        norm_entropy = level_spec.get('normalized_entropy', 0)
        top_ec_frac = level_spec.get('top_ec_frac', 0)

        if norm_entropy > 0.8:
            taxonomy['GENERAL'].append(feat_id)
        elif top_ec_frac > 0.5 and norm_entropy < 0.5:
            taxonomy['DISCRIMINATIVE'].append(feat_id)
        else:
            taxonomy['MIXED'].append(feat_id)

    # Summary
    summary = {}
    for cat, feat_ids in taxonomy.items():
        count = len(feat_ids)
        pct = 100 * count / dict_size

        if feat_ids and cat != 'DEAD':
            # Get top features by total activation
            top = sorted(feat_ids, key=lambda f: feature_specificity[f].get('total_activation', 0),
                         reverse=True)[:5]
            examples = []
            for f in top:
                spec = feature_specificity[f]
                ls = spec.get(ec_level, {})
                examples.append({
                    'feature_id': f,
                    'top_ec': ls.get('top_ec', '?'),
                    'top_ec_frac': ls.get('top_ec_frac', 0),
                    'norm_entropy': ls.get('normalized_entropy', 0),
                    'total_activation': spec.get('total_activation', 0),
                    'n_ecs': ls.get('n_ecs', 0),
                })
        else:
            examples = []

        summary[cat] = {
            'count': count,
            'pct': round(pct, 1),
            'examples': examples,
        }

        print(f"  {cat}: {count} features ({pct:.1f}%)")
        for ex in examples[:3]:
            print(f"    Feature {ex['feature_id']}: top_ec={ex['top_ec']} "
                  f"(frac={ex['top_ec_frac']:.2f}), entropy={ex['norm_entropy']:.2f}, "
                  f"n_ecs={ex['n_ecs']}")

    return taxonomy, summary


# ============================================================
# AMINO ACID PREFERENCE ANALYSIS
# ============================================================

@torch.no_grad()
def analyze_amino_acid_preferences(sae, reprs, metadata, device,
                                   max_proteins=10000, batch_size=4096):
    """
    For each feature, compute which amino acids it prefers.

    A feature that fires primarily at Cysteine positions is likely detecting
    disulfide bonds or metal coordination. A feature firing at His/Ser/Asp
    might detect catalytic triads.

    Returns:
        aa_preferences: dict mapping feature_id → Counter of amino acids
    """
    print("\nAnalyzing amino acid preferences...")

    AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
    aa_preferences = defaultdict(lambda: Counter())

    n_proteins = min(max_proteins, len(metadata))

    for pidx in tqdm(range(n_proteins), desc="AA analysis"):
        meta = metadata[pidx]
        start = meta['residue_start']
        end = meta['residue_end']
        seq_len = end - start

        if seq_len == 0:
            continue

        protein_reprs = torch.from_numpy(reprs[start:end]).float().to(device)

        for i in range(0, seq_len, batch_size):
            batch = protein_reprs[i:i + batch_size]
            acts = sae.encode(batch)  # (batch_len, dict_size)

            active = (acts > 0).cpu()

    return aa_preferences


# ============================================================
# PLOTTING
# ============================================================

def plot_feature_taxonomy(summary, save_dir):
    """Pie chart of feature taxonomy."""
    fig, ax = plt.subplots(figsize=(8, 8))

    categories = ['DISCRIMINATIVE', 'GENERAL', 'MIXED', 'DEAD']
    colors = ['#e74c3c', '#58a6ff', '#f39c12', '#555555']
    counts = [summary[c]['count'] for c in categories]
    labels = [f"{c}\n{summary[c]['count']} ({summary[c]['pct']}%)" for c in categories]

    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors, autopct='',
        startangle=90, textprops={'fontsize': 12}
    )

    ax.set_title('SAE Feature Taxonomy\n(ESM-2 8M, Layer 5, GRIMM Dataset)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_taxonomy_pie.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_specificity_distribution(feature_specificity, dict_size, save_dir, ec_level='level_4'):
    """Histogram of normalized entropy across all features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get entropy values for active features
    entropies = []
    top_fracs = []
    for feat_id in range(dict_size):
        spec = feature_specificity.get(feat_id, {})
        ls = spec.get(ec_level, {})
        if ls.get('n_ecs', 0) > 0:
            entropies.append(ls.get('normalized_entropy', 0))
            top_fracs.append(ls.get('top_ec_frac', 0))

    # Entropy histogram
    ax = axes[0]
    ax.hist(entropies, bins=50, color='#58a6ff', edgecolor='white', alpha=0.8)
    ax.axvline(0.5, color='#e74c3c', linestyle='--', label='Discriminative threshold')
    ax.axvline(0.8, color='#f39c12', linestyle='--', label='General threshold')
    ax.set_xlabel('Normalized Entropy', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title(f'EC Specificity Distribution ({ec_level})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top EC fraction histogram
    ax = axes[1]
    ax.hist(top_fracs, bins=50, color='#4caf50', edgecolor='white', alpha=0.8)
    ax.axvline(0.5, color='#e74c3c', linestyle='--', label='Discriminative threshold')
    ax.set_xlabel('Top EC Fraction', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title(f'Top EC Concentration ({ec_level})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'specificity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ec_level_comparison(feature_specificity, dict_size, save_dir):
    """Compare specificity across EC hierarchy levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for level in range(1, 5):
        level_name = f'level_{level}'
        entropies = []
        for feat_id in range(dict_size):
            spec = feature_specificity.get(feat_id, {})
            ls = spec.get(level_name, {})
            if ls.get('n_ecs', 0) > 0:
                entropies.append(ls.get('normalized_entropy', 0))

        ax.hist(entropies, bins=50, alpha=0.5,
                label=f'EC Level {level} (n={len(entropies)})')

    ax.set_xlabel('Normalized Entropy', fontsize=12)
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Specificity Across EC Hierarchy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ec_level_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_discriminative_features(feature_specificity, taxonomy, save_dir, ec_level='level_4',
                                     n_features=20):
    """Bar chart of top discriminative features and their top EC classes."""
    disc_features = taxonomy['DISCRIMINATIVE']
    if not disc_features:
        print("  No discriminative features to plot")
        return

    # Sort by top_ec_frac
    sorted_feats = sorted(
        disc_features,
        key=lambda f: feature_specificity[f].get(ec_level, {}).get('top_ec_frac', 0),
        reverse=True
    )[:n_features]

    feat_ids = []
    fracs = []
    ecs = []

    for f in sorted_feats:
        ls = feature_specificity[f].get(ec_level, {})
        feat_ids.append(f"F{f}")
        fracs.append(ls.get('top_ec_frac', 0))
        ecs.append(ls.get('top_ec', '?'))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(feat_ids)), fracs, color='#e74c3c', edgecolor='white', alpha=0.85)

    ax.set_yticks(range(len(feat_ids)))
    ax.set_yticklabels([f"{fid} → {ec}" for fid, ec in zip(feat_ids, ecs)], fontsize=9)
    ax.set_xlabel('Top EC Fraction', fontsize=12)
    ax.set_title(f'Top {n_features} Most Discriminative Features ({ec_level})',
                 fontsize=14, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top_discriminative_features.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Feature analysis for ESM-2 TopK SAE")
    parser.add_argument("--dict_size", type=int, default=Config.DICT_SIZE)
    parser.add_argument("--top_k", type=int, default=Config.TOP_K)
    parser.add_argument("--max_proteins", type=int, default=Config.MAX_PROTEINS)
    parser.add_argument("--sae_dir", type=str, default=Config.SAE_DIR)
    parser.add_argument("--reprs_path", type=str, default=Config.REPRS_PATH)
    parser.add_argument("--metadata_path", type=str, default=Config.METADATA_PATH)
    parser.add_argument("--results_dir", type=str, default=Config.RESULTS_DIR)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    args = parser.parse_args()

    print("=" * 70)
    print("SAE FEATURE ANALYSIS — ESM-2 PER-RESIDUE")
    print("=" * 70)
    print(f"  SAE: dict={args.dict_size}, K={args.top_k}")
    print(f"  Max proteins: {args.max_proteins}")
    print(f"  Device: {args.device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SAE ──
    print("\n[1/5] Loading SAE...")
    sae = load_sae(args.sae_dir, args.dict_size, args.top_k, Config.LAYER_IDX, args.device)

    # ── Load data ──
    print("\n[2/5] Loading data...")
    reprs, metadata = load_data(args.reprs_path, args.metadata_path, args.max_proteins)

    # ── Extract feature activations ──
    print("\n[3/5] Extracting feature activations...")
    protein_features, feature_protein_map, feature_ec_counts, \
        feature_total_act, feature_residue_count = extract_feature_activations(
            sae, reprs, metadata, args.device
        )

    # Feature utilization
    n_active = sum(1 for f in range(args.dict_size) if feature_total_act.get(f, 0) > 0)
    n_dead = args.dict_size - n_active
    print(f"\n  Active features: {n_active}/{args.dict_size} ({100 * n_active / args.dict_size:.1f}%)")
    print(f"  Dead features: {n_dead}/{args.dict_size} ({100 * n_dead / args.dict_size:.1f}%)")

    # ── EC specificity ──
    print("\n[4/5] Computing EC specificity...")
    feature_specificity = compute_ec_specificity(
        feature_ec_counts, feature_total_act, feature_residue_count, args.dict_size
    )

    # ── Taxonomy ──
    print("\n[5/5] Classifying features...")
    taxonomy_results = {}
    summary_results = {}

    for level in range(1, 5):
        level_name = f'level_{level}'
        print(f"\n  ── EC {level_name} ──")
        taxonomy, summary = classify_features(
            feature_specificity, args.dict_size, ec_level=level_name
        )
        taxonomy_results[level_name] = taxonomy
        summary_results[level_name] = summary

    # ── Save results ──
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")

    # 1. Feature specificity (full)
    spec_path = results_dir / "feature_specificity.json"
    # Convert int keys to strings for JSON
    spec_json = {str(k): v for k, v in feature_specificity.items()}
    with open(spec_path, 'w') as f:
        json.dump(spec_json, f, indent=2)
    print(f"  ✓ Feature specificity: {spec_path}")

    # 2. Taxonomy summary
    tax_path = results_dir / "feature_taxonomy.json"
    tax_json = {
        'config': {
            'model': 'facebook/esm2_t6_8M_UR50D',
            'layer': Config.LAYER_IDX,
            'dict_size': args.dict_size,
            'top_k': args.top_k,
            'n_proteins': len(metadata),
        },
        'summary': summary_results,
        'taxonomy': {level: {cat: feats for cat, feats in tax.items()}
                     for level, tax in taxonomy_results.items()},
    }
    with open(tax_path, 'w') as f:
        json.dump(tax_json, f, indent=2)
    print(f"  ✓ Feature taxonomy: {tax_path}")

    # 3. Top discriminative features table
    disc_rows = []
    for feat_id in taxonomy_results['level_4']['DISCRIMINATIVE']:
        spec = feature_specificity[feat_id]
        for level in range(1, 5):
            ls = spec.get(f'level_{level}', {})
            disc_rows.append({
                'feature_id': feat_id,
                'ec_level': level,
                'top_ec': ls.get('top_ec', '?'),
                'top_ec_frac': ls.get('top_ec_frac', 0),
                'normalized_entropy': ls.get('normalized_entropy', 0),
                'n_ecs': ls.get('n_ecs', 0),
                'total_activation': spec.get('total_activation', 0),
                'total_residues': spec.get('total_residues', 0),
            })

    if disc_rows:
        disc_df = pd.DataFrame(disc_rows)
        disc_path = results_dir / "discriminative_features.csv"
        disc_df.to_csv(disc_path, index=False)
        print(f"  ✓ Discriminative features: {disc_path}")

    # 4. Per-protein feature summary (sampled)
    sample_proteins = protein_features[:min(100, len(protein_features))]
    protein_path = results_dir / "sample_protein_features.json"
    with open(protein_path, 'w') as f:
        json.dump(sample_proteins, f, indent=2)
    print(f"  ✓ Sample protein features: {protein_path}")

    # ── Final summary ──
    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")

    lvl4 = summary_results['level_4']
    print(f"\n  Feature Taxonomy (EC Level 4):")
    print(f"    DISCRIMINATIVE: {lvl4['DISCRIMINATIVE']['count']} ({lvl4['DISCRIMINATIVE']['pct']}%)")
    print(f"    GENERAL:        {lvl4['GENERAL']['count']} ({lvl4['GENERAL']['pct']}%)")
    print(f"    MIXED:          {lvl4['MIXED']['count']} ({lvl4['MIXED']['pct']}%)")
    print(f"    DEAD:           {lvl4['DEAD']['count']} ({lvl4['DEAD']['pct']}%)")

    print(f"\n  All results saved to: {results_dir}/")



if __name__ == "__main__":
    main()