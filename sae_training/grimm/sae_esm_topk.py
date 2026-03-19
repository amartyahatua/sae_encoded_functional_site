import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Model dimensions (ESM-2 8M, Layer 5)
    D_MODEL = 320
    LAYER_IDX = 5

    # SAE configurations to sweep
    DICT_SIZES = [4096, 8192, 16384]

    TOP_KS = [64, 128, 256]

    # Training hyperparameters
    LR = 3e-4
    BATCH_SIZE = 4096
    EPOCHS = 10
    WARMUP_EPOCHS = 1  # Use MSE-only for first epoch

    # Data
    MAX_RESIDUES = 5000000
    NULL_TEST_RESIDUES = 500000

    # Paths
    REPRS_PATH = "../../layer_selection/data/residue_representations/residue_reprs_layer5.npy"
    METADATA_PATH = "../../layer_selection/data/residue_representations/metadata_layer5.json"
    SAVE_DIR = "../models/sae_topk"
    RESULTS_DIR = "../results/sae_training_topk"

    DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


# ============================================================
# TopK SAE MODEL
# ============================================================

class TopKSAE(nn.Module):
    """
    Sparse Autoencoder with TopK activation.

    Instead of ReLU + L1 penalty (where sparsity depends on L1 tuning),
    TopK directly keeps only the K largest activations and zeros out the rest.

    Architecture:
        Encode:  x → W_enc @ x + b_enc → TopK → sparse features z
        Decode:  z → W_dec @ z → reconstruction

    The decoder weights are kept unit-norm to prevent feature collapse.
    """

    def __init__(self, d_model, dict_size, top_k=128):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.top_k = top_k

        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

        # Initialize decoder with unit-norm columns
        nn.init.normal_(self.decoder.weight, std=0.02)
        with torch.no_grad():
            self.normalize_decoder()

    def encode(self, x):
        """Encode with TopK sparsity."""
        pre_acts = self.encoder(x)  # (batch, dict_size)

        # TopK: keep only top K values, zero out the rest
        topk_values, topk_indices = torch.topk(pre_acts, self.top_k, dim=-1)

        # Apply ReLU to top-K values (only keep positive activations)
        topk_values = F.relu(topk_values)

        # Scatter back to full dict_size
        sparse_acts = torch.zeros_like(pre_acts)
        sparse_acts.scatter_(-1, topk_indices, topk_values)

        return sparse_acts

    def decode(self, z):
        """Decode sparse features back to input space."""
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    @torch.no_grad()
    def normalize_decoder(self):
        """Keep decoder columns unit-norm to prevent feature collapse."""
        W = self.decoder.weight
        self.decoder.weight.copy_(W / W.norm(dim=0, keepdim=True).clamp(min=1e-6))


# ============================================================
# DATA LOADING
# ============================================================

def load_residue_data(reprs_path, max_residues=5_000_000, seed=42):
    """Load per-residue representations with memory-mapping and subsampling."""
    print(f"\nLoading representations from: {reprs_path}")

    mmap = np.load(reprs_path, mmap_mode='r')
    total_residues, d_model = mmap.shape
    print(f"  Total residues: {total_residues:,}")
    print(f"  Hidden dim: {d_model}")

    if max_residues >= total_residues:
        data = np.array(mmap)
    else:
        print(f"  Subsampling {max_residues:,} residues...")
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_residues, size=max_residues, replace=False)
        indices.sort()
        data = mmap[indices].copy()

    print(f"  Loaded: {data.shape}, {data.nbytes / 1e9:.2f} GB")
    return torch.from_numpy(data).float()


# ============================================================
# TRAINING
# ============================================================

def train_sae_topk(activations, d_model, dict_size, top_k, device, config):
    """Train TopK SAE."""
    print(f"\n  Training TopK SAE: d_model={d_model}, dict={dict_size}, K={top_k}")
    print(f"  Sparsity: {top_k}/{dict_size} = {100 * top_k / dict_size:.1f}%")
    print(f"  Residues: {activations.shape[0]:,}")

    sae = TopKSAE(d_model, dict_size, top_k=top_k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=config.LR)

    loader = DataLoader(
        TensorDataset(activations),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=(device == "cpu"),
    )

    for epoch in range(config.EPOCHS):
        sae.train()
        epoch_loss = 0
        n_batches = 0

        for (x,) in loader:
            x = x.to(device)

            recon, acts = sae(x)

            # Pure reconstruction loss — no L1 needed, TopK handles sparsity
            loss = F.mse_loss(recon, x)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 2 == 0 or epoch == 0:
            with torch.no_grad():
                n_sample = min(8192, activations.shape[0])
                x_s = activations[:n_sample].to(device)
                recon_s, acts_s = sae(x_s)
                rel_mse = F.mse_loss(recon_s, x_s) / x_s.var()
                cos = F.cosine_similarity(x_s, recon_s, dim=-1).mean()
                # Verify actual sparsity
                actual_l0 = (acts_s > 0).float().sum(dim=-1).mean()
                del x_s, recon_s, acts_s

            print(f"    Epoch {epoch + 1}/{config.EPOCHS}: "
                  f"MSE={avg_loss:.6f}, RelMSE={rel_mse:.4f}, "
                  f"Cos={cos:.4f}, L0={actual_l0:.0f}/{dict_size}")

    return sae


# ============================================================
# NULL INTERVENTION TEST
# ============================================================

@torch.no_grad()
def null_intervention_test(sae, test_data, device, max_residues=500_000):
    """Null test: encode → decode → re-encode. ICC < 0.05 = PASS."""
    sae.eval()

    n = min(max_residues, test_data.shape[0])
    data = test_data[:n]
    loader = DataLoader(TensorDataset(data), batch_size=4096, shuffle=False)

    icc_values = []
    cosine_sims = []
    recon_mses = []
    l0_values = []

    for (x,) in loader:
        x = x.to(device)

        z_orig = sae.encode(x)
        recon = sae.decode(z_orig)
        z_re = sae.encode(recon)

        # ICC
        change = (z_re - z_orig).abs().mean().item()
        magnitude = z_orig.abs().mean().item()
        icc_values.append(change / (magnitude + 1e-10))

        # Cosine
        cosine_sims.append(F.cosine_similarity(x, recon, dim=-1).mean().item())

        # MSE
        recon_mses.append(F.mse_loss(recon, x).item())

        # L0
        l0_values.append((z_orig > 0).float().sum(dim=-1).mean().item())

    results = {
        "null_icc": float(np.mean(icc_values)),
        "null_cosine_similarity": float(np.mean(cosine_sims)),
        "null_reconstruction_mse": float(np.mean(recon_mses)),
        "null_l0_mean": float(np.mean(l0_values)),
        "null_l0_frac": float(np.mean(l0_values)) / sae.dict_size,
    }

    results["icc_pass"] = results["null_icc"] < 0.05
    results["recon_pass"] = results["null_reconstruction_mse"] < 0.1
    results["overall_pass"] = results["icc_pass"] and results["recon_pass"]

    print(f"    NULL TEST: ICC={results['null_icc']:.4f} {'Yes' if results['icc_pass'] else 'No'}  "
          f"MSE={results['null_reconstruction_mse']:.6f} {'Yes' if results['recon_pass'] else 'No'}  "
          f"Cos={results['null_cosine_similarity']:.4f}  "
          f"L0={results['null_l0_mean']:.0f}/{sae.dict_size}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train TopK SAEs on ESM-2 residue representations")
    parser.add_argument("--reprs_path", type=str, default=Config.REPRS_PATH)
    parser.add_argument("--max_residues", type=int, default=Config.MAX_RESIDUES)
    parser.add_argument("--dict_sizes", type=int, nargs='+', default=Config.DICT_SIZES)
    parser.add_argument("--top_ks", type=int, nargs='+', default=Config.TOP_KS)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--save_dir", type=str, default=Config.SAVE_DIR)
    parser.add_argument("--results_dir", type=str, default=Config.RESULTS_DIR)
    args = parser.parse_args()

    # Override config
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size

    total_configs = len(args.dict_sizes) * len(args.top_ks)

    print("=" * 70)
    print("TopK SAE TRAINING — ESM-2 PER-RESIDUE REPRESENTATIONS")
    print("=" * 70)
    print(f"  d_model: {Config.D_MODEL}")
    print(f"  Layer: {Config.LAYER_IDX}")
    print(f"  Dict sizes: {args.dict_sizes}")
    print(f"  Top-K values: {args.top_ks}")
    print(f"  Total configs: {total_configs}")
    print(f"  Max residues: {args.max_residues:,}")
    print(f"  Device: {Config.DEVICE}")

    # Load data
    all_data = load_residue_data(args.reprs_path, max_residues=args.max_residues)

    n_train = int(0.9 * len(all_data))
    train_data = all_data[:n_train]
    test_data = all_data[n_train:]
    print(f"\n  Train: {n_train:,} residues")
    print(f"  Test: {len(test_data):,} residues")

    # Setup directories
    save_dir = Path(args.save_dir)
    results_dir = Path(args.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    current = 0

    for dict_size in args.dict_sizes:
        for top_k in args.top_ks:
            current += 1

            # Skip invalid combos (K > dict_size)
            if top_k >= dict_size:
                print(f"\n  Skipping K={top_k} ≥ dict={dict_size}")
                continue

            print(f"\n{'#' * 70}")
            print(f"[{current}/{total_configs}] Dict={dict_size}, K={top_k} "
                  f"({100 * top_k / dict_size:.1f}% active)")
            print(f"{'#' * 70}")

            ckpt_name = f"esm2_8M_layer{Config.LAYER_IDX}_dict{dict_size}_topk{top_k}.pt"
            checkpoint_path = save_dir / ckpt_name

            # Skip if exists
            if checkpoint_path.exists():
                print(f"  Exists: {checkpoint_path}, skipping...")
                continue

            # Train
            sae = train_sae_topk(
                train_data,
                d_model=Config.D_MODEL,
                dict_size=dict_size,
                top_k=top_k,
                device=Config.DEVICE,
                config=Config,
            )

            # Save
            torch.save({
                "model": "facebook/esm2_t6_8M_UR50D",
                "layer": Config.LAYER_IDX,
                "d_model": Config.D_MODEL,
                "dict_size": dict_size,
                "top_k": top_k,
                "activation": "topk",
                "representation_type": "per_residue",
                "n_train_residues": n_train,
                "state_dict": sae.state_dict(),
            }, checkpoint_path)
            print(f"  ✓ Saved: {checkpoint_path}")

            # Null test
            null_results = null_intervention_test(sae, test_data, Config.DEVICE)

            result = {
                "dict_size": dict_size,
                "top_k": top_k,
                "sparsity_pct": 100.0 * top_k / dict_size,
                "layer": Config.LAYER_IDX,
                **null_results,
            }
            all_results.append(result)

            # Save individual
            with open(save_dir / f"null_test_dict{dict_size}_topk{top_k}.json", 'w') as f:
                json.dump(result, f, indent=2)

            # Incremental progress
            pd.DataFrame(all_results).to_csv(results_dir / "progress.csv", index=False)

            del sae
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'=' * 70}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        final_path = results_dir / "topk_sae_results.csv"
        results_df.to_csv(final_path, index=False)

        print(f"\n  {'Dict':>6s}  {'K':>4s}  {'Sparsity':>8s}  {'ICC':>7s}  {'MSE':>10s}  "
              f"{'Cos':>6s}  {'L0':>6s}  {'Pass':>4s}")
        print(f"  {'-' * 65}")

        for _, r in results_df.iterrows():
            print(f"  {int(r['dict_size']):>6d}  {int(r['top_k']):>4d}  "
                  f"{r['sparsity_pct']:>7.1f}%  {r['null_icc']:>7.4f}  "
                  f"{r['null_reconstruction_mse']:>10.6f}  {r['null_cosine_similarity']:>6.4f}  "
                  f"{r['null_l0_mean']:>6.0f}")

        print(f"\n  Results: {final_path}")

        # Find optimal config
        passing = results_df[results_df['overall_pass'] == True]
        if len(passing) > 0:
            best = passing.loc[passing['null_reconstruction_mse'].idxmin()]
            print(f"\n  Best passing config: Dict={int(best['dict_size'])}, K={int(best['top_k'])}")
            print(f"    ICC={best['null_icc']:.4f}, MSE={best['null_reconstruction_mse']:.6f}")
        else:
            print(f"\n  No configs passed null test")

    print(f"\n  SAEs saved to: {save_dir}/")


if __name__ == "__main__":
    main()