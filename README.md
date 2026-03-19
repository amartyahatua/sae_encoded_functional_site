# ESM-2 SAE Pipeline

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Layer Selection & Representation Extraction

```bash
bash layer_selection/grimm/run_esm2.sh
```

Runs ESM-2 probing on the GRIMM dataset.
Edit the variables at the top of the script to change model, layer, paths, or mode (`probe` / `extract`).

---

## Step 2 — Train TopK SAE

```bash
bash sae_training/grimm/run_sae_training.sh
```

Trains TopK Sparse Autoencoders on the residue representations produced in Step 1.
Edit the variables at the top of the script to change dictionary sizes, Top-K values, epochs, or paths.
