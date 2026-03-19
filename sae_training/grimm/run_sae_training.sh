#!/bin/bash

# ── Data ──────────────────────────────────────────────────────
REPRS_PATH=${REPRS_PATH:-"../../layer_selection/data/residue_representations/residue_reprs_layer5.npy"}
MAX_RESIDUES=${MAX_RESIDUES:5000000}

# ── SAE sweep config ──────────────────────────────────────────
# Space-separated lists — passed as nargs='+' to argparse
DICT_SIZES=${DICT_SIZES:-"4096 8192 16384"}
TOP_KS=${TOP_KS:"64 128 256"}

# ── Training hyperparameters ──────────────────────────────────
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:4096}

# ── Output ────────────────────────────────────────────────────
SAVE_DIR=${SAVE_DIR:-"../models/sae_topk"}
RESULTS_DIR=${RESULTS_DIR:-"../results/sae_training_topk"}

# ══════════════════════════════════════════════════════════════
echo "=================================================="
echo " TopK SAE Training"
echo "=================================================="
echo "  REPRS_PATH  : $REPRS_PATH"
echo "  MAX_RESIDUES: $MAX_RESIDUES"
echo "  DICT_SIZES  : $DICT_SIZES"
echo "  TOP_KS      : $TOP_KS"
echo "  EPOCHS      : $EPOCHS"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  SAVE_DIR    : $SAVE_DIR"
echo "  RESULTS_DIR : $RESULTS_DIR"
echo "=================================================="

python train_sae.py \
    --reprs_path   "$REPRS_PATH"    \
    --max_residues "$MAX_RESIDUES"  \
    --dict_sizes   $DICT_SIZES      \
    --top_ks       $TOP_KS          \
    --epochs       "$EPOCHS"        \
    --batch_size   "$BATCH_SIZE"    \
    --save_dir     "$SAVE_DIR"      \
    --results_dir  "$RESULTS_DIR"
