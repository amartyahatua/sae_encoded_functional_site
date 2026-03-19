#!/bin/bash
# ── Mode ──────────────────────────────────────────────────────
MODE=${MODE:-"probe"}           # "probe" or "extract"

# ── Model ─────────────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-"facebook/esm2_t6_8M_UR50D"}

# ── Data ──────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-"../../data/grimm"}
SPLIT=${SPLIT:-"train"}
DATASET_SIZE=${DATASET_SIZE:-185419}
MIN_CLASS_SIZE=${MIN_CLASS_SIZE:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-512}

# ── Extract mode ──────────────────────────────────────────────
LAYER=${LAYER:-5}

# ── Output ────────────────────────────────────────────────────
SAVE_DIR=${SAVE_DIR:-"../results/"}
EXTRACT_DIR=${EXTRACT_DIR:-"../data/residue_representations"}

# ── Device ────────────────────────────────────────────────────
DEVICE=${DEVICE:-"cuda"}          # "cuda" or "cpu"

# ══════════════════════════════════════════════════════════════
echo "=================================================="
echo " ESM-2 Pipeline"
echo "=================================================="
echo "  MODE        : $MODE"
echo "  MODEL       : $MODEL_NAME"
echo "  DATA_PATH   : $DATA_PATH"
echo "  SPLIT       : $SPLIT"
echo "  DATASET_SIZE: $DATASET_SIZE"
echo "  MAX_SEQ_LEN : $MAX_SEQ_LEN"
echo "  LAYER       : $LAYER  (extract mode only)"
echo "  DEVICE      : $DEVICE"
echo "  SAVE_DIR    : $SAVE_DIR"
echo "  EXTRACT_DIR : $EXTRACT_DIR"
echo "=================================================="

python esm2_pipeline.py \
    --mode         "$MODE"           \
    --model_name   "$MODEL_NAME"     \
    --data_path    "$DATA_PATH"      \
    --split        "$SPLIT"          \
    --dataset_size "$DATASET_SIZE"   \
    --min_class_size "$MIN_CLASS_SIZE" \
    --max_seq_len  "$MAX_SEQ_LEN"    \
    --layer        "$LAYER"          \
    --save_dir     "$SAVE_DIR"       \
    --extract_dir  "$EXTRACT_DIR"    \
    --device       "$DEVICE"
