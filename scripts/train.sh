#!/usr/bin/env bash
# Fine-tune a model on text-to-SQL with MLX LoRA
# All params can be overridden with environment variables.
#
# Default (0.8B, fits any Mac):
#   bash scripts/train.sh
#
# Try the 2B model (best results):
#   MODEL=mlx-community/Qwen3.5-2B-4bit-OptiQ bash scripts/train.sh
#
# Low-memory mode (16 GB Mac):
#   BATCH_SIZE=1 NUM_LAYERS=4 bash scripts/train.sh
#
# Fast experiment (100 iters):
#   ITERS=100 bash scripts/train.sh

set -euo pipefail

"MODEL="${MODEL:-mlx-community/Qwen3.5-0.8B-OptiQ-4bit}"
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters}"
ITERS="${ITERS:-600}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_LAYERS="${NUM_LAYERS:-16}"
LR="${LR:-1e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:---grad-checkpoint}"

mkdir -p "$ADAPTER_DIR"

echo "============================================"
echo "  MLX LoRA Fine-Tune: Text-to-SQL"
echo "============================================"
echo "  Model:          $MODEL"
echo "  Data:           $DATA_DIR"
echo "  Adapter:        $ADAPTER_DIR"
echo "  Iters:          $ITERS"
echo "  Batch size:     $BATCH_SIZE"
echo "  LoRA layers:    $NUM_LAYERS"
echo "  LR:             $LR"
echo "  Max seq len:    $MAX_SEQ_LEN"
echo "  Grad checkpoint: $([ -n "$GRAD_CHECKPOINT" ] && echo yes || echo no)"
echo "============================================"
echo ""

uv run python -m mlx_lm lora \
  --model "$MODEL" \
  --data "$DATA_DIR" \
  --train \
  --iters "$ITERS" \
  --batch-size "$BATCH_SIZE" \
  --num-layers "$NUM_LAYERS" \
  --learning-rate "$LR" \
  --max-seq-length "$MAX_SEQ_LEN" \
  $GRAD_CHECKPOINT \
  --adapter-path "$ADAPTER_DIR"

echo ""
echo "Training complete! Adapter saved to: $ADAPTER_DIR/"
echo "Run evaluation: uv run python scripts/evaluate.py"
