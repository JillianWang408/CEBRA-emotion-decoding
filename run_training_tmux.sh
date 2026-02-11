#!/bin/bash
# Script to run encoding training in tmux session

cd /Users/wangzihan/Desktop/Projects/xCEBRA

# Activate virtual environment
source cebra_env_arm/bin/activate

# Create tmux session and run training
# Default: uses latent-dim=10 (based on scree plot), includes test patient 28 (EC304)
# Set FORCE_RETRAIN=1 to override existing checkpoints and retrain from scratch
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

if [ "$FORCE_RETRAIN" == "1" ]; then
    echo "⚠️  FORCE RETRAIN enabled - will override existing checkpoints"
    RETRAIN_FLAG="--force-retrain"
else
    echo "ℹ️  Using existing checkpoints if available (set FORCE_RETRAIN=1 to override)"
    RETRAIN_FLAG=""
fi

tmux new-session -d -s encoding_training \
    "source cebra_env_arm/bin/activate && \
     cd /Users/wangzihan/Desktop/Projects/xCEBRA && \
     python src/patient_aggreagation_encoding_finetune.py \
         --test-patient-id 28 \
         $RETRAIN_FLAG \
         2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log"

echo "Training started in tmux session 'encoding_training'"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t encoding_training"
echo ""
echo "To detach (while inside tmux):"
echo "  Press Ctrl+B, then D"
echo ""
echo "To list all sessions:"
echo "  tmux ls"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t encoding_training"

