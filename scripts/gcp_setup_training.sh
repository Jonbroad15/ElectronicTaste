#!/usr/bin/env bash
# STEP 2b — run this INSIDE the training VM (electronic-taste-train).
# Mounts the data disk, installs PyTorch dependencies, launches MAM + fine-tuning.
set -euo pipefail

DATA_DISK="/dev/disk/by-id/google-data"
MOUNT_POINT="/mnt/data"
PROJECT_ROOT="${HOME}/ElectronicTaste"

# ── Mount the 2 TB data disk ───────────────────────────────────────────────

echo "=== Mounting data disk ==="
if ! mountpoint -q "${MOUNT_POINT}"; then
    sudo mkdir -p "${MOUNT_POINT}"
    sudo mount "${DATA_DISK}" "${MOUNT_POINT}"
fi
sudo chmod 777 "${MOUNT_POINT}"
echo "Disk mounted at ${MOUNT_POINT} ($(df -h ${MOUNT_POINT} | tail -1 | awk '{print $4}') free)"

# ── Check data ─────────────────────────────────────────────────────────────

if [ ! -d "${MOUNT_POINT}/djmix" ]; then
    echo "ERROR: ${MOUNT_POINT}/djmix directory not found. Did the download fail?"
    exit 1
fi

echo "Found downloaded classes:"
ls -lh "${MOUNT_POINT}/djmix"

# ── Install dependencies ───────────────────────────────────────────────────

echo "=== Installing python dependencies ==="
cd "${PROJECT_ROOT}"
# The deeplearning image already has PyTorch + CUDA. We just need other packages.
pip install -r requirements.txt --quiet

# ── Launch MAM Pre-training & Fine-tuning ──────────────────────────────────

LOG_DIR="${MOUNT_POINT}/logs"
mkdir -p "${LOG_DIR}"
MAM_LOG="${LOG_DIR}/mam_pretrain.log"

echo "=== Launching MAM pre-training in tmux session 'train' ==="
# Pre-training and fine-tuning command (adjust flags as needed for your pipeline)
tmux new-session -d -s train \
    "cd ${PROJECT_ROOT} && python3 -m src.training.train \
        --data-dir ${MOUNT_POINT}/djmix \
        --checkpoint-dir ${MOUNT_POINT}/models \
        --epochs 10 \
        --batch-size 16 \
        --device cuda \
        2>&1 | tee ${MAM_LOG}"

echo ""
echo "Training launched."
echo "  Attach:        tmux attach -t train"
echo "  Monitor log:   tail -f ${MAM_LOG}"
echo ""
echo "When complete, copy checkpoints and shut down from your local machine:"
echo "  bash scripts/gcp_provision_training.sh (cleanup section)"
