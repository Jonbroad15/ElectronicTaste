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
sudo resize2fs "${DATA_DISK}" || true
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
# First, ensure matching torchaudio version is installed to prevent ABI mismatch.
if python3 -c "import torch" &>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_VER=$(python3 -c "import torch; print(''.join(torch.__version__.split('+')[1:]) if '+' in torch.__version__ else '')")
    if [ -n "${CUDA_VER}" ]; then
        echo "Pre-installed PyTorch ${TORCH_VER}+${CUDA_VER} detected. Installing matching torchaudio..."
        pip install --quiet "torchaudio==${TORCH_VER}+${CUDA_VER}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_VER}"
    else
        echo "Pre-installed PyTorch ${TORCH_VER} detected. Installing matching torchaudio..."
        pip install --quiet "torchaudio==${TORCH_VER}"
    fi
fi

pip install -r requirements.txt --quiet

# ── Launch Preprocessing, MAM Pre-training & Fine-tuning ──────────────────────────────────

LOG_DIR="${MOUNT_POINT}/logs"
mkdir -p "${LOG_DIR}"
PREPROCESS_LOG="${LOG_DIR}/preprocess.log"
MAM_LOG="${LOG_DIR}/mam_pretrain.log"
CLF_LOG="${LOG_DIR}/classifier_train.log"

echo "=== Launching Preprocessing + MAM + fine-tuning in tmux session 'train' ==="
# Chains preprocessing, MAM pre-training, and classifier fine-tuning
tmux new-session -d -s train \
    "cd ${PROJECT_ROOT} && \
     echo '=== PHASE 0: Preprocessing Audio ===' && \
     python3 scripts/preprocess_audio.py \
         --splits splits.json \
         --source-dir ${MOUNT_POINT}/djmix/mixes \
         --target-dir ${MOUNT_POINT}/djmix/processed \
         2>&1 | tee ${PREPROCESS_LOG} && \
     echo '=== PHASE 1: Masked Audio Modeling (MAM) Pre-training ===' && \
     python3 -m src.training.train_mam \
         --data-dir ${MOUNT_POINT}/djmix/processed \
         --checkpoint-dir ${MOUNT_POINT}/models/mam_pretrain \
         --steps 50000 \
         --save-interval 5000 \
         --device cuda \
         2>&1 | tee ${MAM_LOG} && \
     echo '=== PHASE 2: Downstream Subgenre Classifier Fine-tuning ===' && \
     python3 -m src.training.train \
         --data-dir ${MOUNT_POINT}/djmix/processed \
         --checkpoint-dir ${MOUNT_POINT}/models/classifier \
         --mert-model ${MOUNT_POINT}/models/mam_pretrain/mert_adapted \
         --use-lora \
         --temporal-pooling chunk_5s \
         --loss contrastive \
         --epochs 30 \
         --batch-size 16 \
         --device cuda \
         2>&1 | tee ${CLF_LOG}"

echo ""
echo "Training pipeline launched."
echo "  Attach:             tmux attach -t train"
echo "  Monitor pre-train:  tail -f ${MAM_LOG}"
echo "  Monitor classifier: tail -f ${CLF_LOG}"
echo ""
echo "When complete, copy checkpoints and shut down from your local machine: (cleanup section)"
