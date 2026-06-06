#!/usr/bin/env bash
# STEP 1b — run this INSIDE the download VM (electronic-taste-download).
# Mounts the data disk, installs yt-dlp + ffmpeg, launches the Raveform download.
set -euo pipefail

DATA_DISK="/dev/disk/by-id/google-data"
MOUNT_POINT="/mnt/data"
PROJECT_ROOT="${HOME}/ElectronicTaste"

# ── Mount the 2 TB data disk ───────────────────────────────────────────────

echo "=== Mounting data disk ==="
if ! mountpoint -q "${MOUNT_POINT}"; then
    sudo mkdir -p "${MOUNT_POINT}"
    if ! sudo blkid "${DATA_DISK}" &>/dev/null; then
        sudo mkfs.ext4 -F "${DATA_DISK}"
    fi
    sudo mount "${DATA_DISK}" "${MOUNT_POINT}"
    echo "${DATA_DISK}  ${MOUNT_POINT}  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab
fi
sudo chmod 777 "${MOUNT_POINT}"
echo "Disk mounted at ${MOUNT_POINT} ($(df -h ${MOUNT_POINT} | tail -1 | awk '{print $4}') free)"

# ── Install dependencies (no PyTorch needed for download) ──────────────────

echo "=== Installing packages ==="
sudo apt-get update -q
sudo apt-get install -y -q ffmpeg tmux python3-pip

pip3 install --quiet --upgrade pip
pip3 install --quiet yt-dlp

# ── Launch download in tmux ─────────────────────────────────────────────────

OUTPUT_DIR="${MOUNT_POINT}/djmix"
LOG="${MOUNT_POINT}/download_raveform.log"

echo "=== Launching Raveform download in tmux session 'download' ==="
tmux new-session -d -s download \
    "cd ${PROJECT_ROOT} && python3 scripts/download_raveform.py \
        --output-dir ${OUTPUT_DIR} \
        --workers 8 \
        --max-per-class 600 \
        2>&1 | tee ${LOG}"

echo ""
echo "Download launched."
echo "  Attach:        tmux attach -t download"
echo "  Monitor log:   tail -f ${LOG}"
echo "  Check counts:  ls ${OUTPUT_DIR}"
echo ""
echo "When complete, return to your local machine and run:"
echo "  bash scripts/gcp_provision_training.sh"
