#!/usr/bin/env bash
# STEP 1b — run this INSIDE the download VM (electronic-taste-download).
# Mounts the data disk, installs yt-dlp + ffmpeg, annotates any existing mixes,
# then launches the expanded Raveform download in a tmux session.
set -euo pipefail

DATA_DISK="/dev/disk/by-id/google-data"
MOUNT_POINT="/mnt/data"
PROJECT_ROOT="${HOME}/ElectronicTaste"

# ---------------------------------------------------------------------------
# Mount the 2 TB data disk
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
echo "=== Installing packages ==="
sudo apt-get update -q
sudo apt-get install -y -q ffmpeg tmux python3-pip

pip3 install --quiet --break-system-packages --upgrade pip
pip3 install --quiet --break-system-packages yt-dlp

# ---------------------------------------------------------------------------
# Annotate existing mixes (synchronous — must complete before download starts)
# ---------------------------------------------------------------------------
OUTPUT_DIR="${MOUNT_POINT}/djmix"
MANIFEST="${OUTPUT_DIR}/djmix_manifest_raw.json"

echo "=== Annotating existing mixes ==="
if [[ -f "${MANIFEST}" ]]; then
    echo "Found manifest at ${MANIFEST} — running annotate_raveform.py ..."
    python3 "${PROJECT_ROOT}/scripts/annotate_raveform.py" \
        --djmix-dir "${OUTPUT_DIR}" \
        --manifest "${MANIFEST}" \
        --output "${OUTPUT_DIR}/labels.json"
    echo "Annotation complete."
else
    echo "No manifest found at ${MANIFEST} — skipping annotation step."
fi

# ---------------------------------------------------------------------------
# Launch expanded download in a tmux session (background)
# ---------------------------------------------------------------------------
LOG="${MOUNT_POINT}/download_raveform_expanded.log"

echo "=== Launching expanded Raveform download in tmux session 'download' ==="
tmux new-session -d -s download \
    "cd ${PROJECT_ROOT} && python3 scripts/download_raveform.py \
        --output-dir ${OUTPUT_DIR} \
        --workers 8 \
        --manifest-cache ${OUTPUT_DIR}/djmix_manifest_raw.json \
        --labels ${OUTPUT_DIR}/labels.json \
        2>&1 | tee ${LOG}"

echo ""
echo "=========================================================="
echo "  Expanded download launched in tmux session 'download'."
echo "=========================================================="
echo ""
echo "  Attach to session:     tmux attach -t download"
echo "  Monitor log:           tail -f ${LOG}"
echo "  Check downloaded dirs: ls ${OUTPUT_DIR}"
echo ""
echo "Once the download finishes, run create_splits.py:"
echo ""
echo "  python3 ${PROJECT_ROOT}/scripts/create_splits.py \\"
echo "      --manifest ${OUTPUT_DIR}/djmix_manifest_raw.json \\"
echo "      --labels   ${OUTPUT_DIR}/labels.json \\"
echo "      --output-dir ${OUTPUT_DIR}/splits"
echo ""
echo "When fully done, delete this VM from your local machine:"
echo ""
echo "  gcloud compute instances delete electronic-taste-download --zone=<ZONE> --quiet"
echo ""
