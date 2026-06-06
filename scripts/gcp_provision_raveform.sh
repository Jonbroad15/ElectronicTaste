#!/usr/bin/env bash
# STEP 1 of 2 — provision the 2 TB data disk + lightweight CPU VM for downloading.
# Run this LOCALLY (needs gcloud authenticated).
#
# After the download completes, run scripts/gcp_provision_training.sh (step 2)
# to swap in the GPU VM for MAM pre-training + fine-tuning.
set -euo pipefail

PROJECT="project-58e658a7-9bc6-41eb-876"
ZONE="us-central1-a"
DOWNLOAD_INSTANCE="electronic-taste-download"
DATA_DISK="electronic-taste-raveform-data"
DISK_SIZE="2000GB"

echo "=== Creating 2 TB data disk (if it doesn't exist) ==="
if ! gcloud compute disks describe "${DATA_DISK}" \
        --project="${PROJECT}" --zone="${ZONE}" &>/dev/null; then
    gcloud compute disks create "${DATA_DISK}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --size="${DISK_SIZE}" \
        --type="pd-standard"   # pd-standard avoids the 250 GB SSD quota cap
else
    echo "Disk ${DATA_DISK} already exists — skipping."
fi

echo "=== Creating CPU-only download VM (n2-standard-4, no GPU) ==="
gcloud compute instances create "${DOWNLOAD_INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="n2-standard-4" \
    --image-project="debian-cloud" \
    --image-family="debian-12" \
    --boot-disk-size="20GB" \
    --boot-disk-type="pd-standard" \
    --disk="name=${DATA_DISK},device-name=data,auto-delete=no"

echo ""
echo "VM created.  Waiting ~60 s for boot …"
sleep 60

echo "=== Transferring code ==="
gcloud compute scp --recurse \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --exclude=".git,data,models,__pycache__,.venv" \
    "$(git rev-parse --show-toplevel)/." \
    "${DOWNLOAD_INSTANCE}:~/ElectronicTaste"

echo ""
echo "=== Next step: SSH in and run the download setup ==="
echo "  gcloud compute ssh ${DOWNLOAD_INSTANCE} --zone=${ZONE} --project=${PROJECT}"
echo "  bash ~/ElectronicTaste/scripts/gcp_setup_download.sh"
echo ""
echo "When the download finishes, run step 2 from your local machine:"
echo "  bash scripts/gcp_provision_training.sh"
