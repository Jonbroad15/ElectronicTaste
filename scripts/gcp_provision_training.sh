#!/usr/bin/env bash
# STEP 2 of 2 — delete CPU VM, create GPU VM for training, attach 2 TB data disk.
# Run this LOCALLY (needs gcloud authenticated).
set -euo pipefail

PROJECT="project-58e658a7-9bc6-41eb-876"
ZONE="us-central1-a"
DOWNLOAD_INSTANCE="electronic-taste-download"
TRAIN_INSTANCE="electronic-taste-train"
DATA_DISK="electronic-taste-raveform-data"

echo "=== Deleting CPU download VM (saving data disk) ==="
if gcloud compute instances describe "${DOWNLOAD_INSTANCE}" \
        --project="${PROJECT}" --zone="${ZONE}" &>/dev/null; then
    gcloud compute instances delete "${DOWNLOAD_INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --quiet
else
    echo "VM ${DOWNLOAD_INSTANCE} does not exist — skipping delete."
fi

if gcloud compute instances describe "${TRAIN_INSTANCE}" \
        --project="${PROJECT}" --zone="${ZONE}" &>/dev/null; then
    echo "VM ${TRAIN_INSTANCE} already exists — skipping creation."
else
    echo "Resizing data disk to 4TB to ensure sufficient space for processed chunks..."
    gcloud compute disks resize "${DATA_DISK}" \
        --size=4000GB \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --quiet || true

    # Image spec from raveform_dataset_plan.md:
    # pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260518 from deeplearning-platform-release
    gcloud compute instances create "${TRAIN_INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="g2-standard-8" \
        --maintenance-policy="TERMINATE" \
        --accelerator="type=nvidia-l4,count=1" \
        --image-project="deeplearning-platform-release" \
        --image="pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260518" \
        --boot-disk-size="100GB" \
        --boot-disk-type="pd-ssd" \
        --disk="name=${DATA_DISK},device-name=data,auto-delete=no"
fi

echo ""
echo "GPU VM created. Waiting ~60 s for boot …"
sleep 60

echo "=== Transferring code ==="
rsync -avz \
    --exclude="/.git" \
    --exclude="/data" \
    --exclude="/models" \
    --exclude="__pycache__" \
    --exclude="/.venv" \
    -e "bash -c 'instance=\$1; shift; exec gcloud compute ssh \"\$instance\" --project=\"${PROJECT}\" --zone=\"${ZONE}\" -- \"\$@\"' --" \
    "$(git rev-parse --show-toplevel)/" \
    "${TRAIN_INSTANCE}:~/ElectronicTaste"

echo ""
echo "=== Next step: SSH in and run the training setup ==="
echo "  gcloud compute ssh ${TRAIN_INSTANCE} --zone=${ZONE} --project=${PROJECT}"
echo "  bash ~/ElectronicTaste/scripts/gcp_setup_training.sh"
