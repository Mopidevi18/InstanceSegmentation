#!/bin/bash
set -euo pipefail

REGION="us-central1"
DISPLAY_NAME="maskrcnn-training-job"

# Launch the training job and extract the job ID
echo "Launching custom training job..."
JOB_ID=$(gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$DISPLAY_NAME" \
  --config=MLOps/custom-job.yaml \
  --format="value(name)")

echo "Job started: $JOB_ID"

# Poll job status
echo "Waiting for training job to complete..."
while true; do
  STATUS=$(gcloud ai custom-jobs describe "$JOB_ID" --region="$REGION" --format="value(state)")
  echo "  ➤ Current status: $STATUS"
  if [[ "$STATUS" == "JOB_STATE_SUCCEEDED" ]]; then
    echo "Training job succeeded!"
    break
  elif [[ "$STATUS" == "JOB_STATE_FAILED" || "$STATUS" == "JOB_STATE_CANCELLED" ]]; then
    echo "Training job failed or cancelled!"
    exit 1
  fi
  sleep 60  # Check every 30 seconds
done

# Proceed to deploy webapp
echo "Deploying webapp to Cloud Run..."
gcloud run deploy segwaste-webapp \
   --image us-central1-docker.pkg.dev/instancesegmentation-456922/segwaste-webapp/segwaste \
   --region "$REGION" \
   --platform managed \
   --memory=8Gi --cpu 2 \
   --allow-unauthenticated