#!/bin/bash

# CONFIGURABLE VALUES
PROJECT_ID="your-gcp-project-id"
BUCKET_NAME="your-bucket-name"
BQ_DATASET="crowe_ml_pipeline"
BQ_TABLE="metrics_log"
LOCAL_METRICS="logs/metrics.jsonl"
GCS_PATH="gs://${BUCKET_NAME}/ml_metrics/metrics.jsonl"

# 1. Enable BigQuery API (idempotent)
gcloud services enable bigquery.googleapis.com --project=${PROJECT_ID}

# 2. Create BigQuery dataset (if not exists)
bq --project_id=${PROJECT_ID} ls ${BQ_DATASET} >/dev/null 2>&1 || bq --project_id=${PROJECT_ID} mk --dataset ${BQ_DATASET}

# 3. Create table schema (if not exists)
bq show --project_id=${PROJECT_ID} ${BQ_DATASET}.${BQ_TABLE} >/dev/null 2>&1 || bq mk --project_id=${PROJECT_ID} --table ${BQ_DATASET}.${BQ_TABLE} dataset:STRING,n_train:INTEGER,n_test:INTEGER,rmse:FLOAT,r2:FLOAT,timestamp:TIMESTAMP

# 4. Upload to GCS
gsutil cp ${LOCAL_METRICS} ${GCS_PATH}

# 5. Load into BigQuery
bq load --project_id=${PROJECT_ID} --source_format=NEWLINE_DELIMITED_JSON ${BQ_DATASET}.${BQ_TABLE} ${GCS_PATH}

echo "✅ Metrics uploaded and loaded into BigQuery: ${BQ_DATASET}.${BQ_TABLE}"
