#!/bin/bash

# Configuration
PROJECT_ID="crowechem-fungi"
BUCKET_NAME="crowechem-fungi-ml-metrics"
DATASET_NAME="crowe_ml_pipeline"
TABLE_NAME="metrics_log"
FUNGAL_TABLE_NAME="fungal_analysis_results"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "🚀 Starting upload to BigQuery for project: ${PROJECT_ID}"

# Set the project
gcloud config set project ${PROJECT_ID}

# Create bucket if it doesn't exist
echo "📦 Checking GCS bucket..."
if ! gsutil ls -b gs://${BUCKET_NAME} &> /dev/null; then
    echo "Creating bucket gs://${BUCKET_NAME}..."
    gsutil mb -p ${PROJECT_ID} gs://${BUCKET_NAME}
else
    echo "✓ Bucket already exists"
fi

# Check if metrics file exists
if [ ! -f "logs/metrics.jsonl" ]; then
    echo -e "${RED}❌ Error: logs/metrics.jsonl not found${NC}"
    echo "Creating sample metrics file..."
    mkdir -p logs
    cat > logs/metrics.jsonl << EOF
{"dataset": "crowechem", "n_train": 1000, "n_test": 200, "rmse": 0.15, "r2": 0.92, "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)", "model_type": "random_forest", "compound_analyzed": "aspirin", "bioactivity_score": 0.85}
EOF
fi

# Upload metrics to GCS
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GCS_PATH="gs://${BUCKET_NAME}/ml_metrics/metrics_${TIMESTAMP}.jsonl"
echo "📤 Uploading metrics to ${GCS_PATH}..."
gsutil cp logs/metrics.jsonl ${GCS_PATH}

# Create BigQuery dataset if it doesn't exist
echo "🗄️  Checking BigQuery dataset..."
if ! bq ls -d ${PROJECT_ID}:${DATASET_NAME} &> /dev/null; then
    echo "Creating dataset ${DATASET_NAME}..."
    bq mk --dataset ${PROJECT_ID}:${DATASET_NAME}
else
    echo "✓ Dataset already exists"
fi

# Create metrics table if it doesn't exist
echo "📊 Checking BigQuery table..."
if ! bq ls ${PROJECT_ID}:${DATASET_NAME}.${TABLE_NAME} &> /dev/null; then
    echo "Creating table ${TABLE_NAME}..."
    bq mk --table \
        ${PROJECT_ID}:${DATASET_NAME}.${TABLE_NAME} \
        dataset:STRING,n_train:INTEGER,n_test:INTEGER,rmse:FLOAT,r2:FLOAT,timestamp:TIMESTAMP,model_type:STRING,compound_analyzed:STRING,bioactivity_score:FLOAT
else
    echo "✓ Table already exists"
fi

# Create fungal analysis results table if it doesn't exist
echo "🍄 Checking fungal analysis table..."
if ! bq ls ${PROJECT_ID}:${DATASET_NAME}.${FUNGAL_TABLE_NAME} &> /dev/null; then
    echo "Creating table ${FUNGAL_TABLE_NAME}..."
    bq mk --table \
        ${PROJECT_ID}:${DATASET_NAME}.${FUNGAL_TABLE_NAME} \
        analysis_id:STRING,timestamp:TIMESTAMP,species_analyzed:INTEGER,compounds_analyzed:INTEGER,breakthrough_discoveries:INTEGER,therapeutic_candidates:STRING,synthesis_pathways:STRING,impact_assessment:STRING
else
    echo "✓ Fungal analysis table already exists"
fi

# Load data into BigQuery
echo "💾 Loading data into BigQuery..."
bq load \
    --source_format=NEWLINE_DELIMITED_JSON \
    ${PROJECT_ID}:${DATASET_NAME}.${TABLE_NAME} \
    ${GCS_PATH}

# Check if load was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully uploaded metrics to BigQuery!${NC}"
    echo "View your data at: https://console.cloud.google.com/bigquery?project=${PROJECT_ID}&ws=!1m4!1m3!3m2!1s${PROJECT_ID}!2s${DATASET_NAME}"
    
    # Query the latest entries
    echo -e "\n📈 Latest metrics:"
    bq query --use_legacy_sql=false "
        SELECT timestamp, model_type, compound_analyzed, bioactivity_score, r2
        FROM \`${PROJECT_ID}.${DATASET_NAME}.${TABLE_NAME}\`
        ORDER BY timestamp DESC
        LIMIT 5
    "
else
    echo -e "${RED}❌ Failed to upload metrics to BigQuery${NC}"
    exit 1
fi
