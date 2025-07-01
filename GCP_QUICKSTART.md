# Crowe ML Pipeline - GCP Quick Start Guide

## Your GCP Project Details
- **Project ID**: `crowechem-fungi`
- **Bucket**: `crowechem-fungi-ml-metrics`
- **Dataset**: `crowe_ml_pipeline`

## Prerequisites
1. Google Cloud SDK installed (`gcloud`, `bq`, `gsutil`)
2. Python 3.7+ with pip
3. Authenticated GCP account with project access

## Quick Setup

### 1. Authenticate with GCP
```bash
# Login to your Google account
gcloud auth login

# Set your project
gcloud config set project crowechem-fungi

# Enable required APIs
gcloud services enable bigquery.googleapis.com storage.googleapis.com
```

### 2. Install Dependencies
```bash
# Install RDKit (special installation)
pip3 install rdkit-pypi

# Install other requirements
cd universal-fungal-intelligence-system
pip3 install -r requirements.txt
```

### 3. Test the Upload Script
```bash
# Run the metrics upload script (creates sample data if needed)
./upload_metrics_to_bigquery.sh
```

### 4. Run Fungal Analysis
```bash
# Option 1: Run locally without cloud export
cd universal-fungal-intelligence-system
python3 src/main.py

# Option 2: Run with BigQuery export
python3 src/main.py --export-to-bigquery

# Option 3: Use the full deployment script
python3 scripts/deploy_to_gcp.py
```

### 5. Analyze a Specific Compound
```bash
# Example: Analyze aspirin
python3 src/main.py --analyze-compound "CC(=O)Oc1ccccc1C(=O)O"
```

## View Results in BigQuery

1. Go to: https://console.cloud.google.com/bigquery?project=crowechem-fungi
2. Navigate to: `crowechem-fungi` → `crowe_ml_pipeline`
3. View tables:
   - `metrics_log` - ML training metrics
   - `fungal_analysis_results` - Analysis summaries
   - `discovered_compounds` - Compound details

## Troubleshooting

### Authentication Issues
```bash
# Set application default credentials
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-key.json"
```

### Permission Errors
Ensure your account has these roles:
- BigQuery Data Editor
- BigQuery Job User
- Storage Object Admin

### Missing APIs
```bash
# Enable all required APIs at once
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com
```

## Next Steps

1. **Implement Data Collectors**: Start with PubChem API in `src/data/collectors/pubchem_client.py`
2. **Complete Analysis Pipeline**: Fill in the TODOs in `analyze_global_fungal_kingdom()`
3. **Add ML Models**: Implement real bioactivity prediction in `src/ml/models/`
4. **Set Up CI/CD**: Configure Cloud Build triggers in GCP Console

## Useful Commands

```bash
# Check BigQuery datasets
bq ls --project_id=crowechem-fungi

# Check Cloud Storage buckets
gsutil ls -p crowechem-fungi

# View latest metrics
bq query --use_legacy_sql=false \
  "SELECT * FROM crowechem-fungi.crowe_ml_pipeline.metrics_log ORDER BY timestamp DESC LIMIT 5"

# Monitor Cloud Build
gcloud builds list --limit=5

# Stream logs
gcloud logging read "resource.type=cloud_function" --limit 50
```

## Support

For issues or questions:
1. Check logs in `universal-fungal-intelligence-system/logs/`
2. View Cloud Build history in GCP Console
3. Check BigQuery job history for failed loads 