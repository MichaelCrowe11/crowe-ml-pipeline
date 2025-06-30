# Crowe ML Pipeline

This repo automates the process of training cultivation models, exporting yield metrics, and uploading them into BigQuery for analysis.

## Contents

- `upload_metrics_to_bigquery.sh`: Shell script to upload training metrics to GCS and BigQuery.
- `cloudbuild.yaml`: GCP Cloud Build pipeline for CI/CD integration.
- `.github/workflows/validate.yaml`: GitHub Actions job to check formatting and syntax.
- `README.md`: This file.
- `build_dataset.py`: Utility to assemble the CroweChem dataset.

## Usage

### Manual Upload via Shell

```bash
chmod +x upload_metrics_to_bigquery.sh
./upload_metrics_to_bigquery.sh
```

Edit the script to set your `PROJECT_ID` and `BUCKET_NAME`.

### Automated via Cloud Build

Push to GitHub and connect Cloud Build with a trigger using `cloudbuild.yaml`.

### Building CroweChem Dataset

Run the dataset builder to assemble training data:

```bash
python build_dataset.py
```

The script creates `data/crowechem_dataset.jsonl` that can be consumed by your
training jobs.

