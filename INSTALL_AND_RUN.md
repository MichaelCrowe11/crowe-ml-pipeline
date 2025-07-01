# Installation and Run Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install RDKit (special installation required)
pip3 install rdkit-pypi

# Install remaining dependencies
cd universal-fungal-intelligence-system
pip3 install -r requirements.txt
```

### 2. Test the Implementation

```bash
# Run the test script to verify everything works
python3 test_implementation.py
```

### 3. Run the Full Analysis

#### Option A: Local Analysis (No Cloud)
```bash
python3 src/main.py
```

#### Option B: With BigQuery Export
```bash
# First authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project crowechem-fungi

# Run with export
python3 src/main.py --export-to-bigquery
```

#### Option C: Full GCP Deployment
```bash
python3 scripts/deploy_to_gcp.py
```

### 4. Quick Compound Analysis

Analyze a specific compound:
```bash
# Example: Analyze Penicillin
python3 src/main.py --analyze-compound "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"

# Example: Analyze Aspirin
python3 src/main.py --analyze-compound "CC(=O)Oc1ccccc1C(=O)O"
```

## What Was Implemented

### ✅ Priority Step 1: PubChem Data Collector
- Full REST API client with rate limiting
- Compound search by name, SMILES, formula
- Bioactivity data retrieval
- Fungal metabolite search functionality

### ✅ Priority Step 2: Six Analysis Phases
1. **Data Collection** - Searches multiple fungal compound categories
2. **Chemical Analysis** - Uses RDKit for molecular properties
3. **Bioactivity Prediction** - ML models predict activity
4. **Breakthrough Identification** - Multi-criteria scoring
5. **Synthesis Planning** - Routes and feasibility assessment
6. **Impact Evaluation** - Therapeutic potential assessment

### ✅ Priority Step 3: ML Bioactivity Models
- Random Forest classifier for activity prediction
- Gradient Boosting regressor for potency estimation
- Feature importance analysis
- Therapeutic potential assessment

### ✅ Priority Step 4: Supporting Infrastructure
- BigQuery export functionality
- GCP deployment scripts
- Comprehensive logging
- Error handling throughout

## Troubleshooting

### RDKit Installation Issues
If RDKit installation fails:
```bash
# Try conda instead
conda install -c conda-forge rdkit

# Or use Docker
docker run -it continuumio/miniconda3 /bin/bash
conda install -c conda-forge rdkit
```

### Missing Dependencies
```bash
# Install all at once
pip3 install numpy pandas scikit-learn joblib requests aiohttp sqlalchemy pytest google-cloud-bigquery google-cloud-storage rdkit-pypi
```

### Permission Errors
Make sure scripts are executable:
```bash
chmod +x upload_metrics_to_bigquery.sh
chmod +x universal-fungal-intelligence-system/test_implementation.py
```

## Next Steps

1. **Expand Data Sources**: Implement MycoBank and NCBI clients
2. **Train Real Models**: Replace synthetic training data with real bioactivity data
3. **Add More Compounds**: Increase search result limits for production
4. **Optimize Performance**: Add caching and parallel processing
5. **Build UI**: Create web interface for analysis results 