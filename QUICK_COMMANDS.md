# Quick Command Reference

## 🚀 Installation
```bash
# Install all dependencies
pip3 install rdkit-pypi
cd universal-fungal-intelligence-system && pip3 install -r requirements.txt
```

## 🧪 Test Everything
```bash
cd universal-fungal-intelligence-system
python3 test_implementation.py
```

## 💊 Analyze Compounds
```bash
# Analyze Aspirin
python3 src/main.py --analyze-compound "CC(=O)Oc1ccccc1C(=O)O"

# Analyze Penicillin G
python3 src/main.py --analyze-compound "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"

# Analyze Caffeine
python3 src/main.py --analyze-compound "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

## 🧬 Run Full Analysis
```bash
# Local only
python3 src/main.py

# With BigQuery export
python3 src/main.py --export-to-bigquery
```

## ☁️ GCP Commands
```bash
# Authenticate
gcloud auth login
gcloud config set project crowechem-fungi

# Upload metrics
cd .. && ./upload_metrics_to_bigquery.sh

# Full deployment
cd universal-fungal-intelligence-system
python3 scripts/deploy_to_gcp.py
```

## 📊 View Results
- BigQuery: https://console.cloud.google.com/bigquery?project=crowechem-fungi
- Tables: `crowechem-fungi.crowe_ml_pipeline.*`

## 🔍 Quick Debugging
```bash
# Check logs
tail -f logs/*.log

# Test PubChem connection
python3 -c "from src.data.collectors.pubchem_client import PubChemClient; c=PubChemClient(); print(c.get_compound_by_name('Aspirin'))"

# Test molecular analyzer
python3 -c "from src.core.molecular_analyzer import MolecularAnalyzer; m=MolecularAnalyzer(); print(m.analyze_structure('CCO'))"
``` 