# 🍄 Universal Fungal Intelligence System

> **Advanced EI-powered platform for discovering breakthrough therapeutics from the fungal kingdom**

**Powered by Crowe Logic™ AI Engine | Part of the Mycelium EI Ecosystem**

This is the core Universal Fungal Intelligence System - a sophisticated bioinformatics platform that analyzes the entire documented fungal kingdom to discover novel therapeutic compounds using machine learning, multi-source data integration, and cloud-scale processing.

## 🚀 Quick Start

### Launch Web Interface
```bash
# From the universal-fungal-intelligence-system directory
python run_web_ui.py

# Access at http://localhost:8501
```

### Run Analysis Pipeline
```bash
# Run comprehensive fungal analysis
python src/main.py

# Or run specific tests
python test_implementation.py
```

## 🔧 Core Components

### 🔬 Analysis Engine
- **Fungal Intelligence**: 6-phase analysis pipeline
- **Molecular Analyzer**: RDKit-powered structure analysis
- **Bioactivity Predictor**: ML-based activity prediction
- **Breakthrough Identifier**: Multi-criteria discovery scoring

### 🌐 Data Integration
- **PubChem API**: Compound database with bioassay data
- **MycoBank**: Fungal species taxonomy and metabolites
- **NCBI E-utilities**: Literature mining and genomic data

### 🤖 Machine Learning
- **Random Forest**: Activity classification
- **Gradient Boosting**: Potency regression
- **Feature Engineering**: 8 molecular descriptors
- **Cross-Validation**: Robust model evaluation

### 💻 Web Interface
- **Streamlit UI**: Interactive analysis interface
- **Real-time Visualizations**: Radar charts, gauges, pie charts
- **Multiple Analysis Modes**: Compound analysis, kingdom scan, data exploration
- **Training Pipeline**: Interactive ML model training

## 📊 System Architecture

The system follows a modular architecture with clear separation of concerns:

```
src/
├── core/                    # Core analysis engines
│   ├── fungal_intelligence.py      # Main analysis orchestrator
│   ├── molecular_analyzer.py       # Chemical structure analysis
│   ├── bioactivity_predictor.py    # ML-based predictions
│   ├── breakthrough_identifier.py  # Discovery scoring
│   └── synthesis_predictor.py      # Pathway planning
├── data/collectors/         # Data source integrations
│   ├── pubchem_client.py           # PubChem API client
│   ├── mycobank_client.py          # MycoBank integration
│   └── ncbi_client.py              # NCBI E-utilities
├── ml/                      # Machine learning pipeline
│   ├── models/                     # Model definitions
│   └── training/                   # Training pipeline
├── web_ui/                  # Streamlit interface
│   └── app.py                      # Main web application
├── utils/                   # Utilities and helpers
│   ├── bigquery_exporter.py       # Cloud export
│   ├── chemistry_utils.py          # Chemical utilities
│   └── logging_config.py           # Logging setup
└── database/                # Database models and setup
```

## 🎯 Analysis Pipeline

### 6-Phase Fungal Intelligence Process

1. **Data Collection**
   - Multi-source compound gathering
   - Species-metabolite mapping
   - Literature integration

2. **Chemical Analysis**
   - Molecular property calculation
   - Drug-likeness assessment
   - Structure validation

3. **Bioactivity Prediction**
   - ML-powered activity scoring
   - Potency estimation
   - Therapeutic target matching

4. **Breakthrough Identification**
   - Multi-criteria scoring
   - Novelty assessment
   - Impact evaluation

5. **Synthesis Planning**
   - Pathway prediction
   - Feasibility analysis
   - Condition optimization

6. **Impact Evaluation**
   - Therapeutic potential assessment
   - Beneficiary estimation
   - Innovation scoring

## 💻 Web Interface Features

### 🧪 Quick Compound Analysis
- SMILES or name input
- Real-time molecular analysis
- Interactive property visualizations
- Bioactivity confidence scoring

### 🌍 Full Fungal Kingdom Scan
- Comprehensive analysis of all documented fungi
- Breakthrough discovery identification
- Therapeutic area distribution
- Impact metrics and beneficiary estimates

### 📚 Data Source Explorer
- Multi-database search interface
- Real-time data retrieval
- Export capabilities
- Interactive result tables

### 🤖 Training Pipeline Interface
- Interactive model training
- Progress tracking
- Performance visualization
- Model management

## 🛠️ Installation

### Requirements
- Python 3.8+
- 8GB+ RAM (recommended)
- Internet connection for API access

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/setup_database.py

# Configure environment (optional)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Dependencies
```bash
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0
xgboost>=1.5.0

# Chemistry
rdkit-pypi>=2022.3.0
pubchempy>=1.0.4

# Web interface
streamlit>=1.15.0
plotly>=5.10.0

# Cloud integration
google-cloud-bigquery>=3.0.0
google-cloud-storage>=2.5.0

# Async processing
aiohttp>=3.8.0
```

## 🧪 Usage Examples

### Basic Analysis
```python
from src.core.fungal_intelligence import UniversalFungalIntelligence

# Initialize system
system = UniversalFungalIntelligence()

# Run global analysis
results = system.analyze_global_fungal_kingdom()

# View breakthroughs
print(f"Found {len(results['breakthrough_discoveries'])} breakthroughs")
```

### Compound Analysis
```python
from src.core.molecular_analyzer import MolecularAnalyzer

analyzer = MolecularAnalyzer()
analysis = analyzer.analyze_structure("CC(=O)Oc1ccccc1C(=O)O")
print(f"Drug-likeness: {analysis['drug_likeness']}")
```

### ML Training
```python
from src.ml.training.model_trainer import ModelTrainer

trainer = ModelTrainer()
metrics = trainer.run_training_pipeline(num_compounds=1000)
print(f"Accuracy: {metrics['activity_accuracy']:.3f}")
```

## ☁️ Cloud Integration

### BigQuery Export
```python
from src.utils.bigquery_exporter import BigQueryExporter

exporter = BigQueryExporter()
exporter.export_analysis_results(results)
```

### GCP Configuration
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
export GCP_PROJECT_ID="your-project-id"
```

## 🧪 Testing

### Run Tests
```bash
# All tests
python -m pytest tests/

# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# With coverage
python -m pytest --cov=src tests/
```

### Test Implementation
```bash
# Quick system test
python test_implementation.py
```

## 📚 Documentation

- **[Installation Guide](docs/installation.md)**: Detailed setup instructions
- **[API Documentation](docs/api.md)**: Complete API reference
- **[Usage Guide](docs/usage.md)**: Comprehensive examples
- **[Development Guide](docs/development.md)**: Contributing guidelines

## 🎯 Performance Metrics

- **Data Processing**: 1000+ compounds/minute
- **Model Accuracy**: >85% activity classification
- **API Response**: <5 second analysis time
- **Cloud Scalability**: Auto-scaling ready
- **Real-time UI**: Interactive visualizations

## 🔧 Development

### Code Quality
```bash
# Formatting
black src/
flake8 src/

# Type checking
mypy src/
```

### Project Structure
- **Modular Design**: Clear separation of concerns
- **Async Processing**: High-performance data collection
- **Cloud-Ready**: Scalable deployment architecture
- **Test Coverage**: Comprehensive unit and integration tests

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

See [Development Guide](docs/development.md) for detailed guidelines.

---

**Part of the Universal Fungal Intelligence System**  
*Discovering breakthrough therapeutics through AI-powered fungal analysis*
