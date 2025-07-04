import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import sys
import os
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not installed. Please run: pip install plotly")

from core.fungal_intelligence import UniversalFungalIntelligence
from core.molecular_analyzer import MolecularAnalyzer
from core.bioactivity_predictor import BioactivityPredictor
from data.collectors.pubchem_client import PubChemClient
from data.collectors.mycobank_client import MycoBankClient
from data.collectors.ncbi_client import NCBIClient
from database import init_db
from utils.bigquery_exporter import BigQueryExporter

# Import new modular components
from web_ui.components.compound_analysis import *
from web_ui.components.global_analysis import *
from web_ui.components.data_explorer import *
from web_ui.components.integration import integration_manager, render_integration_help

# Page config
st.set_page_config(
    page_title="🍄 Universal Fungal Intelligence System",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(46, 204, 113, 0.4);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .breakthrough-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'compound_data' not in st.session_state:
    st.session_state.compound_data = None
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = None

# Initialize components
@st.cache_resource
def init_components():
    """Initialize system components."""
    init_db()
    molecular_analyzer = MolecularAnalyzer()
    bioactivity_predictor = BioactivityPredictor()
    return molecular_analyzer, bioactivity_predictor

molecular_analyzer, bioactivity_predictor = init_components()

# Header
st.markdown("""
# 🍄 Universal Fungal Intelligence System
### Discovering Breakthrough Therapeutics from the Fungal Kingdom
""")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Quick Compound Analysis", "Full Fungal Kingdom Scan", "Data Source Explorer", 
         "Training Pipeline", "App Integration", "Integration Help"]
    )
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📤 Export Options")
    export_to_bigquery = st.checkbox("Export to BigQuery", value=False)
    
    if export_to_bigquery:
        st.info("Results will be exported to Google BigQuery")

# Main content area
if analysis_mode == "Quick Compound Analysis":
    st.markdown("## 🧪 Quick Compound Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        compound_input = st.text_area(
            "Enter SMILES string or compound name",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O for Aspirin",
            height=100
        )
        
        analyze_btn = st.button("🔍 Analyze Compound", use_container_width=True)
    
    if analyze_btn and compound_input:
        with st.spinner("Analyzing compound..."):
            # Check if it's a SMILES or name
            if any(char in compound_input for char in ['(', ')', '=', '[', ']']):
                # Likely SMILES
                analysis = molecular_analyzer.analyze_structure(compound_input)
                compound_name = "User Compound"
                smiles = compound_input
            else:
                # Search by name
                pubchem_client = PubChemClient()
                compound_data = pubchem_client.get_compound_by_name(compound_input)
                pubchem_client.close()
                
                if compound_data:
                    smiles = compound_data.get('smiles', '')
                    compound_name = compound_data.get('name', compound_input)
                    analysis = molecular_analyzer.analyze_structure(smiles)
                else:
                    st.error("Compound not found in PubChem")
                    analysis = None
            
            if analysis and 'error' not in analysis:
                st.session_state.compound_data = analysis
                st.session_state.selected_compound = compound_name
                
                # Display results
                st.success(f"✅ Analysis complete for {compound_name}")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Molecular Weight", f"{analysis['molecular_weight']:.2f}")
                with col2:
                    st.metric("LogP", f"{analysis['logP']:.2f}")
                with col3:
                    st.metric("Drug-likeness", analysis['drug_likeness'])
                with col4:
                    st.metric("Lipinski Violations", analysis['lipinski_violations'])
                
                # Visualization tabs
                tab1, tab2, tab3 = st.tabs(["📊 Properties", "🎯 Bioactivity", "🧬 Structure"])
                
                with tab1:
                    # Property radar chart
                    properties = {
                        'Molecular Weight': min(analysis['molecular_weight'] / 500, 1),
                        'LogP': (analysis['logP'] + 2) / 7,  # Normalize to 0-1
                        'H Donors': 1 - (analysis['num_h_donors'] / 5),
                        'H Acceptors': 1 - (analysis['num_h_acceptors'] / 10),
                        'TPSA': 1 - (analysis['tpsa'] / 140),
                        'Rotatable Bonds': 1 - (analysis.get('num_rotatable_bonds', 5) / 15)
                    }
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=list(properties.values()),
                        theta=list(properties.keys()),
                        fill='toself',
                        name='Properties'
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])
                        ),
                        showlegend=False,
                        title="Molecular Property Profile"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Predict bioactivity
                    bioactivity = bioactivity_predictor.predict_bioactivity(analysis)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Activity gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=bioactivity['confidence_score'] * 100,
                            title={'text': "Activity Confidence"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkgreen"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                        ### Bioactivity Assessment
                        - **Predicted Activity**: {bioactivity['predicted_activity']}
                        - **Potency Score**: {bioactivity['potency_score']:.2f}
                        - **Therapeutic Potential**: {bioactivity['therapeutic_potential']}
                        """)
                        
                        if bioactivity.get('key_features'):
                            st.markdown("#### Key Contributing Features:")
                            for feature, importance in bioactivity['key_features'].items():
                                st.progress(importance, text=f"{feature}: {importance:.2f}")
                
                with tab3:
                    # Structure visualization would require additional libraries
                    st.info("3D structure visualization requires RDKit integration")
                    st.code(smiles, language='text')

elif analysis_mode == "Full Fungal Kingdom Scan":
    st.markdown("## 🌍 Full Fungal Kingdom Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Global Analysis", use_container_width=True):
            with st.spinner("Analyzing the entire fungal kingdom... This may take a while."):
                # Initialize system
                fungal_system = UniversalFungalIntelligence()
                
                # Run analysis
                results = asyncio.run(fungal_system.analyze_global_fungal_kingdom())
                st.session_state.analysis_results = results
                
                # Export to BigQuery if enabled
                if export_to_bigquery:
                    try:
                        exporter = BigQueryExporter()
                        exporter.export_analysis_results(results)
                        st.success("✅ Results exported to BigQuery")
                    except Exception as e:
                        st.error(f"Failed to export: {str(e)}")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h2>🧬</h2>
                <h3>{}</h3>
                <p>Compounds Analyzed</p>
            </div>
            """.format(results['total_compounds_analyzed']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h2>💡</h2>
                <h3>{}</h3>
                <p>Breakthroughs Found</p>
            </div>
            """.format(len(results['breakthrough_discoveries'])), unsafe_allow_html=True)
        
        with col3:
            impact = results.get('impact_assessment', {})
            st.markdown("""
            <div class="metric-card">
                <h2>🎯</h2>
                <h3>{}</h3>
                <p>Innovation Score</p>
            </div>
            """.format(impact.get('innovation_score', 0)), unsafe_allow_html=True)
        
        with col4:
            beneficiaries = impact.get('potential_beneficiaries', 0)
            st.markdown("""
            <div class="metric-card">
                <h2>👥</h2>
                <h3>{:,}</h3>
                <p>Potential Beneficiaries</p>
            </div>
            """.format(beneficiaries), unsafe_allow_html=True)
        
        # Breakthrough discoveries
        st.markdown("### 🏆 Top Breakthrough Discoveries")
        
        breakthroughs = results.get('breakthrough_discoveries', [])[:5]
        for i, compound in enumerate(breakthroughs):
            st.markdown(f"""
            <div class="breakthrough-card">
                <h4>#{i+1} {compound.get('name', 'Unknown')} (CID: {compound.get('cid')})</h4>
                <p>Breakthrough Score: {compound.get('breakthrough_score', 0)}</p>
                <p>Reasons: {', '.join(compound.get('breakthrough_reasons', []))}</p>
                <p>Targets: {', '.join(compound.get('therapeutic_targets', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Therapeutic areas chart
        if results.get('therapeutic_candidates'):
            st.markdown("### 🎯 Therapeutic Areas Distribution")
            
            areas = results['impact_assessment']['therapeutic_areas']
            df_areas = pd.DataFrame(list(areas.items()), columns=['Area', 'Count'])
            
            fig = px.pie(df_areas, values='Count', names='Area', 
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Data Source Explorer":
    st.markdown("## 📚 Data Source Explorer")
    
    data_source = st.selectbox(
        "Select Data Source",
        ["PubChem", "MycoBank", "NCBI"]
    )
    
    search_query = st.text_input("Search Query", placeholder="Enter search term...")
    
    if st.button("🔍 Search"):
        with st.spinner(f"Searching {data_source}..."):
            if data_source == "PubChem":
                client = PubChemClient()
                results = client.search_fungal_compounds(search_query, max_results=20)
                client.close()
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"pubchem_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results found")
                    
            elif data_source == "MycoBank":
                async def search_mycobank():
                    async with MycoBankClient() as client:
                        return await client._search_species(search_query)
                
                results = asyncio.run(search_mycobank())
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No results found")
                    
            elif data_source == "NCBI":
                client = NCBIClient()
                results = client.search_fungi(search_query, max_results=20)
                client.close()
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No results found")

elif analysis_mode == "Training Pipeline":
    st.markdown("## 🤖 Model Training Pipeline")
    
    st.info("""
    Train new bioactivity prediction models using real data from all sources.
    This process will:
    1. Collect compounds from PubChem, MycoBank, and NCBI
    2. Calculate molecular features
    3. Train ML models
    4. Save models for future use
    """)
    
    num_compounds = st.slider("Number of compounds to collect", 100, 5000, 1000, 100)
    
    if st.button("🚀 Start Training Pipeline"):
        with st.spinner("Running training pipeline... This may take several minutes."):
            from ml.training.model_trainer import ModelTrainer
            
            trainer = ModelTrainer()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            async def train_with_progress():
                status_text.text("Collecting data from PubChem...")
                progress_bar.progress(0.2)
                
                metrics = await trainer.run_training_pipeline(num_compounds)
                
                progress_bar.progress(1.0)
                status_text.text("Training complete!")
                
                return metrics
            
            metrics = asyncio.run(train_with_progress())
            
            # Display results
            st.success("✅ Training complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Activity Accuracy", f"{metrics['activity_accuracy']:.3f}")
            with col2:
                st.metric("Potency R²", f"{metrics['potency_r2']:.3f}")
            with col3:
                st.metric("Cross-Val AUC", f"{metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
            
            # Feature importance
            st.markdown("### Feature Importance")
            
            importance = metrics['feature_importance']
            df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            df_imp = df_imp.sort_values('Importance', ascending=True)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_imp.set_index('Feature')['Importance'])

elif analysis_mode == "App Integration":
    st.markdown("## 🔧 App Integration")
    
    # Show available integration modules
    selected_module = integration_manager.render_module_selector()
    
    if selected_module:
        st.markdown("---")
        integration_manager.render_module_interface(selected_module)
    else:
        st.info("Select an integration module to get started.")
    
    # Option to add custom integration
    with st.expander("➕ Add Custom Integration"):
        st.markdown("### Custom Integration Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### GitHub Repository Integration")
            github_url = st.text_input("GitHub Repository URL", 
                                     placeholder="https://github.com/username/repository")
            if st.button("🔗 Integrate GitHub App"):
                if github_url:
                    st.info(f"Integration with {github_url} would be set up here.")
                    st.markdown("""
                    **Next Steps:**
                    1. Clone the repository
                    2. Analyze the code structure
                    3. Create integration module
                    4. Register with the system
                    """)
        
        with col2:
            st.markdown("#### Local App Integration")
            app_path = st.text_input("Local App Path", 
                                    placeholder="/path/to/your/app")
            if st.button("📁 Integrate Local App"):
                if app_path:
                    st.info(f"Integration with {app_path} would be set up here.")

elif analysis_mode == "Integration Help":
    render_integration_help()
    
    # Show your current app structure for integration guidance
    st.markdown("## 🏗️ Your Current App Structure")
    
    if st.button("📋 Show Integration Template"):
        st.markdown("### Integration Template for Your App")
        st.code("""
class YourAppIntegration(IntegrationModule):
    def get_name(self) -> str:
        return "Your App Name"
    
    def get_description(self) -> str:
        return "Description of what your app does"
    
    def get_icon(self) -> str:
        return "🎯"  # Choose an emoji
    
    def get_dependencies(self) -> List[str]:
        return ["package1", "package2"]  # Your app's dependencies
    
    def render_interface(self) -> None:
        st.markdown("## Your App Interface")
        
        # Your app's Streamlit code here
        # Example:
        user_input = st.text_input("Your Input")
        if st.button("Process"):
            # Your app's processing logic
            result = your_processing_function(user_input)
            st.success(f"Result: {result}")
    
    def your_processing_function(self, input_data):
        # Your app's core functionality
        return f"Processed: {input_data}"

# Register your integration
integration_manager.register_module(YourAppIntegration())
        """, language='python')
        
    st.markdown("### 🤝 Collaboration Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### What We Can Integrate:
        - **Data Analysis Apps**: Combine your analysis with fungal intelligence
        - **Visualization Tools**: Use your charts with our data
        - **ML Models**: Ensemble your models with ours
        - **API Services**: Connect external data sources
        - **Processing Pipelines**: Integrate your workflows
        """)
    
    with col2:
        st.markdown("""
        #### Integration Benefits:
        - **Modular Design**: Easy to add/remove features
        - **Shared Data**: Access to fungal intelligence data
        - **Cross-Validation**: Validate results across systems
        - **Enhanced UI**: Professional Streamlit interface
        - **Cloud Ready**: Deploy integrated system to cloud
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🍄 Universal Fungal Intelligence System | Powered by AI & Fungal Wisdom | 
    <a href='https://github.com/yourusername/fungal-intelligence'>GitHub</a>
</div>
""", unsafe_allow_html=True)
