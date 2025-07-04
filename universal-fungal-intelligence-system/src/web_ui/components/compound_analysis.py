"""
Compound Analysis Components for the Universal Fungal Intelligence System
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def render_compound_input_form() -> tuple:
    """Render the compound input form and return user input."""
    st.markdown("## 🧪 Quick Compound Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        compound_input = st.text_area(
            "Enter SMILES string or compound name",
            placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O for Aspirin",
            height=100
        )
        
        analyze_btn = st.button("🔍 Analyze Compound", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Enter SMILES notation for chemical structure
        - Or enter compound name for PubChem search
        - Examples: Aspirin, Penicillin, Taxol
        """)
    
    return compound_input, analyze_btn

def render_compound_metrics(analysis: Dict[str, Any]) -> None:
    """Render compound analysis metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Molecular Weight", f"{analysis['molecular_weight']:.2f}")
    with col2:
        st.metric("LogP", f"{analysis['logP']:.2f}")
    with col3:
        st.metric("Drug-likeness", analysis['drug_likeness'])
    with col4:
        st.metric("Lipinski Violations", analysis['lipinski_violations'])

def render_property_radar_chart(analysis: Dict[str, Any]) -> None:
    """Render molecular property radar chart."""
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
        name='Properties',
        line_color='rgb(46, 204, 113)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title="Molecular Property Profile",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def render_bioactivity_gauge(bioactivity: Dict[str, Any]) -> None:
    """Render bioactivity confidence gauge."""
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
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_bioactivity_details(bioactivity: Dict[str, Any]) -> None:
    """Render detailed bioactivity information."""
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

def render_feature_importance_chart(metrics: Dict[str, Any]) -> None:
    """Render feature importance chart."""
    if 'feature_importance' not in metrics:
        return
    
    importance = metrics['feature_importance']
    df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values('Importance', ascending=True)
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='viridis',
                title="Feature Importance in Bioactivity Prediction")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_compound_structure_display(smiles: str, compound_name: str) -> None:
    """Render compound structure display."""
    st.markdown("### 🧬 Chemical Structure")
    st.info("3D structure visualization requires RDKit integration")
    
    # Display SMILES
    st.markdown(f"**SMILES Notation:**")
    st.code(smiles, language='text')
    
    # Could add 2D structure rendering here with RDKit
    st.markdown(f"**Compound Name:** {compound_name}")

def render_compound_tabs(analysis: Dict[str, Any], bioactivity: Dict[str, Any], 
                        smiles: str, compound_name: str) -> None:
    """Render tabbed interface for compound analysis results."""
    tab1, tab2, tab3 = st.tabs(["📊 Properties", "🎯 Bioactivity", "🧬 Structure"])
    
    with tab1:
        render_property_radar_chart(analysis)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            render_bioactivity_gauge(bioactivity)
        with col2:
            render_bioactivity_details(bioactivity)
    
    with tab3:
        render_compound_structure_display(smiles, compound_name)
