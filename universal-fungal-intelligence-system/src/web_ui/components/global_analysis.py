"""
Global Analysis Components for the Universal Fungal Intelligence System
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

def render_analysis_launcher() -> bool:
    """Render the global analysis launcher."""
    st.markdown("## 🌍 Full Fungal Kingdom Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🔬 What This Analysis Does:
        - Scans all documented fungal species
        - Analyzes molecular compounds
        - Identifies breakthrough discoveries
        - Predicts therapeutic potential
        - Generates synthesis pathways
        """)
        
        return st.button("🚀 Start Global Analysis", use_container_width=True)
    
    return False

def render_analysis_metrics(results: Dict[str, Any]) -> None:
    """Render analysis summary metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>🧬</h2>
            <h3>{results['total_compounds_analyzed']:,}</h3>
            <p>Compounds Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        breakthroughs = len(results.get('breakthrough_discoveries', []))
        st.markdown(f"""
        <div class="metric-card">
            <h2>💡</h2>
            <h3>{breakthroughs}</h3>
            <p>Breakthroughs Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        impact = results.get('impact_assessment', {})
        innovation_score = impact.get('innovation_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>{innovation_score:.1f}</h3>
            <p>Innovation Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        beneficiaries = impact.get('potential_beneficiaries', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h2>👥</h2>
            <h3>{beneficiaries:,}</h3>
            <p>Potential Beneficiaries</p>
        </div>
        """, unsafe_allow_html=True)

def render_breakthrough_discoveries(results: Dict[str, Any]) -> None:
    """Render breakthrough discovery cards."""
    st.markdown("### 🏆 Top Breakthrough Discoveries")
    
    breakthroughs = results.get('breakthrough_discoveries', [])[:5]
    
    if not breakthroughs:
        st.warning("No breakthrough discoveries found in this analysis.")
        return
    
    for i, compound in enumerate(breakthroughs):
        with st.expander(f"#{i+1} {compound.get('name', 'Unknown')} (CID: {compound.get('cid')})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Breakthrough Score:** {compound.get('breakthrough_score', 0):.2f}
                
                **Reasons:**
                {chr(10).join(f"• {reason}" for reason in compound.get('breakthrough_reasons', []))}
                """)
            
            with col2:
                st.markdown(f"""
                **Therapeutic Targets:**
                {chr(10).join(f"• {target}" for target in compound.get('therapeutic_targets', []))}
                
                **Species Source:** {compound.get('species', 'Unknown')}
                """)

def render_therapeutic_areas_chart(results: Dict[str, Any]) -> None:
    """Render therapeutic areas distribution chart."""
    impact = results.get('impact_assessment', {})
    areas = impact.get('therapeutic_areas', {})
    
    if not areas:
        st.warning("No therapeutic areas data available.")
        return
    
    st.markdown("### 🎯 Therapeutic Areas Distribution")
    
    df_areas = pd.DataFrame(list(areas.items()), columns=['Area', 'Count'])
    
    fig = px.pie(df_areas, values='Count', names='Area', 
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Distribution of Therapeutic Targets")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_compound_timeline(results: Dict[str, Any]) -> None:
    """Render compound discovery timeline."""
    compounds = results.get('breakthrough_discoveries', [])
    
    if not compounds:
        return
    
    st.markdown("### 📈 Discovery Timeline")
    
    # Create timeline data
    timeline_data = []
    for compound in compounds:
        timeline_data.append({
            'Compound': compound.get('name', 'Unknown'),
            'Score': compound.get('breakthrough_score', 0),
            'CID': compound.get('cid', 0),
            'Targets': len(compound.get('therapeutic_targets', []))
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    fig = px.scatter(df_timeline, x='CID', y='Score', 
                    size='Targets', hover_name='Compound',
                    title="Breakthrough Compounds by Score and Targets",
                    labels={'Score': 'Breakthrough Score', 'CID': 'PubChem CID'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_synthesis_pathways_summary(results: Dict[str, Any]) -> None:
    """Render synthesis pathways summary."""
    pathways = results.get('synthesis_pathways', {})
    
    if not pathways:
        st.warning("No synthesis pathways generated.")
        return
    
    st.markdown("### 🧪 Synthesis Pathways Summary")
    
    feasible_count = sum(1 for p in pathways.values() if p.get('feasible', False))
    total_count = len(pathways)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pathways", total_count)
    with col2:
        st.metric("Feasible Pathways", feasible_count)
    with col3:
        feasibility_rate = (feasible_count / total_count * 100) if total_count > 0 else 0
        st.metric("Feasibility Rate", f"{feasibility_rate:.1f}%")
    
    # Show pathway details
    with st.expander("View Pathway Details"):
        for cid, pathway in list(pathways.items())[:5]:  # Show first 5
            st.markdown(f"""
            **Compound CID:** {cid}
            - **Feasible:** {pathway.get('feasible', False)}
            - **Complexity:** {pathway.get('complexity_score', 0):.2f}
            - **Steps:** {pathway.get('num_steps', 0)}
            """)

def render_export_options(results: Dict[str, Any]) -> None:
    """Render export options for analysis results."""
    st.markdown("### 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Download JSON", use_container_width=True):
            import json
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"fungal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download CSV", use_container_width=True):
            # Convert breakthroughs to DataFrame
            breakthroughs = results.get('breakthrough_discoveries', [])
            if breakthroughs:
                df = pd.DataFrame(breakthroughs)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"breakthrough_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("☁️ Export to BigQuery", use_container_width=True):
            try:
                from utils.bigquery_exporter import BigQueryExporter
                exporter = BigQueryExporter()
                job_id = exporter.export_analysis_results(results)
                st.success(f"✅ Exported to BigQuery! Job ID: {job_id}")
            except Exception as e:
                st.error(f"Failed to export: {str(e)}")

def render_analysis_progress(status: str, progress: float) -> None:
    """Render analysis progress indicators."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress)
    
    with col2:
        st.markdown(f"**{status}**")
    
    # Status messages
    if progress < 0.2:
        st.info("🔍 Collecting fungal species data...")
    elif progress < 0.4:
        st.info("🧬 Analyzing molecular compounds...")
    elif progress < 0.6:
        st.info("🤖 Running ML predictions...")
    elif progress < 0.8:
        st.info("💡 Identifying breakthroughs...")
    else:
        st.info("📊 Finalizing analysis...")
