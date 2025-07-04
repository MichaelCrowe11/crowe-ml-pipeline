"""
Data Explorer Components for the Universal Fungal Intelligence System
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

def render_data_source_selector() -> tuple:
    """Render data source selection and search interface."""
    st.markdown("## 📚 Data Source Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        data_source = st.selectbox(
            "Select Data Source",
            ["PubChem", "MycoBank", "NCBI"],
            help="Choose which biological database to search"
        )
    
    with col2:
        search_query = st.text_input(
            "Search Query", 
            placeholder="Enter search term...",
            help="Enter keywords to search for compounds or species"
        )
    
    # Search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider("Max Results", 10, 1000, 100)
            include_synonyms = st.checkbox("Include Synonyms", value=True)
        
        with col2:
            filter_fungal = st.checkbox("Filter Fungal Only", value=True)
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])
    
    search_btn = st.button("🔍 Search", use_container_width=True)
    
    return data_source, search_query, max_results, include_synonyms, filter_fungal, export_format, search_btn

def render_search_results(results: List[Dict[str, Any]], data_source: str, 
                         export_format: str) -> None:
    """Render search results in a formatted table."""
    if not results:
        st.warning("No results found for your search query.")
        return
    
    st.success(f"Found {len(results)} results from {data_source}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Display results
    st.dataframe(df, use_container_width=True, height=400)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{data_source.lower()}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if export_format == "JSON":
            import json
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{data_source.lower()}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if export_format == "Excel":
            # For Excel export, we'd need openpyxl
            st.info("Excel export requires openpyxl package")

def render_compound_details(compound: Dict[str, Any]) -> None:
    """Render detailed view of a selected compound."""
    st.markdown("### 🧪 Compound Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Name:** {compound.get('name', 'Unknown')}
        **CID:** {compound.get('cid', 'N/A')}
        **Formula:** {compound.get('molecular_formula', 'N/A')}
        **Weight:** {compound.get('molecular_weight', 'N/A')}
        """)
    
    with col2:
        st.markdown(f"""
        **SMILES:** {compound.get('smiles', 'N/A')}
        **InChI:** {compound.get('inchi', 'N/A')}
        **Source:** {compound.get('source_species', 'N/A')}
        """)
    
    # Additional properties
    if compound.get('properties'):
        st.markdown("#### Properties")
        props = compound['properties']
        for key, value in props.items():
            st.markdown(f"- **{key}:** {value}")

def render_species_details(species: Dict[str, Any]) -> None:
    """Render detailed view of a selected species."""
    st.markdown("### 🍄 Species Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Name:** {species.get('name', 'Unknown')}
        **Kingdom:** {species.get('kingdom', 'N/A')}
        **Phylum:** {species.get('phylum', 'N/A')}
        **Class:** {species.get('class', 'N/A')}
        """)
    
    with col2:
        st.markdown(f"""
        **Order:** {species.get('order', 'N/A')}
        **Family:** {species.get('family', 'N/A')}
        **Genus:** {species.get('genus', 'N/A')}
        **Species:** {species.get('species', 'N/A')}
        """)
    
    # Known compounds
    if species.get('known_compounds'):
        st.markdown("#### Known Compounds")
        compounds = species['known_compounds']
        for compound in compounds[:5]:  # Show first 5
            st.markdown(f"- {compound}")

def render_search_statistics(results: List[Dict[str, Any]], data_source: str) -> None:
    """Render search result statistics."""
    if not results:
        return
    
    st.markdown("### 📊 Search Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Results", len(results))
    
    with col2:
        # Count unique species if available
        species_count = len(set(r.get('species', 'Unknown') for r in results))
        st.metric("Unique Species", species_count)
    
    with col3:
        # Count compounds with molecular weight
        mw_count = sum(1 for r in results if r.get('molecular_weight'))
        st.metric("With Mol. Weight", mw_count)
    
    with col4:
        # Count compounds with SMILES
        smiles_count = sum(1 for r in results if r.get('smiles'))
        st.metric("With Structure", smiles_count)

def render_data_source_info(data_source: str) -> None:
    """Render information about the selected data source."""
    st.markdown("### ℹ️ Data Source Information")
    
    if data_source == "PubChem":
        st.info("""
        **PubChem** is a free chemistry database maintained by the National Center for Biotechnology Information (NCBI).
        
        - **Content:** Chemical compounds, biological activities, literature
        - **Size:** 100+ million compounds
        - **Updates:** Daily
        - **API Rate Limit:** 5 requests per second
        """)
    
    elif data_source == "MycoBank":
        st.info("""
        **MycoBank** is an online database of fungal names and associated basic information.
        
        - **Content:** Fungal taxonomy, nomenclature, literature
        - **Size:** 500,000+ fungal names
        - **Updates:** Regularly updated by mycologists
        - **Focus:** Fungal species classification
        """)
    
    elif data_source == "NCBI":
        st.info("""
        **NCBI** provides access to biological databases including PubMed, GenBank, and taxonomic information.
        
        - **Content:** Literature, sequences, taxonomy
        - **Size:** 30+ million citations
        - **Updates:** Daily
        - **API Rate Limit:** 3 requests per second
        """)

def render_search_history() -> None:
    """Render search history for the current session."""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if not st.session_state.search_history:
        return
    
    st.markdown("### 🕒 Recent Searches")
    
    for i, search in enumerate(st.session_state.search_history[-5:]):  # Show last 5
        with st.expander(f"{search['query']} in {search['source']} ({search['timestamp']})"):
            st.markdown(f"""
            - **Query:** {search['query']}
            - **Source:** {search['source']}
            - **Results:** {search['result_count']}
            - **Time:** {search['timestamp']}
            """)

def add_to_search_history(query: str, source: str, result_count: int) -> None:
    """Add a search to the session history."""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    st.session_state.search_history.append({
        'query': query,
        'source': source,
        'result_count': result_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Keep only last 10 searches
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[-10:]
