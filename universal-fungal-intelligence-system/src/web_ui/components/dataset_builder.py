
import streamlit as st
import sys
import os
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from build_dataset import main as build_dataset_main

def render_dataset_builder():
    """Renders the dataset builder UI."""
    st.header("Dataset Builder")
    
    output_dir = st.text_input("Output Directory", "data")
    output_file = st.text_input("Output Filename", "crowechem_dataset.jsonl")
    
    if st.button("Build Dataset"):
        with st.spinner("Building dataset..."):
            try:
                # We need to run this as a subprocess to avoid issues with Streamlit's execution model
                process = subprocess.run(
                    [sys.executable, "build_dataset.py", "--output-dir", output_dir, "--output-file", output_file],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success("Dataset built successfully!")
                st.code(process.stdout)
                if process.stderr:
                    st.error("Errors:")
                    st.code(process.stderr)
            except subprocess.CalledProcessError as e:
                st.error("Failed to build dataset.")
                st.code(e.stdout)
                st.code(e.stderr)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

