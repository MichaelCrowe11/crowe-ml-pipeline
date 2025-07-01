Advanced 3D Molecular Visualization Component
Powered by Crowe Logic™ AI Engine | Part of the Mycelium EI Ecosystem
"""

import streamlit as st
import streamlit.components.v1 as components
import json
from rdkit impor"""
t Chem
from rdkit.Chem import AllChem, Descriptors
import py3Dmol
import numpy as np

class Molecular3DVisualizer:
    """
    Advanced 3D molecular visualization with interactive controls
    """
    
    # CPK color scheme
    CPK_COLORS = {
        'H': '#FFFFFF',  # White
        'C': '#909090',  # Grey
        'N': '#3050F8',  # Blue
        'O': '#FF0D0D',  # Red
        'F': '#90E050',  # Green
        'Cl': '#1FF01F', # Green
        'Br': '#A62929', # Dark red
        'I': '#940094',  # Purple
        'S': '#FFFF30',  # Yellow
        'P': '#FF8000',  # Orange
        'B': '#FFB5B5',  # Salmon
        'Si': '#F0C8A0', # Beige
        'Fe': '#E06633', # Orange
        'Cu': '#C88033', # Brown
        'Zn': '#7D80B0', # Grey-blue
        'default': '#FF1493'  # Deep pink for unknown
    }
    
    # Van der Waals radii (in Angstroms)
    VDW_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
        'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
        'Br': 1.85, 'I': 1.98, 'default': 1.70
    }
    
    def __init__(self):
        self.viewer_height = 500
        self.style_options = ['stick', 'sphere', 'cartoon', 'surface']
        
    def smiles_to_3d(self, smiles):
        """Convert SMILES to 3D coordinates using RDKit"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            return mol
        except Exception as e:
            st.error(f"Error generating 3D structure: {e}")
            return None
    
    def get_molecular_data(self, mol):
        """Extract molecular data for visualization"""
        if mol is None:
            return None
            
        conf = mol.GetConformer()
        
        # Get atoms
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            pos = conf.GetAtomPosition(i)
            
            atoms.append({
                'id': f"{symbol}-{i+1}",
                'element': symbol,
                'position': [pos.x, pos.y, pos.z],
                'color': self.CPK_COLORS.get(symbol, self.CPK_COLORS['default']),
                'radius': self.VDW_RADII.get(symbol, self.VDW_RADII['default']) * 0.3
            })
        
        # Get bonds
        bonds = []
        for bond in mol.GetBonds():
            bond_type = 'single'
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                bond_type = 'double'
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                bond_type = 'triple'
            elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                bond_type = 'aromatic'
                
            bonds.append({
                'atom1': f"{mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()}-{bond.GetBeginAtomIdx()+1}",
                'atom2': f"{mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()}-{bond.GetEndAtomIdx()+1}",
                'type': bond_type,
                'order': float(bond.GetBondTypeAsDouble())
            })
        
        # Calculate properties
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        return {
            'formula': formula,
            'atoms': atoms,
            'bonds': bonds,
            'properties': {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
            }
        }
    
    def create_3dmol_viewer(self, mol_data, style='stick'):
        """Create interactive 3Dmol.js viewer"""
        if not mol_data:
            return None
            
        # Generate 3Dmol.js visualization script
        viewer_script = f"""
        <div id="mol3d-viewer" style="height: {self.viewer_height}px; width: 100%; position: relative;"></div>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <script>
        $(function() {{
            let element = $('#mol3d-viewer');
            let config = {{ backgroundColor: 'white' }};
            let viewer = $3Dmol.createViewer(element, config);
            
            // Molecule data
            let atoms = {json.dumps(mol_data['atoms'])};
            let bonds = {json.dumps(mol_data['bonds'])};
            
            // Add atoms
            atoms.forEach(atom => {{
                viewer.addSphere({{
                    center: {{x: atom.position[0], y: atom.position[1], z: atom.position[2]}},
                    radius: atom.radius,
                    color: atom.color,
                    opacity: 0.9
                }});
                
                // Add atom labels
                viewer.addLabel(atom.element, {{
                    position: {{x: atom.position[0], y: atom.position[1], z: atom.position[2]}},
                    backgroundColor: 'transparent',
                    fontColor: 'black',
                    fontSize: 10
                }});
            }});
            
            // Add bonds
            bonds.forEach(bond => {{
                let atom1 = atoms.find(a => a.id === bond.atom1);
                let atom2 = atoms.find(a => a.id === bond.atom2);
                
                if (atom1 && atom2) {{
                    let start = {{x: atom1.position[0], y: atom1.position[1], z: atom1.position[2]}};
                    let end = {{x: atom2.position[0], y: atom2.position[1], z: atom2.position[2]}};
                    
                    if (bond.type === 'single') {{
                        viewer.addCylinder({{
                            start: start,
                            end: end,
                            radius: 0.15,
                            color: 'gray'
                        }});
                    }} else if (bond.type === 'double') {{
                        // Double bond - two parallel cylinders
                        let offset = 0.1;
                        viewer.addCylinder({{
                            start: start,
                            end: end,
                            radius: 0.1,
                            color: 'gray'
                        }});
                        // Second cylinder would need perpendicular offset calculation
                    }} else if (bond.type === 'triple') {{
                        // Triple bond visualization
                        viewer.addCylinder({{
                            start: start,
                            end: end,
                            radius: 0.08,
                            color: 'gray'
                        }});
                    }}
                }}
            }});
            
            // Set view
            viewer.zoomTo();
            viewer.render();
            
            // Enable rotation
            viewer.rotate(90, {{x: 0, y: 1, z: 0}}, 1);
            
            // Animation
            viewer.animate({{loop: "forward", reps: 0}});
        }});
        </script>
        """
        
        return viewer_script
    
    def create_plotly_3d(self, mol_data):
        """Create Plotly 3D visualization as fallback"""
        import plotly.graph_objects as go
        
        if not mol_data:
            return None
            
        # Extract positions
        x = [atom['position'][0] for atom in mol_data['atoms']]
        y = [atom['position'][1] for atom in mol_data['atoms']]
        z = [atom['position'][2] for atom in mol_data['atoms']]
        
        # Create atom trace
        atom_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=[atom['radius']*20 for atom in mol_data['atoms']],
                color=[atom['color'] for atom in mol_data['atoms']],
                line=dict(color='darkgray', width=1)
            ),
            text=[atom['element'] for atom in mol_data['atoms']],
            textposition='top center',
            name='Atoms'
        )
        
        # Create bond traces
        bond_traces = []
        for bond in mol_data['bonds']:
            atom1 = next(a for a in mol_data['atoms'] if a['id'] == bond['atom1'])
            atom2 = next(a for a in mol_data['atoms'] if a['id'] == bond['atom2'])
            
            bond_trace = go.Scatter3d(
                x=[atom1['position'][0], atom2['position'][0]],
                y=[atom1['position'][1], atom2['position'][1]],
                z=[atom1['position'][2], atom2['position'][2]],
                mode='lines',
                line=dict(color='gray', width=3*bond['order']),
                showlegend=False
            )
            bond_traces.append(bond_trace)
        
        # Create figure
        fig = go.Figure(data=[atom_trace] + bond_traces)
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=self.viewer_height
        )
        
        return fig
    
    def render(self, smiles, use_3dmol=True):
        """Render 3D molecular visualization in Streamlit"""
        # Convert SMILES to 3D
        mol = self.smiles_to_3d(smiles)
        if mol is None:
            st.error("Could not generate 3D structure from SMILES")
            return
            
        # Get molecular data
        mol_data = self.get_molecular_data(mol)
        
        # Display molecular info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Formula", mol_data['formula'])
        with col2:
            st.metric("Atoms", len(mol_data['atoms']))
        with col3:
            st.metric("Bonds", len(mol_data['bonds']))
        
        # Create visualization
        if use_3dmol:
            # Use 3Dmol.js for better performance
            viewer_html = self.create_3dmol_viewer(mol_data)
            if viewer_html:
                components.html(viewer_html, height=self.viewer_height + 50)
        else:
            # Fallback to Plotly
            fig = self.create_plotly_3d(mol_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Display properties
        with st.expander("Molecular Properties"):
            props = mol_data['properties']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MW", f"{props['molecular_weight']:.2f}")
                st.metric("LogP", f"{props['logp']:.2f}")
            with col2:
                st.metric("HBD", props['hbd'])
                st.metric("HBA", props['hba'])
            with col3:
                st.metric("TPSA", f"{props['tpsa']:.2f}")
                st.metric("Rotatable", props['rotatable_bonds'])
        
        return mol_data


# Integration with main app
def add_3d_visualization_tab():
    """Add 3D visualization tab to the main Streamlit app"""
    st.markdown("### 🧬 3D Molecular Visualization")
    st.markdown("*Powered by Crowe Logic™ AI Engine*")
    
    # Input options
    input_method = st.radio("Input Method:", ["SMILES", "Compound Name", "PubChem CID"])
    
    visualizer = Molecular3DVisualizer()
    
    if input_method == "SMILES":
        smiles = st.text_input("Enter SMILES:", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        if st.button("Visualize 3D Structure"):
            if smiles:
                with st.spinner("Generating 3D structure..."):
                    visualizer.render(smiles)
                    
    elif input_method == "Compound Name":
        compound_name = st.text_input("Enter compound name:", placeholder="aspirin")
        if st.button("Search and Visualize"):
            if compound_name:
                with st.spinner("Searching compound..."):
                    # Integration with PubChem client
                    from data.collectors.pubchem_client import PubChemClient
                    client = PubChemClient()
                    compound = client.get_compound_by_name(compound_name)
                    if compound and 'smiles' in compound:
                        visualizer.render(compound['smiles'])
                    else:
                        st.error("Compound not found")
                    client.close()
                    
    elif input_method == "PubChem CID":
        cid = st.number_input("Enter PubChem CID:", min_value=1, step=1)
        if st.button("Fetch and Visualize"):
            if cid:
                with st.spinner("Fetching compound..."):
                    from data.collectors.pubchem_client import PubChemClient
                    client = PubChemClient()
                    compound = client.get_compound_by_cid(int(cid))
                    if compound and 'smiles' in compound:
                        visualizer.render(compound['smiles'])
                    else:
                        st.error("Compound not found")
                    client.close()
    
    # Example compounds
    st.markdown("#### Example Compounds")
    examples = {
        "Penicillin G": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
        "Lovastatin": "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Taxol": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)O",
        "Cyclosporin": "CCC1NC(=O)C(C(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O"
    }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, (name, smiles) in enumerate(examples.items()):
        col = [col1, col2, col3, col4, col5][i]
        with col:
            if st.button(name, key=f"example_{i}"):
                with st.spinner("Loading example..."):
                    visualizer.render(smiles)
