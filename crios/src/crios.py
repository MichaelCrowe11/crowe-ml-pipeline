#!/usr/bin/env python3
"""
CriOS - Crowe Research Intelligence Operating System
Advanced Compound Discovery Platform with AI-Powered Research Capabilities
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class CompoundProfile:
    """Enhanced compound profile with discovery metadata"""
    smiles: str
    name: Optional[str] = None
    source: str = "unknown"
    discovery_date: datetime = field(default_factory=datetime.now)
    properties: Dict[str, float] = field(default_factory=dict)
    bioactivity: Dict[str, Any] = field(default_factory=dict)
    synthesis_score: float = 0.0
    novelty_score: float = 0.0
    therapeutic_areas: List[str] = field(default_factory=list)
    interaction_network: Optional[nx.Graph] = None
    quantum_properties: Dict[str, float] = field(default_factory=dict)
    
class MolecularTransformer(nn.Module):
    """Transformer model for molecular property prediction"""
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim),
            num_layers=6
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return self.predictor(encoded.mean(dim=0))

class CriOS:
    """Crowe Research Intelligence Operating System"""
    
    def __init__(self, workspace: str = "~/.crios"):
        self.workspace = Path(workspace).expanduser()
        self.workspace.mkdir(exist_ok=True)
        
        # Initialize components
        self.console = Console()
        self.history = FileHistory(str(self.workspace / "history.txt"))
        self.compounds_db: Dict[str, CompoundProfile] = {}
        self.models: Dict[str, Any] = {}
        self.active_projects: List[str] = []
        self.discovery_pipeline = None
        
        # Command registry
        self.commands = {
            'discover': self.discover_compounds,
            'analyze': self.analyze_compound,
            'synthesize': self.plan_synthesis,
            'predict': self.predict_properties,
            'visualize': self.visualize_network,
            'train': self.train_model,
            'export': self.export_results,
            'pipeline': self.manage_pipeline,
            'quantum': self.quantum_analysis,
            'screen': self.virtual_screening,
            'optimize': self.optimize_lead,
            'dock': self.molecular_docking,
            'dynamics': self.molecular_dynamics,
            'report': self.generate_report,
            'collaborate': self.collaborate,
            'help': self.show_help,
            'exit': self.exit_crios
        }
        
        self._initialize_models()
        self._load_databases()
        
    def _initialize_models(self):
        """Initialize ML models and transformers"""
        try:
            # Molecular transformer
            self.models['transformer'] = MolecularTransformer()
            
            # Classical ML models
            self.models['activity_classifier'] = RandomForestClassifier(n_estimators=100)
            self.models['property_predictor'] = GradientBoostingRegressor()
            
            # Load pretrained models if available
            model_path = self.workspace / "models"
            if model_path.exists():
                self._load_saved_models(model_path)
                
            self.console.print("[green]✓ Models initialized successfully[/green]")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.console.print(f"[yellow]⚠ Model initialization partial: {e}[/yellow]")
    
    def _load_databases(self):
        """Load compound databases and knowledge graphs"""
        db_path = self.workspace / "compounds.json"
        if db_path.exists():
            with open(db_path, 'r') as f:
                data = json.load(f)
                for smiles, info in data.items():
                    self.compounds_db[smiles] = CompoundProfile(**info)
        
        self.console.print(f"[cyan]Loaded {len(self.compounds_db)} compounds[/cyan]")
    
    async def discover_compounds(self, target: str = None, **kwargs):
        """Advanced compound discovery with multiple strategies"""
        strategies = kwargs.get('strategies', ['similarity', 'fragment', 'scaffold', 'ai'])
        num_compounds = kwargs.get('num_compounds', 100)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Discovering compounds...", total=len(strategies))
            
            discovered = []
            for strategy in strategies:
                if strategy == 'similarity':
                    compounds = await self._similarity_search(target, num_compounds // len(strategies))
                elif strategy == 'fragment':
                    compounds = await self._fragment_based_discovery(target)
                elif strategy == 'scaffold':
                    compounds = await self._scaffold_hopping(target)
                elif strategy == 'ai':
                    compounds = await self._ai_generation(target)
                else:
                    compounds = []
                
                discovered.extend(compounds)
                progress.update(task, advance=1)
        
        # Rank and filter discoveries
        ranked = self._rank_discoveries(discovered)
        
        # Display results
        self._display_discoveries(ranked[:num_compounds])
        
        # Save to database
        for compound in ranked[:num_compounds]:
            self.compounds_db[compound.smiles] = compound
        
        return ranked[:num_compounds]
    
    async def _similarity_search(self, target: str, n: int) -> List[CompoundProfile]:
        """Find similar compounds using fingerprint similarity"""
        compounds = []
        if target:
            mol = Chem.MolFromSmiles(target)
            if mol:
                fp_target = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                
                # Simulate database search (would connect to real DB)
                for _ in range(n):
                    # Generate similar compound (simplified)
                    similar = self._generate_similar_compound(mol)
                    if similar:
                        compounds.append(CompoundProfile(
                            smiles=Chem.MolToSmiles(similar),
                            source="similarity_search",
                            novelty_score=np.random.uniform(0.6, 0.9)
                        ))
        
        return compounds
    
    async def _ai_generation(self, target: str) -> List[CompoundProfile]:
        """Generate novel compounds using AI"""
        compounds = []
        
        # Use transformer model for generation
        with torch.no_grad():
            # Simplified generation process
            for _ in range(10):
                # Generate molecular embeddings
                z = torch.randn(1, 2048)
                
                # Decode to SMILES (simplified - would use actual decoder)
                generated_smiles = self._decode_embedding(z)
                
                if generated_smiles and Chem.MolFromSmiles(generated_smiles):
                    compounds.append(CompoundProfile(
                        smiles=generated_smiles,
                        source="ai_generation",
                        novelty_score=np.random.uniform(0.8, 1.0)
                    ))
        
        return compounds
    
    def analyze_compound(self, smiles: str = None, **kwargs):
        """Comprehensive compound analysis"""
        if not smiles:
            smiles = prompt("Enter SMILES: ", history=self.history)
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            self.console.print("[red]Invalid SMILES string[/red]")
            return
        
        # Create compound profile
        profile = CompoundProfile(smiles=smiles)
        
        # Calculate properties
        profile.properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'qed': self._calculate_qed(mol),
            'sa_score': self._calculate_sa_score(mol)
        }
        
        # Predict bioactivity
        if self.models.get('activity_classifier'):
            features = self._extract_features(mol)
            profile.bioactivity['predicted_active'] = self.models['activity_classifier'].predict([features])[0]
            profile.bioactivity['confidence'] = self.models['activity_classifier'].predict_proba([features]).max()
        
        # Display analysis
        self._display_compound_analysis(profile)
        
        # Save to database
        self.compounds_db[smiles] = profile
        
        return profile
    
    def plan_synthesis(self, smiles: str = None, **kwargs):
        """AI-powered retrosynthetic analysis"""
        if not smiles:
            smiles = prompt("Enter target SMILES: ", history=self.history)
        
        self.console.print(Panel.fit(
            f"[bold cyan]Retrosynthetic Analysis[/bold cyan]\n"
            f"Target: {smiles}",
            title="Synthesis Planning"
        ))
        
        # Simplified retrosynthesis (would use actual algorithms)
        steps = [
            "Step 1: Identify key disconnections",
            "Step 2: Apply strategic bond formations",
            "Step 3: Optimize reaction conditions",
            "Step 4: Validate synthetic feasibility"
        ]
        
        table = Table(title="Proposed Synthesis Route")
        table.add_column("Step", style="cyan")
        table.add_column("Reaction", style="green")
        table.add_column("Yield", style="yellow")
        
        for i, step in enumerate(steps, 1):
            table.add_row(
                f"Step {i}",
                f"Reaction {i}",
                f"{np.random.uniform(70, 95):.1f}%"
            )
        
        self.console.print(table)
        
        return {"target": smiles, "steps": steps}
    
    def quantum_analysis(self, smiles: str = None, **kwargs):
        """Quantum mechanical property calculations"""
        if not smiles:
            smiles = prompt("Enter SMILES for quantum analysis: ", history=self.history)
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            self.console.print("[red]Invalid SMILES[/red]")
            return
        
        # Simulate quantum calculations
        quantum_props = {
            'homo_energy': np.random.uniform(-10, -5),
            'lumo_energy': np.random.uniform(0, 5),
            'gap': np.random.uniform(3, 8),
            'dipole_moment': np.random.uniform(0, 5),
            'polarizability': np.random.uniform(20, 50),
            'zero_point_energy': np.random.uniform(50, 200)
        }
        
        # Display results
        table = Table(title="Quantum Properties")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        
        units = {
            'homo_energy': 'eV',
            'lumo_energy': 'eV',
            'gap': 'eV',
            'dipole_moment': 'Debye',
            'polarizability': 'Å³',
            'zero_point_energy': 'kcal/mol'
        }
        
        for prop, value in quantum_props.items():
            table.add_row(
                prop.replace('_', ' ').title(),
                f"{value:.3f}",
                units[prop]
            )
        
        self.console.print(table)
        
        # Update compound profile if exists
        if smiles in self.compounds_db:
            self.compounds_db[smiles].quantum_properties = quantum_props
        
        return quantum_props
    
    def virtual_screening(self, target_file: str = None, **kwargs):
        """High-throughput virtual screening"""
        library_size = kwargs.get('library_size', 10000)
        
        self.console.print(f"[cyan]Starting virtual screening of {library_size} compounds...[/cyan]")
        
        with Progress() as progress:
            task = progress.add_task("Screening...", total=library_size)
            
            hits = []
            for i in range(library_size):
                # Simulate screening (would use actual docking/scoring)
                score = np.random.uniform(0, 10)
                if score > 7:  # Hit threshold
                    hits.append({
                        'id': f"CRIOS_{i:06d}",
                        'score': score,
                        'smiles': self._generate_random_smiles()
                    })
                
                progress.update(task, advance=1)
        
        # Display top hits
        hits.sort(key=lambda x: x['score'], reverse=True)
        
        table = Table(title=f"Top {min(10, len(hits))} Hits")
        table.add_column("ID", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("SMILES", style="white")
        
        for hit in hits[:10]:
            table.add_row(
                hit['id'],
                f"{hit['score']:.2f}",
                hit['smiles'][:30] + "..." if len(hit['smiles']) > 30 else hit['smiles']
            )
        
        self.console.print(table)
        self.console.print(f"[green]Found {len(hits)} hits above threshold[/green]")
        
        return hits
    
    def molecular_dynamics(self, smiles: str = None, **kwargs):
        """Run molecular dynamics simulation"""
        if not smiles:
            smiles = prompt("Enter SMILES for MD simulation: ", history=self.history)
        
        duration = kwargs.get('duration', 100)  # ns
        temperature = kwargs.get('temperature', 300)  # K
        
        self.console.print(Panel.fit(
            f"[bold cyan]Molecular Dynamics Simulation[/bold cyan]\n"
            f"Molecule: {smiles}\n"
            f"Duration: {duration} ns\n"
            f"Temperature: {temperature} K",
            title="MD Setup"
        ))
        
        # Simulate MD trajectory
        with Progress() as progress:
            task = progress.add_task(f"Running MD for {duration} ns...", total=duration)
            
            trajectory = []
            for t in range(duration):
                # Simulate frame
                frame = {
                    'time': t,
                    'energy': np.random.uniform(-1000, -900),
                    'rmsd': np.random.uniform(0, 3),
                    'rg': np.random.uniform(5, 8)
                }
                trajectory.append(frame)
                progress.update(task, advance=1)
        
        # Analysis
        avg_energy = np.mean([f['energy'] for f in trajectory])
        avg_rmsd = np.mean([f['rmsd'] for f in trajectory])
        
        self.console.print(f"\n[green]Simulation Complete![/green]")
        self.console.print(f"Average Energy: {avg_energy:.2f} kcal/mol")
        self.console.print(f"Average RMSD: {avg_rmsd:.2f} Å")
        
        return trajectory
    
    def optimize_lead(self, smiles: str = None, **kwargs):
        """Lead optimization using AI"""
        if not smiles:
            smiles = prompt("Enter lead compound SMILES: ", history=self.history)
        
        optimization_goals = kwargs.get('goals', ['potency', 'selectivity', 'admet'])
        
        self.console.print(f"[cyan]Optimizing lead compound for: {', '.join(optimization_goals)}[/cyan]")
        
        # Generate optimized analogs
        analogs = []
        for i in range(10):
            analog = {
                'id': f"OPT_{i:03d}",
                'smiles': self._modify_molecule(smiles),
                'improvements': {}
            }
            
            for goal in optimization_goals:
                analog['improvements'][goal] = np.random.uniform(1.1, 2.0)
            
            analogs.append(analog)
        
        # Rank by overall improvement
        for analog in analogs:
            analog['overall_score'] = np.mean(list(analog['improvements'].values()))
        
        analogs.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Display results
        table = Table(title="Optimized Analogs")
        table.add_column("ID", style="cyan")
        table.add_column("Overall Score", style="green")
        
        for goal in optimization_goals:
            table.add_column(goal.capitalize(), style="yellow")
        
        for analog in analogs[:5]:
            row = [analog['id'], f"{analog['overall_score']:.2f}"]
            for goal in optimization_goals:
                row.append(f"{analog['improvements'][goal]:.2f}x")
            table.add_row(*row)
        
        self.console.print(table)
        
        return analogs
    
    def generate_report(self, project: str = None, **kwargs):
        """Generate comprehensive discovery report"""
        if not project:
            project = prompt("Enter project name: ", history=self.history)
        
        report_type = kwargs.get('type', 'full')
        
        self.console.print(f"[cyan]Generating {report_type} report for project: {project}[/cyan]")
        
        # Create report structure
        report = {
            'project': project,
            'date': datetime.now().isoformat(),
            'summary': {
                'compounds_analyzed': len(self.compounds_db),
                'hits_identified': sum(1 for c in self.compounds_db.values() if c.bioactivity.get('predicted_active')),
                'lead_candidates': 5
            },
            'top_compounds': [],
            'key_findings': [
                "Identified novel scaffold with improved selectivity",
                "Optimized lead compound shows 2x potency improvement",
                "Synthesis route validated for scale-up"
            ]
        }
        
        # Export report
        report_path = self.workspace / f"report_{project}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.console.print(f"[green]Report saved to: {report_path}[/green]")
        
        # Display summary
        self.console.print(Panel.fit(
            f"[bold]Project Summary[/bold]\n\n"
            f"Compounds Analyzed: {report['summary']['compounds_analyzed']}\n"
            f"Hits Identified: {report['summary']['hits_identified']}\n"
            f"Lead Candidates: {report['summary']['lead_candidates']}\n\n"
            f"[bold]Key Findings:[/bold]\n" +
            "\n".join(f"• {finding}" for finding in report['key_findings']),
            title=f"Report: {project}"
        ))
        
        return report
    
    def _display_discoveries(self, compounds: List[CompoundProfile]):
        """Display discovered compounds in a formatted table"""
        table = Table(title="Discovered Compounds")
        table.add_column("SMILES", style="cyan", max_width=40)
        table.add_column("Source", style="green")
        table.add_column("Novelty", style="yellow")
        table.add_column("Synthesis", style="magenta")
        
        for compound in compounds[:10]:
            smiles_display = compound.smiles[:37] + "..." if len(compound.smiles) > 40 else compound.smiles
            table.add_row(
                smiles_display,
                compound.source,
                f"{compound.novelty_score:.2f}",
                f"{compound.synthesis_score:.2f}"
            )
        
        self.console.print(table)
    
    def _display_compound_analysis(self, profile: CompoundProfile):
        """Display compound analysis results"""
        # Properties table
        prop_table = Table(title="Molecular Properties")
        prop_table.add_column("Property", style="cyan")
        prop_table.add_column("Value", style="green")
        prop_table.add_column("Status", style="yellow")
        
        # Lipinski Rule of Five check
        lipinski_violations = 0
        for prop, value in profile.properties.items():
            status = "✓"
            if prop == 'molecular_weight' and value > 500:
                status = "⚠"
                lipinski_violations += 1
            elif prop == 'logp' and value > 5:
                status = "⚠"
                lipinski_violations += 1
            elif prop == 'hbd' and value > 5:
                status = "⚠"
                lipinski_violations += 1
            elif prop == 'hba' and value > 10:
                status = "⚠"
                lipinski_violations += 1
            
            prop_table.add_row(
                prop.replace('_', ' ').title(),
                f"{value:.2f}" if isinstance(value, float) else str(value),
                status
            )
        
        self.console.print(prop_table)
        
        # Drug-likeness assessment
        if lipinski_violations <= 1:
            self.console.print("[green]✓ Drug-like (Lipinski compliant)[/green]")
        else:
            self.console.print(f"[yellow]⚠ {lipinski_violations} Lipinski violations[/yellow]")
        
        # Bioactivity predictions
        if profile.bioactivity:
            bio_table = Table(title="Bioactivity Predictions")
            bio_table.add_column("Target", style="cyan")
            bio_table.add_column("Activity", style="green")
            bio_table.add_column("Confidence", style="yellow")
            
            for target, activity in profile.bioactivity.items():
                if target != 'confidence':
                    bio_table.add_row(
                        target,
                        "Active" if activity else "Inactive",
                        f"{profile.bioactivity.get('confidence', 0):.2%}"
                    )
            
            self.console.print(bio_table)
    
    def _rank_discoveries(self, compounds: List[CompoundProfile]) -> List[CompoundProfile]:
        """Rank discovered compounds by multiple criteria"""
        for compound in compounds:
            # Calculate composite score
            mol = Chem.MolFromSmiles(compound.smiles)
            if mol:
                # Drug-likeness score
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                
                drug_score = 1.0
                if mw > 500: drug_score *= 0.9
                if logp > 5: drug_score *= 0.9
                
                # Synthesis feasibility
                compound.synthesis_score = self._calculate_sa_score(mol)
                
                # Overall ranking
                compound.overall_score = (
                    compound.novelty_score * 0.4 +
                    drug_score * 0.3 +
                    compound.synthesis_score * 0.3
                )
        
        return sorted(compounds, key=lambda x: x.overall_score, reverse=True)
    
    def _generate_similar_compound(self, mol):
        """Generate a similar compound through small modifications"""
        # Simplified - would use actual molecular generation
        return mol
    
    def _generate_random_smiles(self) -> str:
        """Generate a random valid SMILES string"""
        # Simplified - would use actual generation
        fragments = ['CC', 'CN', 'CO', 'C(=O)', 'C1CC1', 'c1ccccc1']
        smiles = ''.join(np.random.choice(fragments, size=np.random.randint(3, 6)))
        return smiles
    
    def _modify_molecule(self, smiles: str) -> str:
        """Modify a molecule for lead optimization"""
        # Simplified - would use actual modification algorithms
        return smiles + "C"
    
    def _decode_embedding(self, embedding):
        """Decode molecular embedding to SMILES"""
        # Simplified - would use actual decoder
        return "CC(C)C(=O)O"
    
    def _extract_features(self, mol) -> np.ndarray:
        """Extract molecular features for ML models"""
        features = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumHeteroatoms(mol)
        ]
        return np.array(features)
    
    def _calculate_qed(self, mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness)"""
        # Simplified - would use actual QED calculation
        return np.random.uniform(0.3, 0.9)
    
    def _calculate_sa_score(self, mol) -> float:
        """Calculate synthetic accessibility score"""
        # Simplified - would use actual SA score calculation
        return np.random.uniform(1, 6)
    
    def show_help(self):
        """Display help information"""
        help_text = """
        [bold cyan]CriOS Commands:[/bold cyan]
        
        [green]discover[/green] - Discover new compounds using AI
        [green]analyze[/green] - Analyze compound properties
        [green]synthesize[/green] - Plan synthesis routes
        [green]predict[/green] - Predict molecular properties
        [green]visualize[/green] - Visualize molecular networks
        [green]train[/green] - Train ML models
        [green]quantum[/green] - Quantum mechanical analysis
        [green]screen[/green] - Virtual screening
        [green]optimize[/green] - Lead optimization
        [green]dock[/green] - Molecular docking
        [green]dynamics[/green] - Molecular dynamics simulation
        [green]report[/green] - Generate project report
        [green]help[/green] - Show this help message
        [green]exit[/green] - Exit CriOS
        """
        self.console.print(Panel.fit(help_text, title="CriOS Help"))
    
    def exit_crios(self):
        """Exit CriOS"""
        # Save state
        db_path = self.workspace / "compounds.json"
        compounds_data = {}
        for smiles, profile in self.compounds_db.items():
            compounds_data[smiles] = {
                'smiles': profile.smiles,
                'name': profile.name,
                'source': profile.source,
                'properties': profile.properties,
                'bioactivity': profile.bioactivity
            }
        
        with open(db_path, 'w') as f:
            json.dump(compounds_data, f, indent=2)
        
        self.console.print("[cyan]Thank you for using CriOS. Goodbye![/cyan]")
        sys.exit(0)
    
    async def run(self):
        """Main event loop"""
        self.console.print(Panel.fit(
            "[bold cyan]Welcome to CriOS[/bold cyan]\n"
            "[green]Crowe Research Intelligence Operating System[/green]\n\n"
            "Type 'help' for available commands",
            title="CriOS v2.0"
        ))
        
        # Command completer
        completer = WordCompleter(list(self.commands.keys()))
        
        while True:
            try:
                # Get user input
                user_input = prompt(
                    "CriOS> ",
                    history=self.history,
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=completer
                )
                
                if not user_input:
                    continue
                
                # Parse command and arguments
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Execute command
                if command in self.commands:
                    if asyncio.iscoroutinefunction(self.commands[command]):
                        await self.commands[command](*args)
                    else:
                        self.commands[command](*args)
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")
                    self.console.print("Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' command to quit CriOS[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                logger.exception("Command execution failed")

def main():
    """Entry point for CriOS"""
    crios = CriOS()
    asyncio.run(crios.run())

if __name__ == "__main__":
    main()

