import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import asyncio
from typing import Dict, List, Any, Tuple
import os
import json
from datetime import datetime

# Import data collectors
from ...data.collectors.pubchem_client import PubChemClient
from ...data.collectors.mycobank_client import MycoBankClient
from ...data.collectors.ncbi_client import NCBIClient
from ...core.molecular_analyzer import MolecularAnalyzer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train machine learning models on real fungal compound data from multiple sources.
    """
    
    def __init__(self, output_dir: str = "models"):
        """Initialize the model trainer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.molecular_analyzer = MolecularAnalyzer()
        self.scaler = StandardScaler()
        
        # Feature columns for training
        self.feature_columns = [
            'molecular_weight', 'logP', 'num_h_donors', 'num_h_acceptors',
            'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings', 'lipinski_violations'
        ]
        
        # Models
        self.activity_classifier = None
        self.potency_regressor = None
        
    async def collect_training_data(self, num_compounds: int = 1000) -> pd.DataFrame:
        """
        Collect training data from all available sources.
        
        Args:
            num_compounds: Target number of compounds to collect
            
        Returns:
            DataFrame with compound data and features
        """
        logger.info("Starting comprehensive data collection...")
        
        all_compounds = []
        
        # 1. Collect from PubChem
        logger.info("Collecting from PubChem...")
        pubchem_compounds = await self._collect_from_pubchem(num_compounds // 3)
        all_compounds.extend(pubchem_compounds)
        
        # 2. Collect from MycoBank + PubChem integration
        logger.info("Collecting from MycoBank...")
        mycobank_compounds = await self._collect_from_mycobank(num_compounds // 3)
        all_compounds.extend(mycobank_compounds)
        
        # 3. Collect from NCBI literature
        logger.info("Collecting from NCBI...")
        ncbi_compounds = await self._collect_from_ncbi(num_compounds // 3)
        all_compounds.extend(ncbi_compounds)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_compounds)
        
        # Remove duplicates based on SMILES
        if 'smiles' in df.columns:
            df = df.drop_duplicates(subset=['smiles'])
        
        logger.info(f"Collected {len(df)} unique compounds")
        
        # Calculate molecular features
        logger.info("Calculating molecular features...")
        df = self._calculate_features(df)
        
        # Add activity labels based on known bioactivity
        df = self._add_activity_labels(df)
        
        return df
    
    async def _collect_from_pubchem(self, num_compounds: int) -> List[Dict[str, Any]]:
        """Collect compounds from PubChem."""
        compounds = []
        
        pubchem_client = PubChemClient()
        
        try:
            # Search for various fungal compound types
            search_terms = [
                "fungal antibiotic active",
                "mushroom anticancer",
                "fungal metabolite bioactive",
                "mycotoxin pharmacology",
                "endophytic fungus drug"
            ]
            
            compounds_per_term = num_compounds // len(search_terms)
            
            for term in search_terms:
                logger.info(f"Searching PubChem for: {term}")
                results = pubchem_client.search_fungal_compounds(term, max_results=compounds_per_term)
                
                for compound in results:
                    # Ensure we have essential data
                    if compound.get('smiles') and compound.get('bioactivity'):
                        compounds.append({
                            'source': 'PubChem',
                            'cid': compound.get('cid'),
                            'name': compound.get('name'),
                            'smiles': compound.get('smiles'),
                            'molecular_formula': compound.get('molecular_formula'),
                            'active_assays': compound.get('bioactivity', {}).get('active_assays', 0),
                            'total_assays': compound.get('bioactivity', {}).get('total_assays', 0),
                            'activity_ratio': compound.get('bioactivity', {}).get('active_assays', 0) / 
                                            max(compound.get('bioactivity', {}).get('total_assays', 1), 1)
                        })
                
        finally:
            pubchem_client.close()
        
        return compounds
    
    async def _collect_from_mycobank(self, num_compounds: int) -> List[Dict[str, Any]]:
        """Collect compounds from MycoBank species."""
        compounds = []
        
        async with MycoBankClient() as mycobank_client:
            pubchem_client = PubChemClient()
            
            try:
                # Get fungal species
                species_list = await mycobank_client.fetch_all_species(limit=50)
                
                for species in species_list:
                    # Get metabolites for this species
                    metabolites = await mycobank_client.get_species_metabolites(
                        species.get('scientific_name', '')
                    )
                    
                    for metabolite in metabolites:
                        # Look up compound in PubChem
                        if metabolite.get('pubchem_cid'):
                            compound_data = pubchem_client.get_compound_by_cid(
                                metabolite['pubchem_cid']
                            )
                            
                            if compound_data.get('smiles'):
                                compounds.append({
                                    'source': 'MycoBank+PubChem',
                                    'species': species.get('scientific_name'),
                                    'cid': compound_data.get('cid'),
                                    'name': metabolite.get('name'),
                                    'smiles': compound_data.get('smiles'),
                                    'metabolite_type': metabolite.get('type'),
                                    'active_assays': compound_data.get('bioactivity', {}).get('active_assays', 0),
                                    'total_assays': compound_data.get('bioactivity', {}).get('total_assays', 0)
                                })
                                
                                if len(compounds) >= num_compounds:
                                    break
                    
                    if len(compounds) >= num_compounds:
                        break
                        
            finally:
                pubchem_client.close()
        
        return compounds
    
    async def _collect_from_ncbi(self, num_compounds: int) -> List[Dict[str, Any]]:
        """Collect compounds from NCBI literature."""
        compounds = []
        
        ncbi_client = NCBIClient()
        pubchem_client = PubChemClient()
        
        try:
            # Search for fungal metabolites in literature
            search_queries = [
                "fungal secondary metabolite structure",
                "mushroom bioactive compound isolation",
                "endophytic fungus natural product"
            ]
            
            compounds_per_query = num_compounds // len(search_queries)
            
            for query in search_queries:
                publications = ncbi_client.search_fungi(query, database="pubmed", max_results=20)
                
                # Extract compound names from abstracts
                for pub in publications:
                    abstract = pub.get('abstract', '')
                    
                    # Simple extraction of compound names (in production, use NLP)
                    compound_patterns = [
                        r'compound\s+(\w+)',
                        r'isolated\s+(\w+)',
                        r'metabolite\s+(\w+)'
                    ]
                    
                    import re
                    for pattern in compound_patterns:
                        matches = re.findall(pattern, abstract, re.IGNORECASE)
                        for compound_name in matches[:2]:  # Limit per publication
                            # Try to find in PubChem
                            compound_data = pubchem_client.get_compound_by_name(compound_name)
                            
                            if compound_data.get('smiles'):
                                compounds.append({
                                    'source': 'NCBI+PubChem',
                                    'pmid': pub.get('pmid'),
                                    'name': compound_name,
                                    'cid': compound_data.get('cid'),
                                    'smiles': compound_data.get('smiles'),
                                    'publication_year': pub.get('year'),
                                    'active_assays': compound_data.get('bioactivity', {}).get('active_assays', 0),
                                    'total_assays': compound_data.get('bioactivity', {}).get('total_assays', 0)
                                })
                                
                                if len(compounds) >= num_compounds:
                                    return compounds
                                    
        finally:
            ncbi_client.close()
            pubchem_client.close()
        
        return compounds
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate molecular features for all compounds."""
        features_list = []
        
        for idx, row in df.iterrows():
            smiles = row.get('smiles', '')
            if smiles:
                try:
                    # Analyze molecular structure
                    analysis = self.molecular_analyzer.analyze_structure(smiles)
                    
                    if 'error' not in analysis:
                        features_list.append({
                            'idx': idx,
                            **{k: analysis.get(k, 0) for k in self.feature_columns}
                        })
                except Exception as e:
                    logger.error(f"Error analyzing compound {idx}: {e}")
        
        # Merge features back
        features_df = pd.DataFrame(features_list)
        if not features_df.empty:
            features_df = features_df.set_index('idx')
            df = df.join(features_df, how='inner')
        
        return df
    
    def _add_activity_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add activity labels based on bioassay data."""
        # Binary activity (active if >50% assays are positive)
        df['is_active'] = (df['active_assays'] / df['total_assays'].clip(lower=1)) > 0.5
        
        # Potency score (0-1)
        df['potency_score'] = df['active_assays'] / df['total_assays'].clip(lower=1)
        
        # Weight by number of assays (more assays = more confidence)
        df['confidence_weight'] = np.log1p(df['total_assays']) / 10
        
        return df
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train activity and potency models on the collected data.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training models on real data...")
        
        # Prepare features and labels
        feature_mask = df[self.feature_columns].notna().all(axis=1)
        df_clean = df[feature_mask].copy()
        
        if len(df_clean) < 100:
            logger.warning(f"Only {len(df_clean)} samples available for training")
        
        X = df_clean[self.feature_columns].values
        y_activity = df_clean['is_active'].astype(int).values
        y_potency = df_clean['potency_score'].values
        weights = df_clean['confidence_weight'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_act_train, y_act_test, y_pot_train, y_pot_test, w_train, w_test = \
            train_test_split(X_scaled, y_activity, y_potency, weights, test_size=0.2, random_state=42)
        
        # Train activity classifier
        logger.info("Training activity classifier...")
        self.activity_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.activity_classifier.fit(X_train, y_act_train, sample_weight=w_train)
        
        # Train potency regressor
        logger.info("Training potency regressor...")
        self.potency_regressor = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            random_state=42
        )
        self.potency_regressor.fit(X_train, y_pot_train, sample_weight=w_train)
        
        # Evaluate models
        metrics = self._evaluate_models(X_test, y_act_test, y_pot_test, w_test)
        
        # Cross-validation scores
        cv_scores_classifier = cross_val_score(
            self.activity_classifier, X_scaled, y_activity, cv=5, scoring='roc_auc'
        )
        cv_scores_regressor = cross_val_score(
            self.potency_regressor, X_scaled, y_potency, cv=5, scoring='r2'
        )
        
        metrics['cv_auc_mean'] = cv_scores_classifier.mean()
        metrics['cv_auc_std'] = cv_scores_classifier.std()
        metrics['cv_r2_mean'] = cv_scores_regressor.mean()
        metrics['cv_r2_std'] = cv_scores_regressor.std()
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_columns,
            self.activity_classifier.feature_importances_
        ))
        
        logger.info(f"Training complete. Activity AUC: {metrics['cv_auc_mean']:.3f}, "
                   f"Potency R²: {metrics['cv_r2_mean']:.3f}")
        
        return metrics
    
    def _evaluate_models(self, X_test, y_act_test, y_pot_test, weights) -> Dict[str, Any]:
        """Evaluate model performance."""
        # Activity predictions
        y_act_pred = self.activity_classifier.predict(X_test)
        y_act_proba = self.activity_classifier.predict_proba(X_test)[:, 1]
        
        # Potency predictions
        y_pot_pred = self.potency_regressor.predict(X_test)
        
        # Classification report
        class_report = classification_report(
            y_act_test, y_act_pred, output_dict=True, sample_weight=weights
        )
        
        # Regression metrics
        mse = mean_squared_error(y_pot_test, y_pot_pred, sample_weight=weights)
        r2 = r2_score(y_pot_test, y_pot_pred, sample_weight=weights)
        
        return {
            'classification_report': class_report,
            'activity_accuracy': class_report['accuracy'],
            'potency_mse': mse,
            'potency_r2': r2,
            'test_size': len(X_test)
        }
    
    def save_models(self, metrics: Dict[str, Any]):
        """Save trained models and metadata."""
        # Save models
        joblib.dump(self.activity_classifier, os.path.join(self.output_dir, "bioactivity_classifier.pkl"))
        joblib.dump(self.potency_regressor, os.path.join(self.output_dir, "potency_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(self.output_dir, "feature_scaler.pkl"))
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'metrics': metrics,
            'model_versions': {
                'activity_classifier': 'RandomForestClassifier',
                'potency_regressor': 'GradientBoostingRegressor'
            }
        }
        
        with open(os.path.join(self.output_dir, "model_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.output_dir}")
    
    async def run_training_pipeline(self, num_compounds: int = 1000):
        """Run the complete training pipeline."""
        logger.info("Starting model training pipeline...")
        
        # Collect data
        df = await self.collect_training_data(num_compounds)
        
        # Save collected data
        df.to_csv(os.path.join(self.output_dir, "training_data.csv"), index=False)
        logger.info(f"Saved {len(df)} compounds to training_data.csv")
        
        # Train models
        metrics = self.train_models(df)
        
        # Save models
        self.save_models(metrics)
        
        # Export results to BigQuery
        try:
            from ...utils.bigquery_exporter import BigQueryExporter
            exporter = BigQueryExporter()
            
            # Export training metrics
            training_metrics = {
                'dataset': 'fungal_compounds',
                'n_train': int(len(df) * 0.8),
                'n_test': int(len(df) * 0.2),
                'rmse': np.sqrt(metrics['potency_mse']),
                'r2': metrics['potency_r2'],
                'model_type': 'ensemble',
                'compound_analyzed': 'multiple',
                'bioactivity_score': metrics['activity_accuracy']
            }
            
            exporter.export_metrics(training_metrics)
            logger.info("Training metrics exported to BigQuery")
            
        except Exception as e:
            logger.error(f"Failed to export to BigQuery: {e}")
        
        return metrics


async def main():
    """Run the training pipeline."""
    trainer = ModelTrainer()
    metrics = await trainer.run_training_pipeline(num_compounds=500)
    print("\nTraining complete!")
    print(f"Activity accuracy: {metrics['activity_accuracy']:.3f}")
    print(f"Potency R²: {metrics['potency_r2']:.3f}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())