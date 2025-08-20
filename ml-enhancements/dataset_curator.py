#!/usr/bin/env python3
"""
Advanced ML Dataset Curation System with Automated Labeling
Part of the Crowe ML Pipeline
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm
import joblib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import asyncio
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetMetadata:
    """Metadata for curated datasets"""
    name: str
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    source_files: List[str] = field(default_factory=list)
    num_samples: int = 0
    num_features: int = 0
    num_labels: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    feature_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    curation_strategy: str = "unknown"
    splits: Dict[str, List[int]] = field(default_factory=dict)

class MolecularDataset(Dataset):
    """PyTorch dataset for molecular data"""
    
    def __init__(self, smiles_list: List[str], labels: Optional[np.ndarray] = None):
        self.smiles = smiles_list
        self.labels = labels
        self.features = []
        self._precompute_features()
    
    def _precompute_features(self):
        """Precompute molecular features"""
        for smi in tqdm(self.smiles, desc="Computing features"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                self.features.append(np.array(fp))
            else:
                self.features.append(np.zeros(2048))
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        sample = {
            'smiles': self.smiles[idx],
            'features': torch.FloatTensor(self.features[idx])
        }
        if self.labels is not None:
            sample['label'] = torch.FloatTensor([self.labels[idx]])
        return sample

class ActiveLearner:
    """Active learning for intelligent sample selection"""
    
    def __init__(self, model: Any, query_strategy: str = 'uncertainty'):
        self.model = model
        self.query_strategy = query_strategy
        self.labeled_indices = set()
        self.unlabeled_indices = set()
    
    def query(self, X: np.ndarray, n_instances: int = 10) -> List[int]:
        """Select most informative samples for labeling"""
        if self.query_strategy == 'uncertainty':
            return self._uncertainty_sampling(X, n_instances)
        elif self.query_strategy == 'diversity':
            return self._diversity_sampling(X, n_instances)
        elif self.query_strategy == 'hybrid':
            return self._hybrid_sampling(X, n_instances)
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")
    
    def _uncertainty_sampling(self, X: np.ndarray, n: int) -> List[int]:
        """Select samples with highest prediction uncertainty"""
        unlabeled_X = X[list(self.unlabeled_indices)]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(unlabeled_X)
            uncertainty = 1 - np.max(proba, axis=1)
        else:
            # For neural networks
            predictions = []
            for _ in range(10):  # Monte Carlo dropout
                pred = self.model(torch.FloatTensor(unlabeled_X))
                predictions.append(pred.detach().numpy())
            uncertainty = np.std(predictions, axis=0).mean(axis=1)
        
        indices = np.argsort(uncertainty)[-n:]
        return [list(self.unlabeled_indices)[i] for i in indices]
    
    def _diversity_sampling(self, X: np.ndarray, n: int) -> List[int]:
        """Select diverse samples using clustering"""
        from sklearn.cluster import KMeans
        
        unlabeled_X = X[list(self.unlabeled_indices)]
        kmeans = KMeans(n_clusters=min(n, len(unlabeled_X)))
        kmeans.fit(unlabeled_X)
        
        # Select sample closest to each cluster center
        selected = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(unlabeled_X - center, axis=1)
            closest = np.argmin(distances)
            selected.append(list(self.unlabeled_indices)[closest])
        
        return selected[:n]
    
    def _hybrid_sampling(self, X: np.ndarray, n: int) -> List[int]:
        """Combine uncertainty and diversity"""
        n_uncertainty = n // 2
        n_diversity = n - n_uncertainty
        
        uncertain = self._uncertainty_sampling(X, n_uncertainty)
        diverse = self._diversity_sampling(X, n_diversity)
        
        return list(set(uncertain + diverse))[:n]

class DatasetCurator:
    """Advanced dataset curation with automated labeling"""
    
    def __init__(self, workspace: str = "./curated_datasets"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)
        
        self.metadata_cache: Dict[str, DatasetMetadata] = {}
        self.active_learners: Dict[str, ActiveLearner] = {}
        self.label_models: Dict[str, Any] = {}
        self.quality_threshold = 0.8
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize labeling models"""
        # Pre-trained models for different tasks
        self.label_models['toxicity'] = RandomForestClassifier(n_estimators=100)
        self.label_models['activity'] = RandomForestClassifier(n_estimators=100)
        self.label_models['solubility'] = RandomForestClassifier(n_estimators=50)
        
        # Semi-supervised models
        self.label_models['semi_supervised'] = LabelPropagation(kernel='rbf', gamma=20)
        
        logger.info("Initialized labeling models")
    
    async def curate_dataset(
        self,
        data_source: Union[str, pd.DataFrame],
        dataset_name: str,
        labeling_strategy: str = 'supervised',
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> DatasetMetadata:
        """Curate a dataset with automated labeling and quality control"""
        
        logger.info(f"Starting curation of dataset: {dataset_name}")
        
        # Load data
        if isinstance(data_source, str):
            df = await self._load_data(data_source)
        else:
            df = data_source
        
        # Data validation
        df = self._validate_and_clean(df)
        
        # Feature extraction
        if 'smiles' in df.columns:
            features = await self._extract_molecular_features(df['smiles'].tolist())
        elif feature_columns:
            features = df[feature_columns].values
        else:
            features = df.select_dtypes(include=[np.number]).values
        
        # Labeling
        if labeling_strategy == 'supervised' and target_column:
            labels = df[target_column].values
        elif labeling_strategy == 'semi_supervised':
            labels = await self._semi_supervised_labeling(features, df.get(target_column))
        elif labeling_strategy == 'active_learning':
            labels = await self._active_learning_labeling(features, dataset_name)
        elif labeling_strategy == 'weak_supervision':
            labels = await self._weak_supervision_labeling(df)
        else:
            labels = None
        
        # Quality assessment
        quality_metrics = self._assess_quality(features, labels)
        
        # Create splits
        splits = self._create_splits(
            len(features),
            labels,
            validation_split,
            test_split
        )
        
        # Create metadata
        metadata = DatasetMetadata(
            name=dataset_name,
            version="1.0.0",
            num_samples=len(features),
            num_features=features.shape[1],
            num_labels=len(np.unique(labels)) if labels is not None else 0,
            label_distribution=self._get_label_distribution(labels),
            feature_statistics=self._compute_feature_statistics(features),
            quality_metrics=quality_metrics,
            curation_strategy=labeling_strategy,
            splits=splits
        )
        
        # Save curated dataset
        await self._save_dataset(dataset_name, features, labels, metadata)
        
        # Cache metadata
        self.metadata_cache[dataset_name] = metadata
        
        logger.info(f"Dataset curation complete: {dataset_name}")
        return metadata
    
    async def _load_data(self, path: str) -> pd.DataFrame:
        """Load data from various formats"""
        path = Path(path)
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.sdf':
            return self._load_sdf(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _load_sdf(self, path: Path) -> pd.DataFrame:
        """Load SDF file with molecular structures"""
        supplier = Chem.SDMolSupplier(str(path))
        data = []
        
        for mol in supplier:
            if mol:
                data.append({
                    'smiles': Chem.MolToSmiles(mol),
                    'name': mol.GetProp('_Name') if mol.HasProp('_Name') else None,
                    **{prop: mol.GetProp(prop) for prop in mol.GetPropNames()}
                })
        
        return pd.DataFrame(data)
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean dataset"""
        initial_size = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Validate SMILES if present
        if 'smiles' in df.columns:
            valid_smiles = []
            for smi in df['smiles']:
                mol = Chem.MolFromSmiles(str(smi))
                valid_smiles.append(mol is not None)
            df = df[valid_smiles]
        
        final_size = len(df)
        logger.info(f"Data cleaning: {initial_size} -> {final_size} samples")
        
        return df
    
    async def _extract_molecular_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract molecular features asynchronously"""
        features = []
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for batch in self._batch_generator(smiles_list, 100):
                future = executor.submit(self._compute_batch_features, batch)
                futures.append(future)
            
            for future in tqdm(futures, desc="Extracting features"):
                batch_features = future.result()
                features.extend(batch_features)
        
        return np.array(features)
    
    def _compute_batch_features(self, smiles_batch: List[str]) -> List[np.ndarray]:
        """Compute features for a batch of molecules"""
        features = []
        
        for smi in smiles_batch:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                
                # Additional descriptors
                descriptors = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumHeteroatoms(mol)
                ]
                
                # Combine features
                combined = np.concatenate([np.array(fp), descriptors])
                features.append(combined)
            else:
                features.append(np.zeros(2056))  # 2048 + 8
        
        return features
    
    async def _semi_supervised_labeling(
        self,
        features: np.ndarray,
        partial_labels: Optional[pd.Series] = None
    ) -> np.ndarray:
        """Semi-supervised labeling using label propagation"""
        
        if partial_labels is None:
            # Start with a small set of labeled samples
            n_labeled = min(100, len(features) // 10)
            labels = np.full(len(features), -1)
            
            # Randomly assign initial labels (would be manually labeled in practice)
            labeled_indices = np.random.choice(len(features), n_labeled, replace=False)
            labels[labeled_indices] = np.random.randint(0, 2, n_labeled)
        else:
            labels = partial_labels.values
            labels[pd.isna(labels)] = -1
        
        # Train label propagation model
        model = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
        model.fit(features, labels)
        
        # Predict labels for unlabeled samples
        predicted_labels = model.transduction_
        
        # Confidence scores
        label_distributions = model.label_distributions_
        confidence = np.max(label_distributions, axis=1)
        
        # Only accept high-confidence predictions
        mask = confidence > self.quality_threshold
        final_labels = labels.copy()
        final_labels[mask & (labels == -1)] = predicted_labels[mask & (labels == -1)]
        
        logger.info(f"Semi-supervised labeling: {np.sum(final_labels != -1)} labeled samples")
        
        return final_labels
    
    async def _active_learning_labeling(
        self,
        features: np.ndarray,
        dataset_name: str
    ) -> np.ndarray:
        """Active learning for efficient labeling"""
        
        # Initialize with small labeled set
        n_initial = min(50, len(features) // 20)
        labels = np.full(len(features), -1)
        
        # Create active learner
        model = RandomForestClassifier(n_estimators=100)
        active_learner = ActiveLearner(model, query_strategy='hybrid')
        
        # Initial random labeling
        initial_indices = np.random.choice(len(features), n_initial, replace=False)
        labels[initial_indices] = np.random.randint(0, 2, n_initial)
        active_learner.labeled_indices = set(initial_indices)
        active_learner.unlabeled_indices = set(range(len(features))) - active_learner.labeled_indices
        
        # Active learning loop
        n_queries = 10
        batch_size = 20
        
        for iteration in range(n_queries):
            # Train model on labeled data
            labeled_features = features[list(active_learner.labeled_indices)]
            labeled_labels = labels[list(active_learner.labeled_indices)]
            model.fit(labeled_features, labeled_labels)
            
            # Query most informative samples
            query_indices = active_learner.query(features, batch_size)
            
            # Simulate labeling (would be manual in practice)
            for idx in query_indices:
                # Use model prediction with noise as simulated label
                pred = model.predict(features[idx:idx+1])[0]
                noise = np.random.uniform() < 0.1  # 10% label noise
                labels[idx] = 1 - pred if noise else pred
                
                active_learner.labeled_indices.add(idx)
                active_learner.unlabeled_indices.discard(idx)
            
            logger.info(f"Active learning iteration {iteration+1}: {len(active_learner.labeled_indices)} labeled")
        
        # Final prediction for remaining unlabeled
        if active_learner.unlabeled_indices:
            model.fit(
                features[list(active_learner.labeled_indices)],
                labels[list(active_learner.labeled_indices)]
            )
            
            unlabeled_features = features[list(active_learner.unlabeled_indices)]
            predictions = model.predict(unlabeled_features)
            
            for i, idx in enumerate(active_learner.unlabeled_indices):
                labels[idx] = predictions[i]
        
        self.active_learners[dataset_name] = active_learner
        
        return labels
    
    async def _weak_supervision_labeling(self, df: pd.DataFrame) -> np.ndarray:
        """Weak supervision using labeling functions"""
        
        labels = []
        
        for _, row in df.iterrows():
            # Define labeling functions (domain-specific rules)
            votes = []
            
            # Example labeling functions for drug-likeness
            if 'molecular_weight' in row:
                votes.append(1 if 150 < row['molecular_weight'] < 500 else 0)
            
            if 'logp' in row:
                votes.append(1 if -0.4 < row['logp'] < 5.6 else 0)
            
            if 'smiles' in row:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    # Check for toxic substructures
                    toxic_patterns = ['[N+](=O)[O-]', 'C(=O)Cl', '[SH]']
                    has_toxic = any(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                                  for p in toxic_patterns)
                    votes.append(0 if has_toxic else 1)
            
            # Majority voting
            if votes:
                label = 1 if sum(votes) > len(votes) / 2 else 0
            else:
                label = -1  # Unknown
            
            labels.append(label)
        
        return np.array(labels)
    
    def _assess_quality(self, features: np.ndarray, labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Assess dataset quality"""
        metrics = {}
        
        # Feature quality
        metrics['feature_completeness'] = 1 - np.sum(np.isnan(features)) / features.size
        metrics['feature_variance'] = np.mean(np.var(features, axis=0))
        
        # Label quality
        if labels is not None:
            unique_labels = np.unique(labels[labels != -1])
            metrics['label_completeness'] = np.sum(labels != -1) / len(labels)
            metrics['label_balance'] = min(np.bincount(labels[labels != -1])) / max(np.bincount(labels[labels != -1]))
            metrics['num_classes'] = len(unique_labels)
        
        # Overall quality score
        metrics['overall_quality'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _create_splits(
        self,
        n_samples: int,
        labels: Optional[np.ndarray],
        val_split: float,
        test_split: float
    ) -> Dict[str, List[int]]:
        """Create train/val/test splits"""
        
        indices = np.arange(n_samples)
        
        if labels is not None and labels.ndim == 1:
            # Stratified split
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_split,
                stratify=labels[labels != -1] if -1 in labels else labels,
                random_state=42
            )
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_split / (1 - test_split),
                stratify=labels[train_val_idx],
                random_state=42
            )
        else:
            # Random split
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=test_split,
                random_state=42
            )
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_split / (1 - test_split),
                random_state=42
            )
        
        return {
            'train': train_idx.tolist(),
            'validation': val_idx.tolist(),
            'test': test_idx.tolist()
        }
    
    def _get_label_distribution(self, labels: Optional[np.ndarray]) -> Dict[str, int]:
        """Get label distribution"""
        if labels is None:
            return {}
        
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        return {str(label): int(count) for label, count in zip(unique, counts)}
    
    def _compute_feature_statistics(self, features: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute feature statistics"""
        stats = {}
        
        for i in range(min(10, features.shape[1])):  # First 10 features
            stats[f'feature_{i}'] = {
                'mean': float(np.mean(features[:, i])),
                'std': float(np.std(features[:, i])),
                'min': float(np.min(features[:, i])),
                'max': float(np.max(features[:, i]))
            }
        
        return stats
    
    async def _save_dataset(
        self,
        name: str,
        features: np.ndarray,
        labels: Optional[np.ndarray],
        metadata: DatasetMetadata
    ):
        """Save curated dataset"""
        dataset_dir = self.workspace / name
        dataset_dir.mkdir(exist_ok=True)
        
        # Save features
        np.save(dataset_dir / 'features.npy', features)
        
        # Save labels
        if labels is not None:
            np.save(dataset_dir / 'labels.npy', labels)
        
        # Save metadata
        with open(dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata.__dict__, f, indent=2, default=str)
        
        # Create PyTorch dataset file
        torch.save({
            'features': torch.FloatTensor(features),
            'labels': torch.FloatTensor(labels) if labels is not None else None,
            'metadata': metadata
        }, dataset_dir / 'dataset.pt')
        
        logger.info(f"Dataset saved to {dataset_dir}")
    
    def _batch_generator(self, data: List, batch_size: int):
        """Generate batches from data"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def load_dataset(self, name: str) -> Tuple[np.ndarray, Optional[np.ndarray], DatasetMetadata]:
        """Load a curated dataset"""
        dataset_dir = self.workspace / name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset not found: {name}")
        
        # Load features
        features = np.load(dataset_dir / 'features.npy')
        
        # Load labels if exist
        labels_path = dataset_dir / 'labels.npy'
        labels = np.load(labels_path) if labels_path.exists() else None
        
        # Load metadata
        with open(dataset_dir / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
            metadata = DatasetMetadata(**metadata_dict)
        
        return features, labels, metadata
    
    def get_pytorch_dataset(self, name: str) -> MolecularDataset:
        """Get PyTorch dataset"""
        features, labels, metadata = self.load_dataset(name)
        
        # Convert back to SMILES if needed (simplified)
        smiles = [f"PLACEHOLDER_{i}" for i in range(len(features))]
        
        return MolecularDataset(smiles, labels)

# CLI Interface
async def main():
    """CLI for dataset curation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crowe ML Dataset Curator")
    parser.add_argument('--input', required=True, help="Input data file")
    parser.add_argument('--name', required=True, help="Dataset name")
    parser.add_argument('--strategy', default='supervised',
                       choices=['supervised', 'semi_supervised', 'active_learning', 'weak_supervision'])
    parser.add_argument('--target', help="Target column for supervised learning")
    parser.add_argument('--features', nargs='+', help="Feature columns")
    parser.add_argument('--workspace', default='./curated_datasets')
    
    args = parser.parse_args()
    
    curator = DatasetCurator(args.workspace)
    
    metadata = await curator.curate_dataset(
        data_source=args.input,
        dataset_name=args.name,
        labeling_strategy=args.strategy,
        target_column=args.target,
        feature_columns=args.features
    )
    
    print(f"\nDataset curated successfully!")
    print(f"Name: {metadata.name}")
    print(f"Samples: {metadata.num_samples}")
    print(f"Features: {metadata.num_features}")
    print(f"Quality Score: {metadata.quality_metrics.get('overall_quality', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(main())

