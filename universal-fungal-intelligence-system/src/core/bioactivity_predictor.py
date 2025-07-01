import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class BioactivityPredictor:
    """
    BioactivityPredictor class assesses the bioactivity of chemical compounds
    using machine learning models trained on molecular descriptors.
    """

    def __init__(self):
        """Initialize the BioactivityPredictor."""
        self.activity_classifier = None
        self.potency_regressor = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'molecular_weight', 'logP', 'num_h_donors', 'num_h_acceptors',
            'tpsa', 'num_rotatable_bonds', 'num_aromatic_rings', 'lipinski_violations'
        ]
        
        # Initialize or load models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models or load pre-trained ones."""
        model_dir = "models"
        classifier_path = os.path.join(model_dir, "bioactivity_classifier.pkl")
        regressor_path = os.path.join(model_dir, "potency_regressor.pkl")
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        
        if os.path.exists(classifier_path) and os.path.exists(regressor_path):
            # Load pre-trained models
            logger.info("Loading pre-trained bioactivity models...")
            self.activity_classifier = joblib.load(classifier_path)
            self.potency_regressor = joblib.load(regressor_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # Create new models with default parameters
            logger.info("Initializing new bioactivity models...")
            self.activity_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.potency_regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            # Train on synthetic data for demonstration
            self._train_on_synthetic_data()
    
    def _train_on_synthetic_data(self):
        """Train models on synthetic data for demonstration purposes."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, len(self.feature_names))
        
        # Adjust feature distributions to be more realistic
        X[:, 0] = np.abs(X[:, 0]) * 100 + 300  # molecular_weight: 200-500
        X[:, 1] = X[:, 1] * 2 + 2  # logP: -2 to 6
        X[:, 2] = np.abs(X[:, 2].astype(int)) % 6  # h_donors: 0-5
        X[:, 3] = np.abs(X[:, 3].astype(int)) % 11  # h_acceptors: 0-10
        X[:, 4] = np.abs(X[:, 4]) * 50 + 50  # tpsa: 0-150
        X[:, 5] = np.abs(X[:, 5].astype(int)) % 15  # rotatable_bonds: 0-15
        X[:, 6] = np.abs(X[:, 6].astype(int)) % 5  # aromatic_rings: 0-4
        X[:, 7] = np.abs(X[:, 7].astype(int)) % 5  # lipinski_violations: 0-4
        
        # Generate labels based on feature relationships
        # Compounds with good drug-like properties are more likely to be active
        activity_score = (
            (X[:, 0] < 450).astype(float) * 0.2 +  # MW < 450
            (X[:, 1] > 0).astype(float) * (X[:, 1] < 5).astype(float) * 0.2 +  # 0 < logP < 5
            (X[:, 2] <= 5).astype(float) * 0.15 +  # h_donors <= 5
            (X[:, 3] <= 10).astype(float) * 0.15 +  # h_acceptors <= 10
            (X[:, 4] < 140).astype(float) * 0.1 +  # tpsa < 140
            (X[:, 7] <= 1).astype(float) * 0.2  # few lipinski violations
        )
        
        # Binary classification labels
        y_activity = (activity_score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        # Potency scores (0-1)
        y_potency = np.clip(activity_score + np.random.normal(0, 0.1, n_samples), 0, 1)
        
        # Fit the models
        X_scaled = self.scaler.fit_transform(X)
        self.activity_classifier.fit(X_scaled, y_activity)
        self.potency_regressor.fit(X_scaled, y_potency)
        
        logger.info("Models trained on synthetic data")

    def predict_bioactivity(self, compound_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the bioactivity of a given compound.

        Args:
            compound_data: A dictionary containing compound information.

        Returns:
            Dict with predicted bioactivity results.
        """
        try:
            # Extract features
            features = self._extract_features(compound_data)
            
            if features is None:
                return self._get_default_prediction(compound_data)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict activity class
            activity_proba = self.activity_classifier.predict_proba(features_scaled)[0]
            is_active = self.activity_classifier.predict(features_scaled)[0]
            
            # Predict potency
            potency_score = self.potency_regressor.predict(features_scaled)[0]
            
            # Get feature importance for interpretability
            feature_importance = self._get_feature_importance(features_scaled)
            
            # Determine activity level
            if is_active and potency_score > 0.8:
                activity_level = "Highly Active"
            elif is_active and potency_score > 0.6:
                activity_level = "Active"
            elif is_active:
                activity_level = "Moderately Active"
            else:
                activity_level = "Inactive"
            
            return {
                'compound': compound_data.get('name', 'Unknown'),
                'predicted_activity': activity_level,
                'confidence_score': float(activity_proba[1]),  # Probability of being active
                'potency_score': float(potency_score),
                'is_active': bool(is_active),
                'key_features': feature_importance,
                'therapeutic_potential': self._assess_therapeutic_potential(
                    is_active, potency_score, compound_data
                )
            }
            
        except Exception as e:
            logger.error(f"Error predicting bioactivity: {str(e)}")
            return self._get_default_prediction(compound_data)
    
    def _extract_features(self, compound_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from compound data."""
        try:
            features = []
            for feature_name in self.feature_names:
                if feature_name in compound_data:
                    features.append(float(compound_data[feature_name]))
                else:
                    # Handle missing features
                    if feature_name == 'lipinski_violations':
                        features.append(0)
                    elif feature_name == 'num_rotatable_bonds':
                        features.append(5)  # Default average
                    elif feature_name == 'num_aromatic_rings':
                        features.append(2)  # Default average
                    else:
                        logger.warning(f"Missing feature: {feature_name}")
                        return None
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def _get_feature_importance(self, features_scaled: np.ndarray) -> Dict[str, float]:
        """Get feature importance for the prediction."""
        importances = self.activity_classifier.feature_importances_
        
        # Create importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if importances[i] > 0.05:  # Only show important features
                feature_importance[feature_name] = float(importances[i])
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def _assess_therapeutic_potential(self, is_active: bool, potency: float, 
                                    compound_data: Dict[str, Any]) -> str:
        """Assess the therapeutic potential of a compound."""
        if not is_active:
            return "Low"
        
        # Consider multiple factors
        factors = 0
        
        if potency > 0.7:
            factors += 2
        elif potency > 0.5:
            factors += 1
        
        # Check drug-likeness
        if compound_data.get('drug_likeness') in ['Excellent', 'Good']:
            factors += 1
        
        # Check if it has known bioactivity
        bioactivity = compound_data.get('bioactivity', {})
        if bioactivity.get('active_assays', 0) > 10:
            factors += 2
        elif bioactivity.get('active_assays', 0) > 5:
            factors += 1
        
        # Determine potential
        if factors >= 4:
            return "High"
        elif factors >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _get_default_prediction(self, compound_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return default prediction when features are missing."""
        # Use simple heuristics based on available data
        drug_likeness = compound_data.get('drug_likeness', 'Unknown')
        bioactivity = compound_data.get('bioactivity', {})
        
        if drug_likeness == 'Excellent' and bioactivity.get('active_assays', 0) > 5:
            return {
                'compound': compound_data.get('name', 'Unknown'),
                'predicted_activity': 'Likely Active',
                'confidence_score': 0.75,
                'potency_score': 0.7,
                'is_active': True,
                'key_features': {'drug_likeness': 1.0, 'known_bioactivity': 1.0},
                'therapeutic_potential': 'Moderate'
            }
        else:
            return {
                'compound': compound_data.get('name', 'Unknown'),
                'predicted_activity': 'Unknown',
                'confidence_score': 0.5,
                'potency_score': 0.5,
                'is_active': False,
                'key_features': {},
                'therapeutic_potential': 'Unknown'
            }

    def assess_bioactivity(self, compounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assess the bioactivity of multiple compounds.

        Args:
            compounds: A list of dictionaries containing compound information.

        Returns:
            List of dictionaries with bioactivity assessment results.
        """
        results = []
        for compound in compounds:
            result = self.predict_bioactivity(compound)
            results.append(result)
        return results
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk."""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.activity_classifier, os.path.join(model_dir, "bioactivity_classifier.pkl"))
        joblib.dump(self.potency_regressor, os.path.join(model_dir, "potency_regressor.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "feature_scaler.pkl"))
        
        logger.info(f"Models saved to {model_dir}")