import logging
import numpy as np

class MLPredictor:
    """Machine learning predictor for exoplanet detection."""
    
    def __init__(self, ml_manager):
        self.ml_manager = ml_manager
        self.logger = logging.getLogger(__name__)
    
    def predict_exoplanet(self, features):
        """Predict exoplanet probability using ML models."""
        try:
            # Prepare features for prediction
            feature_vector = self._prepare_features(features)
            
            # Try to use trained model
            if 'exoplanet_classifier' in self.ml_manager.models:
                prediction = self.ml_manager.predict('exoplanet_classifier', feature_vector)
                return prediction
            else:
                # Use fallback prediction
                return self._fallback_prediction(features)
                
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(features)
    
    def _prepare_features(self, features):
        """Prepare features for ML prediction."""
        try:
            # Ensure all required features are present
            required_features = [
                'bls_power', 'period', 'transit_depth', 'snr', 'transit_duration',
                'odd_even_ratio', 'secondary_eclipse', 'stellar_variability',
                'data_quality', 'observation_count'
            ]
            
            prepared_features = {}
            for feature in required_features:
                prepared_features[feature] = features.get(feature, 0.0)
            
            return prepared_features
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return features
    
    def _fallback_prediction(self, features):
        """Fallback prediction using rule-based approach."""
        try:
            confidence = 0.0
            
            # Rule-based scoring
            if features.get('snr', 0) > 7.0:
                confidence += 0.3
            if features.get('transit_depth', 0) > 0.001:
                confidence += 0.2
            if features.get('bls_power', 0) > 0.1:
                confidence += 0.2
            if features.get('data_quality', 0) > 0.7:
                confidence += 0.3
            
            return {
                'confidence': min(confidence, 1.0),
                'prediction': 'candidate' if confidence > 0.5 else 'no_candidate',
                'method': 'fallback_rule_based'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {e}")
            return {
                'confidence': 0.0,
                'prediction': 'unknown',
                'method': 'fallback_failed'
            }
    
    def train_model(self, training_data):
        """Train ML model with provided data."""
        try:
            # Define features and target
            features = [
                'bls_power', 'period', 'transit_depth', 'snr', 'transit_duration',
                'odd_even_ratio', 'secondary_eclipse', 'stellar_variability',
                'data_quality', 'observation_count'
            ]
            target = 'is_exoplanet'
            
            # Train model
            accuracy = self.ml_manager.train_model('exoplanet_classifier', training_data, features, target)
            
            self.logger.info(f"ML model trained with accuracy: {accuracy}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            return None 