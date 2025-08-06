import logging
import joblib
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

class MLModelManager:
    """Machine learning model manager for exoplanet detection."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
    
    def train_model(self, model_name, training_data, features, target):
        """Train a machine learning model."""
        try:
            # Prepare data
            X = training_data[features]
            y = training_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = model.score(X_test_scaled, y_test)
            
            # Save model and scaler
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Save performance metrics
            self._save_model_performance(model_name, accuracy, len(X_test), features)
            
            self.logger.info(f"Model {model_name} trained with accuracy: {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Failed to train model {model_name}: {e}")
            return None
    
    def predict(self, model_name, features):
        """Make predictions using a trained model."""
        try:
            # Load model if not in memory
            if model_name not in self.models:
                model_path = self.models_dir / f"{model_name}.pkl"
                scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
                
                if model_path.exists() and scaler_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                else:
                    raise FileNotFoundError(f"Model {model_name} not found")
            
            # Prepare features
            feature_vector = np.array([features.get(f, 0) for f in self._get_feature_names(model_name)])
            feature_vector = feature_vector.reshape(1, -1)
            
            # Scale features
            scaler = self.scalers[model_name]
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(feature_vector_scaled)[0]
            confidence = model.predict_proba(feature_vector_scaled)[0].max()
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to make prediction with model {model_name}: {e}")
            return {'prediction': 'unknown', 'confidence': 0.0, 'model_name': model_name}
    
    def _get_feature_names(self, model_name):
        """Get feature names for a model."""
        # Default feature names for exoplanet detection
        return [
            'bls_power', 'period', 'transit_depth', 'snr', 'transit_duration',
            'odd_even_ratio', 'secondary_eclipse', 'stellar_variability',
            'data_quality', 'observation_count'
        ]
    
    def _save_model_performance(self, model_name, accuracy, test_size, features):
        """Save model performance metrics."""
        try:
            performance_data = {
                'model_name': model_name,
                'accuracy': accuracy,
                'test_size': test_size,
                'features': features,
                'timestamp': str(np.datetime64('now'))
            }
            
            performance_path = self.models_dir / f"{model_name}_performance.json"
            import json
            with open(performance_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model performance: {e}")
    
    def load_model(self, model_name):
        """Load a model into memory."""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(scaler_path)
                self.logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                self.logger.warning(f"Model {model_name} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False 