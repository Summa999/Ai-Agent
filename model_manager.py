import os
import pickle
import json
import numpy as np
from datetime import datetime
import logging
import joblib

class ModelManager:
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
        
    def save_model(self, model, model_name, metadata=None):
        """Save model with metadata"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            # Save model using joblib (better for sklearn models)
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata['save_time'] = datetime.now().isoformat()
                metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            self.logger.info(f"Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
            return False
    
    def load_model(self, model_name):
        """Load model with metadata"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata if exists
            metadata = None
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def save_feature_importance(self, feature_importance, filename='feature_importance.pkl'):
        """Save feature importance"""
        try:
            filepath = os.path.join(self.model_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(feature_importance, f)
            self.logger.info("Feature importance saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving feature importance: {e}")
    
    def load_feature_importance(self, filename='feature_importance.pkl'):
        """Load feature importance"""
        try:
            filepath = os.path.join(self.model_dir, filename)
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading feature importance: {e}")
            return None
    
    def save_scaler(self, scaler, scaler_name='feature_scaler'):
        """Save data scaler"""
        try:
            scaler_path = os.path.join(self.model_dir, f"{scaler_name}.pkl")
            joblib.dump(scaler, scaler_path)
            self.logger.info(f"Scaler {scaler_name} saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
    
    def load_scaler(self, scaler_name='feature_scaler'):
        """Load data scaler"""
        try:
            scaler_path = os.path.join(self.model_dir, f"{scaler_name}.pkl")
            return joblib.load(scaler_path)
        except Exception as e:
            self.logger.error(f"Error loading scaler: {e}")
            return None
    
    def list_saved_models(self):
        """List all saved models"""
        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl') and not file.endswith('_metadata.pkl'):
                models.append(file[:-4])  # Remove .pkl extension
        return models
    
    def delete_model(self, model_name):
        """Delete a saved model"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            self.logger.info(f"Model {model_name} deleted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting model {model_name}: {e}")
            return False