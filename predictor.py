import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class ModelPredictor:
    def __init__(self, models: Dict, feature_engineer):
        self.models = models
        self.feature_engineer = feature_engineer
        self.logger = logging.getLogger(__name__)
        self.prediction_history = []
        
    def get_predictions(self, data: pd.DataFrame, 
                       use_ensemble: bool = True) -> Dict:
        """Get predictions from all models"""
        # Create features
        featured_data = self.feature_engineer.create_all_features(data)
        
        # Get latest features
        latest_features = featured_data[self.feature_engineer.feature_names].iloc[-1:]
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                pred = self.predict_single_model(model, latest_features, model_name)
                predictions[model_name] = pred
            except Exception as e:
                self.logger.error(f"Prediction error for {model_name}: {e}")
                
        # Ensemble predictions if requested
        if use_ensemble and len(predictions) > 1:
            ensemble_pred = self.ensemble_predictions(predictions)
            predictions['ensemble'] = ensemble_pred
        
        # Store prediction
        self.store_prediction(predictions, data.index[-1])
        
        return predictions
    
    def predict_single_model(self, model, features: pd.DataFrame, 
                           model_name: str) -> Dict:
        """Get prediction from a single model"""
        if model_name == 'lstm':
            # Prepare sequence data for LSTM
            sequence_data = self.prepare_lstm_data(features)
            pred_proba, uncertainty = model.predict(sequence_data)
            
            return {
                'prediction': np.argmax(pred_proba[0]) - 1,
                'probabilities': pred_proba[0],
                'confidence': np.max(pred_proba[0]),
                'uncertainty': uncertainty[0]
            }
            
        elif model_name == 'transformer':
            # Prepare data for transformer
            sequence_data = self.prepare_transformer_data(features)
            pred = model(sequence_data)
            pred_proba = torch.softmax(pred, dim=-1).cpu().numpy()
            
            return {
                'prediction': np.argmax(pred_proba[0]) - 1,
                'probabilities': pred_proba[0],
                'confidence': np.max(pred_proba[0])
            }
            
        elif model_name == 'ensemble':
            # Ensemble model prediction
            pred_result = model.predict_with_explanation(features)
            
            return {
                'prediction': np.argmax(pred_result['ensemble_prediction'][0]) - 1,
                'probabilities': pred_result['ensemble_prediction'][0],
                'confidence': pred_result['confidence'][0],
                'individual_predictions': pred_result['individual_predictions'],
                'agreement_score': pred_result['agreement_scores'][0]
            }
            
        else:
            # Standard sklearn-like model
            pred_proba = model.predict_proba(features)
            
            return {
                'prediction': np.argmax(pred_proba[0]) - 1,
                'probabilities': pred_proba[0],
                'confidence': np.max(pred_proba[0])
            }
    
    def prepare_lstm_data(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare data for LSTM model"""
        # This is simplified - would need actual sequence preparation
        sequence_length = 50
        n_features = len(features.columns)
        
        # Create dummy sequence for now
        sequence = np.random.randn(1, sequence_length, n_features)
        sequence[0, -1, :] = features.values[0]
        
        return sequence
    
    def prepare_transformer_data(self, features: pd.DataFrame) -> torch.Tensor:
        """Prepare data for transformer model"""
        import torch
        
        # Similar to LSTM preparation
        sequence_length = 50
        n_features = len(features.columns)
        
        sequence = torch.randn(1, sequence_length, n_features)
        sequence[0, -1, :] = torch.tensor(features.values[0], dtype=torch.float32)
        
        return sequence
    
    def ensemble_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple models"""
        # Extract predictions and confidences
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        
        for model_name, pred in predictions.items():
            all_predictions.append(pred['prediction'])
            all_confidences.append(pred['confidence'])
            all_probabilities.append(pred['probabilities'])
        
        # Weighted average based on confidence
        weights = np.array(all_confidences)
        weights = weights / weights.sum()
        
        # Calculate ensemble probabilities
        ensemble_probs = np.zeros_like(all_probabilities[0])
        for i, prob in enumerate(all_probabilities):
            ensemble_probs += prob * weights[i]
        
        # Majority vote for discrete prediction
        from collections import Counter
        vote_counts = Counter(all_predictions)
        majority_prediction = vote_counts.most_common(1)[0][0]
        
        # Agreement score
        agreement = vote_counts[majority_prediction] / len(all_predictions)
        
        return {
            'prediction': majority_prediction,
            'probabilities': ensemble_probs,
            'confidence': np.max(ensemble_probs),
            'agreement_score': agreement,
            'weighted_prediction': np.argmax(ensemble_probs) - 1,
            'model_weights': dict(zip(predictions.keys(), weights))
        }
    
    def store_prediction(self, predictions: Dict, timestamp):
        """Store prediction for later analysis"""
        self.prediction_history.append({
            'timestamp': timestamp,
            'predictions': predictions,
            'datetime': datetime.now()
        })
        
        # Keep only recent predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_prediction_accuracy(self, lookback_periods: int = 100) -> Dict:
        """Calculate prediction accuracy from history"""
        if len(self.prediction_history) < lookback_periods:
            return {}
        
        recent_predictions = self.prediction_history[-lookback_periods:]
        
        # This would need actual outcome data to calculate real accuracy
        # For now, returning mock accuracy
        
        accuracy_by_model = {}
        for model_name in self.models.keys():
            accuracy_by_model[model_name] = {
                'accuracy': 0.65,  # Placeholder
                'precision': 0.62,
                'recall': 0.68,
                'f1_score': 0.65
            }
        
        return accuracy_by_model