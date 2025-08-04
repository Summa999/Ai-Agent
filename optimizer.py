import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import optuna
from sklearn.model_selection import TimeSeriesSplit
import logging

class ModelOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        
    def optimize_model_weights(self, models: Dict, validation_data: pd.DataFrame,
                             validation_labels: np.ndarray) -> Dict:
        """Optimize ensemble model weights"""
        
        def objective(trial):
            # Generate weights for each model
            weights = {}
            remaining_weight = 1.0
            
            for i, model_name in enumerate(list(models.keys())[:-1]):
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, remaining_weight)
                weights[model_name] = weight
                remaining_weight -= weight
            
            # Last model gets remaining weight
            weights[list(models.keys())[-1]] = remaining_weight
            
            # Calculate ensemble performance
            ensemble_pred = self.weighted_ensemble_predict(
                models, validation_data, weights
            )
            
            # Calculate sharpe ratio as objective
            returns = self.calculate_strategy_returns(
                ensemble_pred, validation_labels, validation_data
            )
            
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            
            return sharpe_ratio
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Get best weights
        best_weights = {}
        remaining_weight = 1.0
        
        for i, model_name in enumerate(list(models.keys())[:-1]):
            weight = study.best_params[f'weight_{model_name}']
            best_weights[model_name] = weight
            remaining_weight -= weight
        
        best_weights[list(models.keys())[-1]] = remaining_weight
        
        self.logger.info(f"Optimized model weights: {best_weights}")
        
        return best_weights
    
    def weighted_ensemble_predict(self, models: Dict, features: pd.DataFrame,
                                weights: Dict) -> np.ndarray:
        """Make weighted ensemble prediction"""
        predictions = []
        
        for model_name, model in models.items():
            pred = model.predict_proba(features)
            predictions.append(pred * weights.get(model_name, 0))
        
        ensemble_pred = np.sum(predictions, axis=0)
        return np.argmax(ensemble_pred, axis=1)
    
    def calculate_strategy_returns(self, predictions: np.ndarray,
                                 labels: np.ndarray,
                                 data: pd.DataFrame) -> np.ndarray:
        """Calculate returns from trading strategy"""
        # Convert predictions to trading signals
        signals = predictions - 1  # Convert 0,1,2 to -1,0,1
        
        # Get price returns
        returns = data['returns'].values
        
        # Calculate strategy returns
        strategy_returns = signals[:-1] * returns[1:]
        
        return strategy_returns
    
    def calculate_sharpe_ratio(self, returns: np.ndarray,
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def optimize_hyperparameters(self, model_class, X_train, y_train,
                               param_space: Dict) -> Dict:
        """Optimize model hyperparameters using Optuna"""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Create model with sampled parameters
            model = model_class(**params)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_tr, y_tr)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def adaptive_learning_rate(self, epoch: int, initial_lr: float,
                             decay_rate: float = 0.95) -> float:
        """Calculate adaptive learning rate"""
        return initial_lr * (decay_rate ** epoch)