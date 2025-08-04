import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from typing import Dict, List, Tuple
import mlflow
import mlflow.sklearn
import mlflow.pytorch

class AdvancedModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_params = {}
        self.performance_metrics = {}
        
    def create_train_val_test_split(self, data, train_ratio=0.7, val_ratio=0.15):
        """Create time-based train/validation/test split"""
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def create_labels(self, data, method='returns', threshold=0.001):
        """Create trading labels"""
        if method == 'returns':
            # Future returns
            data['future_returns'] = data['close'].shift(-1) / data['close'] - 1
            
            # Three classes: Buy (1), Hold (0), Sell (-1)
            conditions = [
                data['future_returns'] > threshold,
                data['future_returns'] < -threshold
            ]
            choices = [2, 0]  # Buy=2, Sell=0
            data['label'] = np.select(conditions, choices, default=1)  # Hold=1
            
        elif method == 'triple_barrier':
            # Triple barrier labeling
            data['label'] = self._triple_barrier_labeling(
                data, 
                upper_barrier=threshold,
                lower_barrier=-threshold,
                max_holding_period=10
            )
            
        elif method == 'trend':
            # Trend-based labeling
            data['sma_fast'] = data['close'].rolling(10).mean()
            data['sma_slow'] = data['close'].rolling(30).mean()
            
            conditions = [
                (data['sma_fast'] > data['sma_slow']) & (data['sma_fast'].shift(1) <= data['sma_slow'].shift(1)),
                (data['sma_fast'] < data['sma_slow']) & (data['sma_fast'].shift(1) >= data['sma_slow'].shift(1))
            ]
            choices = [2, 0]
            data['label'] = np.select(conditions, choices, default=1)
        
        return data
    
    def _triple_barrier_labeling(self, data, upper_barrier, lower_barrier, max_holding_period):
        """Implement triple barrier labeling method"""
        labels = []
        
        for i in range(len(data) - max_holding_period):
            future_returns = (data['close'].iloc[i+1:i+max_holding_period+1].values / 
                            data['close'].iloc[i] - 1)
            
            # Check upper barrier
            upper_crossed = np.where(future_returns > upper_barrier)[0]
            # Check lower barrier
            lower_crossed = np.where(future_returns < lower_barrier)[0]
            
            if len(upper_crossed) > 0 and len(lower_crossed) > 0:
                # Both barriers crossed, take the first one
                if upper_crossed[0] < lower_crossed[0]:
                    labels.append(2)  # Buy signal
                else:
                    labels.append(0)  # Sell signal
            elif len(upper_crossed) > 0:
                labels.append(2)  # Buy signal
            elif len(lower_crossed) > 0:
                labels.append(0)  # Sell signal
            else:
                # No barrier crossed, check final return
                final_return = future_returns[-1]
                if final_return > 0:
                    labels.append(2)
                elif final_return < 0:
                    labels.append(0)
                else:
                    labels.append(1)  # Hold
        
        # Pad the remaining values
        labels.extend([1] * max_holding_period)
        
        return labels
    
    def optimize_hyperparameters(self, model_type, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                }
                
                import xgboost as xgb
                model = xgb.XGBClassifier(**params, objective='multi:softprob', 
                                        n_jobs=-1, random_state=42)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                }
                
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params, n_jobs=-1, random_state=42)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        self.best_params[model_type] = study.best_params
        
        return study.best_params
    
    def train_with_cross_validation(self, model, X, y, cv_folds=5):
        """Train with time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            score = model.score(X_val_fold, y_val_fold)
            scores.append(score)
            
            print(f"Fold {fold + 1} Score: {score:.4f}")
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"Average CV Score: {avg_score:.4f} (+/- {std_score:.4f})")
        
        return scores
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Classification metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Trading-specific metrics
        trading_metrics = self.calculate_trading_metrics(y_test, y_pred, X_test)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'trading_metrics': trading_metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def calculate_trading_metrics(self, y_true, y_pred, X_test):
        """Calculate trading-specific performance metrics"""
        # Convert to trading signals (-1, 0, 1)
        signal_map = {0: -1, 1: 0, 2: 1}
        true_signals = pd.Series(y_true).map(signal_map)
        pred_signals = pd.Series(y_pred).map(signal_map)
        
        # Calculate returns (simplified)
        returns = X_test['returns'].values
        
        # Strategy returns
        strategy_returns = pred_signals.values[:-1] * returns[1:]
        
        # Performance metrics
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(np.cumsum(strategy_returns))
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
        
        # Profit factor
        gross_profits = np.sum(strategy_returns[strategy_returns > 0])
        gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = gross_profits / (gross_losses + 1e-10)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': np.sum(pred_signals != 0)
        }
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        return np.min(drawdown)
    
    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train and evaluate all models"""
        mlflow.set_experiment("ai_trading_models")
        
        models_to_train = ['xgboost', 'lightgbm', 'neural_net', 'ensemble']
        
        for model_type in models_to_train:
            with mlflow.start_run(run_name=f"{model_type}_training"):
                print(f"\nTraining {model_type}...")
                
                # Hyperparameter optimization
                if model_type in ['xgboost', 'lightgbm']:
                    best_params = self.optimize_hyperparameters(
                        model_type, X_train, y_train, X_val, y_val
                    )
                    mlflow.log_params(best_params)
                
                # Create and train model
                if model_type == 'xgboost':
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**self.best_params.get(model_type, {}),
                                            objective='multi:softprob',
                                            n_jobs=-1, random_state=42)
                elif model_type == 'lightgbm':
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**self.best_params.get(model_type, {}),
                                             n_jobs=-1, random_state=42)
                elif model_type == 'neural_net':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(
                        hidden_layer_sizes=(256, 128, 64),
                        activation='relu',
                        solver='adam',
                        alpha=0.001,
                        batch_size=32,
                        learning_rate='adaptive',
                        max_iter=500,
                        random_state=42
                    )
                elif model_type == 'ensemble':
                    from ai.models.ensemble_model import EnsembleTradingModel
                    ensemble = EnsembleTradingModel(self.config)
                    ensemble.create_base_models()
                    ensemble.create_stacking_ensemble()
                    model = ensemble.ensemble
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                eval_results = self.evaluate_model(model, X_test, y_test)
                
                # Log metrics
                mlflow.log_metrics({
                    'accuracy': eval_results['classification_report']['accuracy'],
                    'total_return': eval_results['trading_metrics']['total_return'],
                    'sharpe_ratio': eval_results['trading_metrics']['sharpe_ratio'],
                    'max_drawdown': eval_results['trading_metrics']['max_drawdown'],
                    'win_rate': eval_results['trading_metrics']['win_rate'],
                    'profit_factor': eval_results['trading_metrics']['profit_factor']
                })
                
                # Log model
                mlflow.sklearn.log_model(model, model_type)
                
                # Store results
                self.models[model_type] = model
                self.performance_metrics[model_type] = eval_results
                
                print(f"{model_type} - Accuracy: {eval_results['classification_report']['accuracy']:.4f}")
                print(f"{model_type} - Sharpe Ratio: {eval_results['trading_metrics']['sharpe_ratio']:.4f}")
        
        return self.models, self.performance_metrics