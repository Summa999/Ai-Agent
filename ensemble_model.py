import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

class EnsembleTradingModel:
    def __init__(self, config):
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.ensemble = None
        self.feature_importance = {}
        
    def create_base_models(self):
        """Create diverse base models"""
        
        # XGBoost
        self.base_models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=42
        )
        
        # LightGBM
        self.base_models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        
        # CatBoost
        self.base_models['catboost'] = CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
        
        # Neural Network
        self.base_models['neural_net'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Extra Trees
        from sklearn.ensemble import ExtraTreesClassifier
        self.base_models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        # Gradient Boosting with different parameters
        self.base_models['xgboost_2'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=43
        )
        
        return self.base_models
    
    def create_stacking_ensemble(self):
        """Create stacking ensemble with meta-learner"""
        # Meta-learner
        self.meta_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        
        # Create stacking ensemble
        self.ensemble = StackingClassifier(
            estimators=list(self.base_models.items()),
            final_estimator=self.meta_model,
            cv=5,  # 5-fold cross-validation for generating meta-features
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return self.ensemble
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble model"""
        print("Training ensemble model...")
        
        # Train individual models for feature importance
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Calculate feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Validate if validation data provided
        if X_val is not None and y_val is not None:
            val_score = self.ensemble.score(X_val, y_val)
            print(f"Validation accuracy: {val_score:.4f}")
            
            # Individual model scores
            for name, model in self.base_models.items():
                score = model.score(X_val, y_val)
                print(f"{name} validation accuracy: {score:.4f}")
        
        return self
    
    def predict_proba_with_confidence(self, X):
        """Predict with confidence intervals"""
        # Get predictions from all base models
        base_predictions = []
        for name, model in self.base_models.items():
            pred = model.predict_proba(X)
            base_predictions.append(pred)
        
        base_predictions = np.array(base_predictions)
        
        # Calculate mean and std across models
        mean_predictions = np.mean(base_predictions, axis=0)
        std_predictions = np.std(base_predictions, axis=0)
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(X)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'mean_prediction': mean_predictions,
            'std_prediction': std_predictions,
            'confidence': 1 - std_predictions.mean(axis=1)  # Higher std = lower confidence
        }
    
    def predict_with_explanation(self, X):
        """Predict with model explanations"""
        predictions = self.predict_proba_with_confidence(X)
        
        # Get individual model predictions
        individual_predictions = {}
        for name, model in self.base_models.items():
            individual_predictions[name] = model.predict_proba(X)
        
        # Calculate agreement between models
        all_preds = np.array(list(individual_predictions.values()))
        pred_classes = np.argmax(all_preds, axis=2)
        
        # Mode prediction (most common prediction)
        from scipy import stats
        mode_predictions = stats.mode(pred_classes, axis=0)[0].flatten()
        
        # Agreement score
        agreement_scores = []
        for i in range(len(X)):
            preds_i = pred_classes[:, i]
            agreement = np.sum(preds_i == mode_predictions[i]) / len(self.base_models)
            agreement_scores.append(agreement)
        
        return {
            'predictions': predictions,
            'individual_predictions': individual_predictions,
            'agreement_scores': np.array(agreement_scores),
            'mode_predictions': mode_predictions
        }
    
    def adaptive_retraining(self, new_X, new_y, retrain_threshold=0.1):
        """Adaptively retrain models based on performance"""
        # Evaluate current performance
        current_score = self.ensemble.score(new_X, new_y)
        
        # Check individual model performance
        model_scores = {}
        for name, model in self.base_models.items():
            model_scores[name] = model.score(new_X, new_y)
        
        # Identify underperforming models
        avg_score = np.mean(list(model_scores.values()))
        underperforming = [name for name, score in model_scores.items() 
                          if score < avg_score - retrain_threshold]
        
        # Retrain underperforming models
        if underperforming:
            print(f"Retraining underperforming models: {underperforming}")
            for name in underperforming:
                self.base_models[name].fit(new_X, new_y)
            
            # Retrain ensemble
            self.create_stacking_ensemble()
            self.ensemble.fit(new_X, new_y)
            
            new_score = self.ensemble.score(new_X, new_y)
            print(f"Performance improved from {current_score:.4f} to {new_score:.4f}")
        
        return self
    
    def save_models(self, path):
        """Save all models"""
        # Save base models
        for name, model in self.base_models.items():
            joblib.dump(model, f"{path}/base_model_{name}.pkl")
        
        # Save ensemble
        joblib.dump(self.ensemble, f"{path}/ensemble_model.pkl")
        
        # Save feature importance
        joblib.dump(self.feature_importance, f"{path}/feature_importance.pkl")
    
    def load_models(self, path):
        """Load all models"""
        import os
        
        # Load base models
        self.base_models = {}
        for filename in os.listdir(path):
            if filename.startswith('base_model_'):
                name = filename.replace('base_model_', '').replace('.pkl', '')
                self.base_models[name] = joblib.load(f"{path}/{filename}")
        
        # Load ensemble
        self.ensemble = joblib.load(f"{path}/ensemble_model.pkl")
        
        # Load feature importance
        self.feature_importance = joblib.load(f"{path}/feature_importance.pkl")