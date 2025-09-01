import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from typing import Dict, Any, Tuple

class EntryConfidenceModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model_name = None
        self.calibrated_models = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, 
            random_state=self.config.random_seed, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['standard'] = scaler
        
        results = {}
        
        xgb_model, xgb_metrics = self._train_xgboost(X_train, y_train, X_val, y_val)
        self.models['xgboost'] = xgb_model
        results['xgboost'] = xgb_metrics
        
        lgb_model, lgb_metrics = self._train_lightgbm(X_train, y_train, X_val, y_val)
        self.models['lightgbm'] = lgb_model
        results['lightgbm'] = lgb_metrics
        
        nn_model, nn_metrics = self._train_neural_network(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        self.models['neural_network'] = nn_model
        results['neural_network'] = nn_metrics
        
        ensemble_model, ensemble_metrics = self._create_ensemble(
            X_train, y_train, X_val, y_val
        )
        self.models['ensemble'] = ensemble_model
        results['ensemble'] = ensemble_metrics
        
        self._calibrate_models(X_train, y_train)
        
        self.best_model_name = max(results, key=lambda k: results[k]['val_roc_auc'])
        
        return results
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0)
            }
            
            model = xgb.XGBClassifier(**params, random_state=self.config.random_seed,
                                     use_label_encoder=False, eval_metric='logloss',
                                     callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            return roc_auc
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        best_params = study.best_params
        model = xgb.XGBClassifier(**best_params, random_state=self.config.random_seed,
                                 use_label_encoder=False, eval_metric='logloss',
                                 callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0)
            }
            
            model = lgb.LGBMClassifier(**params, random_state=self.config.random_seed,
                                      verbosity=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            return roc_auc
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        best_params = study.best_params
        model = lgb.LGBMClassifier(**best_params, random_state=self.config.random_seed,
                                  verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        def create_model(trial):
            n_layers = trial.suggest_int('n_layers', 2, 5)
            
            model = keras.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))
            
            for i in range(n_layers):
                n_units = trial.suggest_int(f'n_units_{i}', 32, 256)
                dropout_rate = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
                
                model.add(layers.Dense(n_units, activation='relu'))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC()]
            )
            
            return model
        
        def objective(trial):
            model = create_model(trial)
            
            class_weight = {0: 1.0, 1: trial.suggest_float('class_weight', 1.0, 5.0)}
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[early_stopping],
                verbose=0
            )
            
            y_pred_proba = model.predict(X_val).flatten()
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            return roc_auc
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        model = create_model(study.best_trial)
        class_weight = {0: 1.0, 1: study.best_trial.params.get('class_weight', 2.0)}
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred_proba = model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        return model, metrics
    
    def _create_ensemble(self, X_train, y_train, X_val, y_val) -> Tuple[Dict, Dict]:
        predictions_proba = {}
        for name, model in self.models.items():
            if name != 'ensemble':
                if name == 'neural_network':
                    X_val_scaled = self.scalers['standard'].transform(X_val)
                    predictions_proba[name] = model.predict(X_val_scaled).flatten()
                else:
                    predictions_proba[name] = model.predict_proba(X_val)[:, 1]
        
        pred_df = pd.DataFrame(predictions_proba)
        
        from sklearn.linear_model import LogisticRegression
        meta_model = LogisticRegression(random_state=self.config.random_seed)
        meta_model.fit(pred_df, y_val)
        
        ensemble_pred_proba = meta_model.predict_proba(pred_df)[:, 1]
        ensemble_pred = meta_model.predict(pred_df)
        
        metrics = self._calculate_metrics(y_val, ensemble_pred, ensemble_pred_proba)
        
        ensemble_model = {
            'base_models': self.models.copy(),
            'meta_model': meta_model,
            'scalers': self.scalers.copy()
        }
        
        return ensemble_model, metrics
    
    def _calibrate_models(self, X_train, y_train):
        for name, model in self.models.items():
            if name != 'ensemble':
                if name == 'neural_network':
                    continue
                calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
                calibrated.fit(X_train, y_train)
                self.calibrated_models[name] = calibrated
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        return {
            'val_accuracy': accuracy_score(y_true, y_pred),
            'val_precision': precision_score(y_true, y_pred),
            'val_recall': recall_score(y_true, y_pred),
            'val_f1': f1_score(y_true, y_pred),
            'val_roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = None, 
                     use_calibrated: bool = True) -> np.ndarray:
        if model_name is None:
            model_name = self.best_model_name
        
        if use_calibrated and model_name in self.calibrated_models:
            return self.calibrated_models[model_name].predict_proba(X)[:, 1]
        
        if model_name == 'ensemble':
            predictions_proba = {}
            for name, model in self.models['ensemble']['base_models'].items():
                if name != 'ensemble':
                    if name == 'neural_network':
                        X_scaled = self.scalers['standard'].transform(X)
                        predictions_proba[name] = model.predict(X_scaled).flatten()
                    else:
                        predictions_proba[name] = model.predict_proba(X)[:, 1]
            
            pred_df = pd.DataFrame(predictions_proba)
            return self.models['ensemble']['meta_model'].predict_proba(pred_df)[:, 1]
        
        elif model_name == 'neural_network':
            X_scaled = self.scalers['standard'].transform(X)
            return self.models[model_name].predict(X_scaled).flatten()
        
        else:
            return self.models[model_name].predict_proba(X)[:, 1]
    
    def filter_trades(self, trade_signals: pd.DataFrame, 
                     confidence_threshold: float = 0.6) -> pd.DataFrame:
        confidence_scores = self.predict_proba(trade_signals)
        filtered_signals = trade_signals[confidence_scores >= confidence_threshold].copy()
        filtered_signals['confidence_score'] = confidence_scores[confidence_scores >= confidence_threshold]
        
        return filtered_signals