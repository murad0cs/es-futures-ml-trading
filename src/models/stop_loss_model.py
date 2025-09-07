import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from typing import Dict, Any, Tuple

class DynamicStopLossModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model_name = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, 
            random_state=self.config.random_seed
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
        
        self.best_model_name = min(results, key=lambda k: results[k]['val_mse'])
        
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
            }
            
            model = xgb.XGBRegressor(**params, random_state=self.config.random_seed, 
                                    early_stopping_rounds=50)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        best_params = study.best_params
        model = xgb.XGBRegressor(**best_params, random_state=self.config.random_seed,
                                early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        metrics = {
            'val_mse': mean_squared_error(y_val, y_pred),
            'val_mae': mean_absolute_error(y_val, y_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred))
        }
        
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
            }
            
            model = lgb.LGBMRegressor(**params, random_state=self.config.random_seed, verbosity=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        
        best_params = study.best_params
        model = lgb.LGBMRegressor(**best_params, random_state=self.config.random_seed, verbosity=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        y_pred = model.predict(X_val)
        metrics = {
            'val_mse': mean_squared_error(y_val, y_pred),
            'val_mae': mean_absolute_error(y_val, y_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred))
        }
        
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
            
            model.add(layers.Dense(1))
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        def objective(trial):
            model = create_model(trial)
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5, show_progress_bar=False)
        
        model = create_model(study.best_trial)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = model.predict(X_val).flatten()
        metrics = {
            'val_mse': mean_squared_error(y_val, y_pred),
            'val_mae': mean_absolute_error(y_val, y_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred))
        }
        
        return model, metrics
    
    def _create_ensemble(self, X_train, y_train, X_val, y_val) -> Tuple[Dict, Dict]:
        predictions = {}
        for name, model in self.models.items():
            if name != 'ensemble':
                if name == 'neural_network':
                    X_val_scaled = self.scalers['standard'].transform(X_val)
                    predictions[name] = model.predict(X_val_scaled).flatten()
                else:
                    predictions[name] = model.predict(X_val)
        
        pred_df = pd.DataFrame(predictions)
        
        from sklearn.linear_model import LinearRegression
        meta_model = LinearRegression()
        meta_model.fit(pred_df, y_val)
        
        ensemble_pred = meta_model.predict(pred_df)
        
        metrics = {
            'val_mse': mean_squared_error(y_val, ensemble_pred),
            'val_mae': mean_absolute_error(y_val, ensemble_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred))
        }
        
        ensemble_model = {
            'base_models': self.models.copy(),
            'meta_model': meta_model,
            'scalers': self.scalers.copy()
        }
        
        return ensemble_model, metrics
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name == 'ensemble':
            predictions = {}
            for name, model in self.models['ensemble']['base_models'].items():
                if name != 'ensemble':
                    if name == 'neural_network':
                        X_scaled = self.scalers['standard'].transform(X)
                        predictions[name] = model.predict(X_scaled).flatten()
                    else:
                        predictions[name] = model.predict(X)
            
            pred_df = pd.DataFrame(predictions)
            return self.models['ensemble']['meta_model'].predict(pred_df)
        
        elif model_name == 'neural_network':
            X_scaled = self.scalers['standard'].transform(X)
            return self.models[model_name].predict(X_scaled).flatten()
        
        else:
            return self.models[model_name].predict(X)
    
    def evaluate_stop_loss_performance(self, df: pd.DataFrame, 
                                      static_stops: pd.Series,
                                      dynamic_stops: pd.Series) -> Dict[str, Any]:
        results = {
            'static_performance': self._calculate_stop_metrics(df, static_stops),
            'dynamic_performance': self._calculate_stop_metrics(df, dynamic_stops),
            'improvement': {}
        }
        
        for metric in results['static_performance']:
            static_val = results['static_performance'][metric]
            dynamic_val = results['dynamic_performance'][metric]
            if static_val != 0:
                improvement = ((dynamic_val - static_val) / abs(static_val)) * 100
            else:
                improvement = 0
            results['improvement'][metric] = improvement
        
        return results
    
    def _calculate_stop_metrics(self, df: pd.DataFrame, stops: pd.Series) -> Dict[str, float]:
        df = df.copy()
        df['stop_loss'] = stops
        
        df['stopped_out'] = False
        df['stop_price'] = np.nan
        df['trade_return'] = np.nan
        
        for i in range(len(df) - self.config.prediction_horizon):
            stop_level = df.iloc[i]['stop_loss']
            future_prices = df.iloc[i+1:i+self.config.prediction_horizon+1]
            
            stop_hit = future_prices['low'] <= stop_level
            if stop_hit.any():
                stop_idx = stop_hit.idxmax()
                df.loc[df.index[i], 'stopped_out'] = True
                df.loc[df.index[i], 'stop_price'] = stop_level
                df.loc[df.index[i], 'trade_return'] = (stop_level - df.iloc[i]['close']) / df.iloc[i]['close']
            else:
                final_price = future_prices.iloc[-1]['close']
                df.loc[df.index[i], 'trade_return'] = (final_price - df.iloc[i]['close']) / df.iloc[i]['close']
        
        winning_trades = df[df['trade_return'] > 0]
        losing_trades = df[df['trade_return'] <= 0]
        
        metrics = {
            'avg_win': winning_trades['trade_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': abs(losing_trades['trade_return'].mean()) if len(losing_trades) > 0 else 0,
            'win_loss_ratio': 0,
            'win_rate': len(winning_trades) / len(df[df['trade_return'].notna()]) if len(df[df['trade_return'].notna()]) > 0 else 0,
            'avg_trade_duration': self.config.prediction_horizon,
            'stop_hit_rate': df['stopped_out'].sum() / len(df[df['trade_return'].notna()]) if len(df[df['trade_return'].notna()]) > 0 else 0
        }
        
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = metrics['avg_win'] / metrics['avg_loss']
        
        return metrics