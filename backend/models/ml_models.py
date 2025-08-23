"""
Machine Learning Models Module
Implements ensemble methods for alpha factor prediction using XGBoost and Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class EnsembleAlphaModel:
    """
    Ensemble model combining XGBoost and Random Forest for alpha factor prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Initialize models with more conservative parameters
        self.xgboost = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        
        self.random_forest = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Ensemble weights (XGBoost gets higher weight)
        self.weights = {'xgboost': 0.6, 'random_forest': 0.4}
        
        # Model artifacts
        self.feature_importance = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, alpha_factors_df):
        """
        Prepare features and target from alpha factors DataFrame
        """
        # Remove non-feature columns
        exclude_cols = ['date', 'price', 'future_return_5d', 'symbol', 'index']
        feature_cols = [col for col in alpha_factors_df.columns 
                       if col not in exclude_cols]
        
        # Extract features and target
        X = alpha_factors_df[feature_cols].copy()
        y = alpha_factors_df['future_return_5d'].copy()
        
        # Remove samples without target variable
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0)
        
        # Remove constant features (no variation)
        constant_features = X.columns[X.var() == 0].tolist()
        if constant_features:
            print(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def fit(self, alpha_factors_df):
        """
        Train the ensemble model on alpha factors
        
        Parameters:
        alpha_factors_df (pd.DataFrame): DataFrame with alpha factors and target
        
        Returns:
        dict: Training results including predictions and metrics
        """
        print(f'Training ensemble model with {len(alpha_factors_df)} samples')
        
        # Prepare data
        X, y = self.prepare_features(alpha_factors_df)
        
        if len(X) < 100:
            print('Warning: Insufficient training data')
            return None
            
        if len(self.feature_names) == 0:
            print('Error: No valid features found')
            return None
            
        print(f'Training samples after filtering: {len(X)}')
        print(f'Features: {len(self.feature_names)}')
        
        # Scale features
        try:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        except Exception as e:
            print(f'Error scaling features: {e}')
            return None
        
        # Train both models
        try:
            print('Training XGBoost model...')
            self.xgboost.fit(X_scaled, y)
            
            print('Training Random Forest model...')
            self.random_forest.fit(X_scaled, y)
        except Exception as e:
            print(f'Error training models: {e}')
            return None
        
        # Calculate feature importance
        self._calculate_feature_importance(X_scaled)
        
        self.is_trained = True
        print('Ensemble training complete')
        
        # Generate predictions for training data
        predictions = self.predict(X_scaled)
        
        # Create results DataFrame
        results_df = alpha_factors_df[~alpha_factors_df['future_return_5d'].isna()].copy()
        results_df = results_df.iloc[:len(predictions)].copy()  # Ensure same length
        results_df['predicted_return'] = predictions
        results_df['actual_return'] = y.values
        
        # Evaluate model
        evaluation_metrics = self.evaluate_model(results_df)
        
        return {
            'predictions': results_df,
            'feature_importance': self.feature_importance,
            'evaluation_metrics': evaluation_metrics,
            'model_summary': self.get_model_summary()
        }
    
    def predict(self, X):
        """
        Generate ensemble predictions
        
        Parameters:
        X (pd.DataFrame): Features to predict on
        
        Returns:
        np.array: Ensemble predictions
        """
        if not self.is_trained:
            print('Warning: Model not trained yet')
            return np.zeros(len(X))
        
        try:
            # Scale features if needed
            if isinstance(X, pd.DataFrame):
                # Ensure we have the right features
                missing_features = set(self.feature_names) - set(X.columns)
                if missing_features:
                    print(f'Warning: Missing features: {missing_features}')
                    # Add missing features with zeros
                    for feature in missing_features:
                        X[feature] = 0
                
                # Select only training features in the same order
                X_selected = X[self.feature_names]
                X_scaled = self.scaler.transform(X_selected)
                X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X_selected.index)
            else:
                X_scaled = X
            
            # Get predictions from both models
            xgb_pred = self.xgboost.predict(X_scaled)
            rf_pred = self.random_forest.predict(X_scaled)
            
            # Ensemble prediction
            ensemble_pred = (self.weights['xgboost'] * xgb_pred + 
                            self.weights['random_forest'] * rf_pred)
            
            return ensemble_pred
            
        except Exception as e:
            print(f'Error generating predictions: {e}')
            return np.zeros(len(X))
    
    def _calculate_feature_importance(self, X):
        """
        Calculate combined feature importance from both models
        """
        try:
            # Get feature importance from both models
            xgb_importance = self.xgboost.feature_importances_
            rf_importance = self.random_forest.feature_importances_
            
            # Combine with ensemble weights
            combined_importance = (self.weights['xgboost'] * xgb_importance + 
                                 self.weights['random_forest'] * rf_importance)
            
            # Create importance dictionary
            self.feature_importance = dict(zip(self.feature_names, combined_importance))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        except Exception as e:
            print(f'Error calculating feature importance: {e}')
            self.feature_importance = {}
    
    def evaluate_model(self, results_df):
        """
        Evaluate model performance with comprehensive metrics
        
        Parameters:
        results_df (pd.DataFrame): DataFrame with predicted_return and actual_return
        
        Returns:
        dict: Evaluation metrics
        """
        try:
            valid_results = results_df[
                (~results_df['predicted_return'].isna()) & 
                (~results_df['actual_return'].isna()) &
                (np.isfinite(results_df['predicted_return'])) &
                (np.isfinite(results_df['actual_return']))
            ].copy()
            
            if len(valid_results) == 0:
                print('Warning: No valid results for evaluation')
                return {
                    'r_squared': 0.0,
                    'rmse': 0.0,
                    'mae': 0.0,
                    'correlation': 0.0,
                    'direction_accuracy': 0.5,
                    'hit_rate': 0.5,
                    'information_coefficient': 0.0,
                    'sample_size': 0
                }
            
            y_true = valid_results['actual_return'].values
            y_pred = valid_results['predicted_return'].values
            
            # Basic regression metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Correlation
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
            if np.isnan(correlation):
                correlation = 0.0
            
            # Direction accuracy (sign prediction)
            correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
            direction_accuracy = correct_direction / len(y_true)
            
            # Hit rate for significant predictions
            threshold = 0.005  # 0.5% threshold
            significant_pred = np.abs(y_pred) > threshold
            if np.sum(significant_pred) > 0:
                hit_rate = np.sum((y_true[significant_pred] * y_pred[significant_pred]) > 0) / np.sum(significant_pred)
            else:
                hit_rate = 0.5
            
            # Information Coefficient (IC)
            ic = correlation
            
            return {
                'r_squared': r2,
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'direction_accuracy': direction_accuracy,
                'hit_rate': hit_rate,
                'information_coefficient': ic,
                'sample_size': len(valid_results)
            }
            
        except Exception as e:
            print(f'Error evaluating model: {e}')
            return {
                'r_squared': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'correlation': 0.0,
                'direction_accuracy': 0.5,
                'hit_rate': 0.5,
                'information_coefficient': 0.0,
                'sample_size': 0
            }
    
    def cross_validate(self, alpha_factors_df, cv_folds=5):
        """
        Perform time-series cross-validation
        
        Parameters:
        alpha_factors_df (pd.DataFrame): Alpha factors DataFrame
        cv_folds (int): Number of cross-validation folds
        
        Returns:
        dict: Cross-validation results
        """
        try:
            X, y = self.prepare_features(alpha_factors_df)
            
            if len(X) < cv_folds * 50:  # Need enough data for each fold
                print('Insufficient data for cross-validation')
                return None
            
            # Use TimeSeriesSplit for temporal data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = []
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f'Processing fold {fold + 1}/{cv_folds}...')
                
                # Split data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                fold_scaler = StandardScaler()
                X_train_scaled = fold_scaler.fit_transform(X_train)
                X_val_scaled = fold_scaler.transform(X_val)
                
                # Train models
                fold_xgb = xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=self.random_state, n_jobs=-1,
                    objective='reg:squarederror'
                )
                fold_rf = RandomForestRegressor(
                    n_estimators=100, max_depth=10,
                    random_state=self.random_state, n_jobs=-1
                )
                
                fold_xgb.fit(X_train_scaled, y_train)
                fold_rf.fit(X_train_scaled, y_train)
                
                # Predict
                xgb_pred = fold_xgb.predict(X_val_scaled)
                rf_pred = fold_rf.predict(X_val_scaled)
                ensemble_pred = (self.weights['xgboost'] * xgb_pred + 
                               self.weights['random_forest'] * rf_pred)
                
                # Evaluate fold
                fold_r2 = r2_score(y_val, ensemble_pred)
                fold_ic = np.corrcoef(y_val, ensemble_pred)[0, 1]
                if np.isnan(fold_ic):
                    fold_ic = 0.0
                direction_acc = np.sum(np.sign(y_val) == np.sign(ensemble_pred)) / len(y_val)
                
                cv_scores.append(fold_ic)  # Use IC as primary metric
                fold_metrics.append({
                    'fold': fold + 1,
                    'r2_score': fold_r2,
                    'information_coefficient': fold_ic,
                    'direction_accuracy': direction_acc,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                })
            
            return {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'fold_scores': cv_scores,
                'fold_metrics': fold_metrics,
                'cv_method': 'TimeSeriesSplit'
            }
            
        except Exception as e:
            print(f'Error in cross-validation: {e}')
            return None
    
    def get_model_summary(self):
        """
        Get comprehensive model summary
        """
        return {
            'model_type': 'Ensemble (XGBoost + Random Forest)',
            'xgboost_params': {
                'n_estimators': self.xgboost.n_estimators,
                'learning_rate': self.xgboost.learning_rate,
                'max_depth': self.xgboost.max_depth
            },
            'random_forest_params': {
                'n_estimators': self.random_forest.n_estimators,
                'max_depth': self.random_forest.max_depth
            },
            'ensemble_weights': self.weights,
            'total_features': len(self.feature_names),
            'top_features': self.get_top_features(10),
            'is_trained': self.is_trained
        }
    
    def get_top_features(self, n=10):
        """
        Get top N most important features
        """
        if not self.feature_importance:
            return []
        
        top_features = list(self.feature_importance.items())[:n]
        return [
            {
                'feature': feature.replace('_', ' ').title(),
                'importance': f"{importance:.3f}",
                'importance_pct': f"{importance * 100:.1f}%"
            }
            for feature, importance in top_features
        ]
    
    def feature_selection(self, X, y, top_k=20):
        """
        Select top K features based on importance
        """
        if not self.is_trained:
            print('Model must be trained before feature selection')
            return X
        
        available_features = [f for f in list(self.feature_importance.keys())[:top_k] if f in X.columns]
        return X[available_features]

# Utility functions for model training and evaluation
def split_train_test(df, test_size=0.2, time_based=True):
    """
    Split data into train and test sets
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    test_size (float): Proportion of test data
    time_based (bool): Use time-based split vs random split
    
    Returns:
    tuple: (train_df, test_df)
    """
    if time_based and 'date' in df.columns:
        # Time-based split (more realistic for financial data)
        df_sorted = df.sort_values('date').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
    else:
        # Random split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42
        )
    
    return train_df, test_df

def walk_forward_validation(model_class, df, window_size=1000, step_size=50):
    """
    Perform walk-forward validation for time series data
    
    Parameters:
    model_class: Model class to instantiate
    df (pd.DataFrame): Full dataset
    window_size (int): Size of training window
    step_size (int): Step size for moving window
    
    Returns:
    dict: Validation results
    """
    if len(df) < window_size + step_size:
        print('Insufficient data for walk-forward validation')
        return None
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    predictions = []
    actuals = []
    dates = []
    
    start_idx = window_size
    while start_idx + step_size <= len(df_sorted):
        # Define windows
        train_end = start_idx
        test_start = start_idx
        test_end = min(start_idx + step_size, len(df_sorted))
        
        # Split data
        train_data = df_sorted.iloc[:train_end]
        test_data = df_sorted.iloc[test_start:test_end]
        
        # Train model
        model = model_class()
        train_result = model.fit(train_data)
        
        if train_result is None:
            start_idx += step_size
            continue
        
        # Predict on test data
        try:
            test_features, test_targets = model.prepare_features(test_data)
            if len(test_features) == 0:
                start_idx += step_size
                continue
                
            test_pred = model.predict(test_features)
            
            # Store results
            predictions.extend(test_pred)
            actuals.extend(test_targets.values)
            dates.extend(test_data['date'].iloc[:len(test_pred)])
            
        except Exception as e:
            print(f'Error in walk-forward step: {e}')
        
        start_idx += step_size
    
    if len(predictions) == 0:
        return None
    
    # Calculate overall metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    direction_accuracy = np.sum(np.sign(actuals) == np.sign(predictions)) / len(actuals)
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates,
        'correlation': correlation,
        'rmse': rmse,
        'direction_accuracy': direction_accuracy,
        'total_predictions': len(predictions)
    }

def hyperparameter_tuning(X, y, param_grid=None, cv_folds=3):
    """
    Perform hyperparameter tuning using time series cross-validation
    
    Parameters:
    X (pd.DataFrame): Features
    y (pd.Series): Target
    param_grid (dict): Parameter grid to search
    cv_folds (int): Number of CV folds
    
    Returns:
    dict: Best parameters and scores
    """
    if param_grid is None:
        param_grid = {
            'xgb': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [4, 6, 8]
            },
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15]
            }
        }
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    best_score = -np.inf
    best_params = {}
    
    # Grid search for XGBoost
    print('Tuning XGBoost parameters...')
    for n_est in param_grid['xgb']['n_estimators']:
        for lr in param_grid['xgb']['learning_rate']:
            for depth in param_grid['xgb']['max_depth']:
                params = {
                    'n_estimators': n_est, 
                    'learning_rate': lr, 
                    'max_depth': depth,
                    'random_state': 42,
                    'n_jobs': -1,
                    'objective': 'reg:squarederror'
                }
                
                try:
                    model = xgb.XGBRegressor(**params)
                    scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    mean_score = np.mean(scores)
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params['xgb'] = params
                except Exception as e:
                    print(f'Error tuning XGBoost with params {params}: {e}')
                    continue
    
    # Grid search for Random Forest
    print('Tuning Random Forest parameters...')
    best_rf_score = -np.inf
    for n_est in param_grid['rf']['n_estimators']:
        for depth in param_grid['rf']['max_depth']:
            params = {
                'n_estimators': n_est, 
                'max_depth': depth,
                'random_state': 42,
                'n_jobs': -1
            }
            
            try:
                model = RandomForestRegressor(**params)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                mean_score = np.mean(scores)
                
                if mean_score > best_rf_score:
                    best_rf_score = mean_score
                    best_params['rf'] = params
            except Exception as e:
                print(f'Error tuning Random Forest with params {params}: {e}')
                continue
    
    return {
        'best_xgb_params': best_params.get('xgb', {}),
        'best_rf_params': best_params.get('rf', {}),
        'best_xgb_score': best_score,
        'best_rf_score': best_rf_score
    }