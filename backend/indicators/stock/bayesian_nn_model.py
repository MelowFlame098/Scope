"""Bayesian Neural Network Model for Stock Price Prediction

This module implements Bayesian Neural Networks for stock price prediction
with uncertainty quantification using Monte Carlo Dropout.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Deep Learning imports (with fallbacks)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified Bayesian implementation.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using alternative implementations.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class BayesianPrediction:
    """Bayesian prediction with uncertainty estimates."""
    mean: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_interval_lower: float
    confidence_interval_upper: float

@dataclass
class BayesianNNResults:
    """Results from Bayesian Neural Network."""
    predictions: List[BayesianPrediction]
    actual_values: np.ndarray
    mean_predictions: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    calibration_score: float
    model_architecture: Dict[str, Any]
    training_history: Optional[Dict[str, List[float]]] = None

class BayesianNeuralNetwork:
    """Bayesian Neural Network with Monte Carlo Dropout for uncertainty quantification."""
    
    def __init__(self, input_dim: int, architecture: Dict[str, Any] = None):
        self.input_dim = input_dim
        self.architecture = architecture or {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu',
            'output_activation': 'linear'
        }
        self.model = None
        self.scaler = StandardScaler()
        self.n_monte_carlo_samples = 100
        
    def build_model(self) -> None:
        """Build Bayesian Neural Network architecture."""
        if TORCH_AVAILABLE:
            self._build_pytorch_model()
        elif TF_AVAILABLE:
            self._build_tensorflow_model()
        else:
            print("Neither PyTorch nor TensorFlow available. Using simplified Bayesian implementation.")
            self._build_simplified_model()
            
    def _build_pytorch_model(self) -> None:
        """Build model using PyTorch."""
        class BayesianNet(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout_rate):
                super(BayesianNet, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    prev_dim = hidden_dim
                    
                # Output layer (mean)
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
                # Additional layer for aleatoric uncertainty (log variance)
                self.log_var_layer = nn.Linear(prev_dim, 1)
                
            def forward(self, x, return_uncertainty=False):
                # Forward pass through main network
                features = x
                for layer in self.network[:-1]:
                    features = layer(features)
                    
                # Mean prediction
                mean = self.network[-1](features)
                
                if return_uncertainty:
                    # Log variance for aleatoric uncertainty
                    log_var = self.log_var_layer(features)
                    return mean, log_var
                else:
                    return mean
                    
        self.model = BayesianNet(
            self.input_dim,
            self.architecture['hidden_layers'],
            self.architecture['dropout_rate']
        )
        
    def _build_tensorflow_model(self) -> None:
        """Build model using TensorFlow/Keras."""
        # Mean prediction model
        mean_model = Sequential()
        mean_model.add(Dense(self.architecture['hidden_layers'][0], 
                           activation=self.architecture['activation'],
                           input_shape=(self.input_dim,)))
        mean_model.add(Dropout(self.architecture['dropout_rate']))
        
        for units in self.architecture['hidden_layers'][1:]:
            mean_model.add(Dense(units, activation=self.architecture['activation']))
            mean_model.add(Dropout(self.architecture['dropout_rate']))
            
        mean_model.add(Dense(1, activation=self.architecture['output_activation']))
        
        # Variance prediction model (for aleatoric uncertainty)
        var_model = Sequential()
        var_model.add(Dense(self.architecture['hidden_layers'][0], 
                          activation=self.architecture['activation'],
                          input_shape=(self.input_dim,)))
        var_model.add(Dropout(self.architecture['dropout_rate']))
        
        for units in self.architecture['hidden_layers'][1:]:
            var_model.add(Dense(units, activation=self.architecture['activation']))
            var_model.add(Dropout(self.architecture['dropout_rate']))
            
        var_model.add(Dense(1, activation='softplus'))  # Ensure positive variance
        
        self.model = {'mean': mean_model, 'variance': var_model}
        
        # Compile models
        optimizer = Adam(learning_rate=0.001)
        self.model['mean'].compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        self.model['variance'].compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
    def _build_simplified_model(self) -> None:
        """Build simplified Bayesian model when deep learning frameworks are not available."""
        # Use Bayesian Ridge Regression as fallback
        self.model = {
            'bayesian_ridge': BayesianRidge(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> BayesianNNResults:
        """Train Bayesian Neural Network."""
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            return self._train_pytorch(X_train_scaled, y_train, X_val_scaled, y_val, epochs, batch_size)
        elif TF_AVAILABLE and isinstance(self.model, dict) and 'mean' in self.model:
            return self._train_tensorflow(X_train_scaled, y_train, X_val_scaled, y_val, epochs, batch_size)
        else:
            return self._train_simplified(X_train_scaled, y_train, X_val_scaled, y_val)
            
    def _train_pytorch(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int, batch_size: int) -> BayesianNNResults:
        """Train using PyTorch."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass with uncertainty
                mean_pred, log_var_pred = self.model(batch_X, return_uncertainty=True)
                
                # Bayesian loss (negative log likelihood)
                precision = torch.exp(-log_var_pred)
                loss = torch.mean(0.5 * precision * (batch_y - mean_pred)**2 + 0.5 * log_var_pred)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_mean, val_log_var = self.model(X_val_tensor, return_uncertainty=True)
                val_precision = torch.exp(-val_log_var)
                val_loss = torch.mean(0.5 * val_precision * (y_val_tensor - val_mean)**2 + 0.5 * val_log_var)
                
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss.item())
            
        # Generate predictions with uncertainty
        predictions = self._predict_with_uncertainty_pytorch(X_val_tensor)
        
        # Calculate metrics
        mean_preds = np.array([pred.mean for pred in predictions])
        mse = mean_squared_error(y_val, mean_preds)
        mae = mean_absolute_error(y_val, mean_preds)
        r2 = r2_score(y_val, mean_preds)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(mean_preds))
        directional_accuracy = np.mean(val_direction == pred_direction) if len(val_direction) > 0 else 0.0
        
        # Calibration score
        calibration_score = self._calculate_calibration_score(predictions, y_val)
        
        # Extract uncertainty arrays
        epistemic_uncertainty = np.array([pred.epistemic_uncertainty for pred in predictions])
        aleatoric_uncertainty = np.array([pred.aleatoric_uncertainty for pred in predictions])
        total_uncertainty = np.array([pred.total_uncertainty for pred in predictions])
        
        return BayesianNNResults(
            predictions=predictions,
            actual_values=y_val,
            mean_predictions=mean_preds,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            calibration_score=calibration_score,
            model_architecture=self.architecture,
            training_history={'train_loss': train_losses, 'val_loss': val_losses}
        )
        
    def _train_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int, batch_size: int) -> BayesianNNResults:
        """Train using TensorFlow."""
        # Train mean model
        mean_history = self.model['mean'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Train variance model (using squared residuals)
        mean_pred_train = self.model['mean'].predict(X_train)
        residuals_squared = (y_train - mean_pred_train.flatten())**2
        
        var_history = self.model['variance'].fit(
            X_train, residuals_squared,
            validation_data=(X_val, (y_val - self.model['mean'].predict(X_val).flatten())**2),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Generate predictions with uncertainty
        predictions = self._predict_with_uncertainty_tensorflow(X_val)
        
        # Calculate metrics
        mean_preds = np.array([pred.mean for pred in predictions])
        mse = mean_squared_error(y_val, mean_preds)
        mae = mean_absolute_error(y_val, mean_preds)
        r2 = r2_score(y_val, mean_preds)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(mean_preds))
        directional_accuracy = np.mean(val_direction == pred_direction) if len(val_direction) > 0 else 0.0
        
        # Calibration score
        calibration_score = self._calculate_calibration_score(predictions, y_val)
        
        # Extract uncertainty arrays
        epistemic_uncertainty = np.array([pred.epistemic_uncertainty for pred in predictions])
        aleatoric_uncertainty = np.array([pred.aleatoric_uncertainty for pred in predictions])
        total_uncertainty = np.array([pred.total_uncertainty for pred in predictions])
        
        return BayesianNNResults(
            predictions=predictions,
            actual_values=y_val,
            mean_predictions=mean_preds,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            calibration_score=calibration_score,
            model_architecture=self.architecture,
            training_history={
                'train_loss': mean_history.history['loss'],
                'val_loss': mean_history.history['val_loss']
            }
        )
        
    def _train_simplified(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> BayesianNNResults:
        """Train using simplified Bayesian models."""
        # Train Bayesian Ridge
        self.model['bayesian_ridge'].fit(X_train, y_train)
        
        # Train Random Forest for uncertainty estimation
        self.model['random_forest'].fit(X_train, y_train)
        
        # Generate predictions with uncertainty
        predictions = self._predict_with_uncertainty_simplified(X_val)
        
        # Calculate metrics
        mean_preds = np.array([pred.mean for pred in predictions])
        mse = mean_squared_error(y_val, mean_preds)
        mae = mean_absolute_error(y_val, mean_preds)
        r2 = r2_score(y_val, mean_preds)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(mean_preds))
        directional_accuracy = np.mean(val_direction == pred_direction) if len(val_direction) > 0 else 0.0
        
        # Calibration score
        calibration_score = self._calculate_calibration_score(predictions, y_val)
        
        # Extract uncertainty arrays
        epistemic_uncertainty = np.array([pred.epistemic_uncertainty for pred in predictions])
        aleatoric_uncertainty = np.array([pred.aleatoric_uncertainty for pred in predictions])
        total_uncertainty = np.array([pred.total_uncertainty for pred in predictions])
        
        return BayesianNNResults(
            predictions=predictions,
            actual_values=y_val,
            mean_predictions=mean_preds,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            calibration_score=calibration_score,
            model_architecture=self.architecture
        )
        
    def _predict_with_uncertainty_pytorch(self, X: torch.Tensor) -> List[BayesianPrediction]:
        """Generate predictions with uncertainty using Monte Carlo Dropout."""
        self.model.train()  # Keep dropout active
        
        predictions = []
        
        with torch.no_grad():
            # Monte Carlo sampling
            mc_predictions = []
            aleatoric_vars = []
            
            for _ in range(self.n_monte_carlo_samples):
                mean_pred, log_var_pred = self.model(X, return_uncertainty=True)
                mc_predictions.append(mean_pred.numpy().flatten())
                aleatoric_vars.append(torch.exp(log_var_pred).numpy().flatten())
                
            mc_predictions = np.array(mc_predictions)
            aleatoric_vars = np.array(aleatoric_vars)
            
            # Calculate uncertainties
            mean_pred = np.mean(mc_predictions, axis=0)
            epistemic_uncertainty = np.var(mc_predictions, axis=0)  # Model uncertainty
            aleatoric_uncertainty = np.mean(aleatoric_vars, axis=0)  # Data uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # Confidence intervals (95%)
            confidence_interval = 1.96 * np.sqrt(total_uncertainty)
            
            for i in range(len(mean_pred)):
                predictions.append(BayesianPrediction(
                    mean=mean_pred[i],
                    epistemic_uncertainty=epistemic_uncertainty[i],
                    aleatoric_uncertainty=aleatoric_uncertainty[i],
                    total_uncertainty=total_uncertainty[i],
                    confidence_interval_lower=mean_pred[i] - confidence_interval[i],
                    confidence_interval_upper=mean_pred[i] + confidence_interval[i]
                ))
                
        return predictions
        
    def _predict_with_uncertainty_tensorflow(self, X: np.ndarray) -> List[BayesianPrediction]:
        """Generate predictions with uncertainty using Monte Carlo Dropout."""
        predictions = []
        
        # Monte Carlo sampling
        mc_predictions = []
        
        for _ in range(self.n_monte_carlo_samples):
            # Enable dropout during inference
            mean_pred = self.model['mean'](X, training=True).numpy().flatten()
            mc_predictions.append(mean_pred)
            
        mc_predictions = np.array(mc_predictions)
        
        # Aleatoric uncertainty from variance model
        aleatoric_uncertainty = self.model['variance'].predict(X).flatten()
        
        # Calculate uncertainties
        mean_pred = np.mean(mc_predictions, axis=0)
        epistemic_uncertainty = np.var(mc_predictions, axis=0)  # Model uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals (95%)
        confidence_interval = 1.96 * np.sqrt(total_uncertainty)
        
        for i in range(len(mean_pred)):
            predictions.append(BayesianPrediction(
                mean=mean_pred[i],
                epistemic_uncertainty=epistemic_uncertainty[i],
                aleatoric_uncertainty=aleatoric_uncertainty[i],
                total_uncertainty=total_uncertainty[i],
                confidence_interval_lower=mean_pred[i] - confidence_interval[i],
                confidence_interval_upper=mean_pred[i] + confidence_interval[i]
            ))
            
        return predictions
        
    def _predict_with_uncertainty_simplified(self, X: np.ndarray) -> List[BayesianPrediction]:
        """Generate predictions with uncertainty using simplified models."""
        predictions = []
        
        # Bayesian Ridge predictions with uncertainty
        mean_pred, std_pred = self.model['bayesian_ridge'].predict(X, return_std=True)
        
        # Random Forest for additional uncertainty estimation
        rf_preds = []
        for estimator in self.model['random_forest'].estimators_:
            rf_preds.append(estimator.predict(X))
        rf_preds = np.array(rf_preds)
        
        # Epistemic uncertainty from Random Forest variance
        epistemic_uncertainty = np.var(rf_preds, axis=0)
        
        # Aleatoric uncertainty from Bayesian Ridge
        aleatoric_uncertainty = std_pred**2
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals (95%)
        confidence_interval = 1.96 * np.sqrt(total_uncertainty)
        
        for i in range(len(mean_pred)):
            predictions.append(BayesianPrediction(
                mean=mean_pred[i],
                epistemic_uncertainty=epistemic_uncertainty[i],
                aleatoric_uncertainty=aleatoric_uncertainty[i],
                total_uncertainty=total_uncertainty[i],
                confidence_interval_lower=mean_pred[i] - confidence_interval[i],
                confidence_interval_upper=mean_pred[i] + confidence_interval[i]
            ))
            
        return predictions
        
    def _calculate_calibration_score(self, predictions: List[BayesianPrediction], 
                                   actual_values: np.ndarray) -> float:
        """Calculate calibration score for uncertainty estimates."""
        # Check if actual values fall within confidence intervals
        within_ci = 0
        
        for i, pred in enumerate(predictions):
            if (actual_values[i] >= pred.confidence_interval_lower and 
                actual_values[i] <= pred.confidence_interval_upper):
                within_ci += 1
                
        # Expected coverage is 95% for 95% confidence intervals
        expected_coverage = 0.95
        actual_coverage = within_ci / len(predictions)
        
        # Calibration score (closer to 0 is better)
        calibration_score = abs(actual_coverage - expected_coverage)
        
        return calibration_score
        
    def predict(self, X: np.ndarray) -> List[BayesianPrediction]:
        """Make predictions with uncertainty estimates."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        X_scaled = self.scaler.transform(X)
        
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            X_tensor = torch.FloatTensor(X_scaled)
            return self._predict_with_uncertainty_pytorch(X_tensor)
        elif TF_AVAILABLE and isinstance(self.model, dict) and 'mean' in self.model:
            return self._predict_with_uncertainty_tensorflow(X_scaled)
        else:
            return self._predict_with_uncertainty_simplified(X_scaled)
            
    def plot_uncertainty(self, results: BayesianNNResults, sample_size: int = 100):
        """Plot predictions with uncertainty bands."""
        try:
            import matplotlib.pyplot as plt
            
            # Sample data for visualization
            if len(results.mean_predictions) > sample_size:
                indices = np.random.choice(len(results.mean_predictions), sample_size, replace=False)
                indices = np.sort(indices)
            else:
                indices = np.arange(len(results.mean_predictions))
                
            actual_sample = results.actual_values[indices]
            mean_sample = results.mean_predictions[indices]
            epistemic_sample = results.epistemic_uncertainty[indices]
            aleatoric_sample = results.aleatoric_uncertainty[indices]
            total_sample = results.total_uncertainty[indices]
            
            plt.figure(figsize=(15, 10))
            
            # Main prediction plot
            plt.subplot(2, 2, 1)
            plt.scatter(range(len(indices)), actual_sample, label='Actual', alpha=0.7, color='blue')
            plt.plot(range(len(indices)), mean_sample, label='Predicted', color='red', linewidth=2)
            
            # Uncertainty bands
            plt.fill_between(range(len(indices)), 
                           mean_sample - 1.96 * np.sqrt(total_sample),
                           mean_sample + 1.96 * np.sqrt(total_sample),
                           alpha=0.3, color='red', label='95% Confidence Interval')
            
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Predictions with Uncertainty')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Uncertainty decomposition
            plt.subplot(2, 2, 2)
            plt.plot(range(len(indices)), epistemic_sample, label='Epistemic (Model)', color='green')
            plt.plot(range(len(indices)), aleatoric_sample, label='Aleatoric (Data)', color='orange')
            plt.plot(range(len(indices)), total_sample, label='Total', color='red')
            plt.xlabel('Sample Index')
            plt.ylabel('Uncertainty')
            plt.title('Uncertainty Decomposition')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Residuals vs Uncertainty
            plt.subplot(2, 2, 3)
            residuals = np.abs(actual_sample - mean_sample)
            plt.scatter(np.sqrt(total_sample), residuals, alpha=0.6)
            plt.xlabel('Total Uncertainty (std)')
            plt.ylabel('Absolute Residuals')
            plt.title('Residuals vs Uncertainty')
            plt.grid(True, alpha=0.3)
            
            # Training history (if available)
            plt.subplot(2, 2, 4)
            if results.training_history:
                plt.plot(results.training_history['train_loss'], label='Training Loss')
                plt.plot(results.training_history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No training history available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Training History')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage
if __name__ == "__main__":
    # Generate sample stock data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features and noise
    true_coeffs = np.random.randn(n_features) * 0.5
    noise = np.random.randn(n_samples) * 0.2
    y = X @ true_coeffs + noise
    
    # Add trend
    trend = np.linspace(0, 5, n_samples)
    y += trend
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build and train Bayesian Neural Network
    bnn_model = BayesianNeuralNetwork(input_dim=n_features)
    
    try:
        print("Building Bayesian Neural Network...")
        bnn_model.build_model()
        
        print("Training Bayesian Neural Network...")
        results = bnn_model.train(X_train, y_train, X_test, y_test, epochs=50)
        
        print("\n=== Bayesian Neural Network Results ===")
        print(f"MSE: {results.mse:.6f}")
        print(f"MAE: {results.mae:.6f}")
        print(f"R²: {results.r2:.6f}")
        print(f"Directional Accuracy: {results.directional_accuracy:.6f}")
        print(f"Calibration Score: {results.calibration_score:.6f}")
        
        # Uncertainty statistics
        print(f"\nUncertainty Statistics:")
        print(f"Mean Epistemic Uncertainty: {np.mean(results.epistemic_uncertainty):.6f}")
        print(f"Mean Aleatoric Uncertainty: {np.mean(results.aleatoric_uncertainty):.6f}")
        print(f"Mean Total Uncertainty: {np.mean(results.total_uncertainty):.6f}")
        
        # Plot results
        bnn_model.plot_uncertainty(results)
        
        print("\nBayesian Neural Network training completed successfully!")
        
    except Exception as e:
        print(f"Bayesian Neural Network training failed: {e}")
        import traceback
        traceback.print_exc()