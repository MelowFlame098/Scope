import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class APTModel(BaseModel):
    """
    Arbitrage Pricing Theory (APT) model for index prediction.
    Multi-factor model that relates asset returns to various economic factors.
    """
    
    def __init__(self):
        super().__init__(
            model_id="index_apt",
            name="Arbitrage Pricing Theory",
            category=AssetCategory.INDEXES,
            model_type=ModelType.FUNDAMENTAL,
            description="Multi-factor APT model for index prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for APT analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for APT model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate index returns
        df['returns'] = df['close'].pct_change()
        
        # Add economic factors (these would typically come from external data)
        # For demonstration, we'll create proxy factors
        
        # Market factor (broad market return)
        df['market_factor'] = kwargs.get('market_returns', df['returns'].rolling(window=252).mean())
        
        # Size factor (small vs large cap)
        df['size_factor'] = kwargs.get('size_factor', np.random.normal(0, 0.02, len(df)))
        
        # Value factor (value vs growth)
        df['value_factor'] = kwargs.get('value_factor', np.random.normal(0, 0.015, len(df)))
        
        # Momentum factor
        df['momentum_factor'] = df['returns'].rolling(window=12).mean()
        
        # Interest rate factor
        df['interest_rate_factor'] = kwargs.get('interest_rate_changes', np.random.normal(0, 0.001, len(df)))
        
        # Inflation factor
        df['inflation_factor'] = kwargs.get('inflation_changes', np.random.normal(0, 0.002, len(df)))
        
        # Oil price factor
        df['oil_factor'] = kwargs.get('oil_returns', np.random.normal(0, 0.03, len(df)))
        
        # Currency factor
        df['currency_factor'] = kwargs.get('currency_returns', np.random.normal(0, 0.01, len(df)))
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train APT model using multiple regression.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Define factors
            factor_cols = ['market_factor', 'size_factor', 'value_factor', 'momentum_factor',
                          'interest_rate_factor', 'inflation_factor', 'oil_factor', 'currency_factor']
            
            X = prepared_data[factor_cols].values
            y = prepared_data['returns'].values
            
            # Remove NaN values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Factor loadings (betas)
            factor_loadings = dict(zip(factor_cols, self.model_instance.coef_))
            alpha = self.model_instance.intercept_
            
            # Calculate factor importance
            factor_importance = {factor: abs(loading) for factor, loading in factor_loadings.items()}
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'factor_loadings': factor_loadings,
                'alpha': alpha,
                'r2': r2,
                'mse': mse,
                'factor_cols': factor_cols,
                'factor_importance': factor_importance
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'factor_loadings': factor_loadings,
                'alpha': alpha,
                'r2': r2,
                'accuracy': accuracy,
                'factor_importance': factor_importance
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate APT-based predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            X = prepared_data[self.parameters['factor_cols']].values
            
            # Generate predictions
            predicted_returns = self.model_instance.predict(X)
            
            # Convert to price predictions
            current_prices = prepared_data['close'].values
            predicted_prices = current_prices * (1 + predicted_returns)
            
            # Generate signals
            signals = []
            for pred_return in predicted_returns:
                if pred_return > 0.01:  # 1% threshold
                    signals.append(1)  # Buy
                elif pred_return < -0.01:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'predicted_returns': predicted_returns.tolist(),
                'predicted_prices': predicted_prices.tolist(),
                'signals': signals,
                'factor_contributions': self._calculate_factor_contributions(X),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_factor_contributions(self, X: np.ndarray) -> Dict[str, List[float]]:
        """Calculate individual factor contributions to predictions."""
        contributions = {}
        factor_cols = self.parameters['factor_cols']
        factor_loadings = self.parameters['factor_loadings']
        
        for i, factor in enumerate(factor_cols):
            contributions[factor] = (X[:, i] * factor_loadings[factor]).tolist()
        
        return contributions
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get factor importance from APT model."""
        return self.parameters.get('factor_importance')

class DDMModel(BaseModel):
    """
    Dividend Discount Model (DDM) for index valuation.
    """
    
    def __init__(self):
        super().__init__(
            model_id="index_ddm",
            name="Dividend Discount Model",
            category=AssetCategory.INDEXES,
            model_type=ModelType.FUNDAMENTAL,
            description="Dividend discount model for index valuation"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for DDM analysis.
        """
        required_cols = ['timestamp', 'close', 'dividend_yield']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for DDM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate dividend per share (approximation)
        df['dividend_per_share'] = df['close'] * df['dividend_yield'] / 100
        
        # Calculate dividend growth rate
        df['dividend_growth'] = df['dividend_per_share'].pct_change()
        
        # Calculate earnings yield (if available)
        if 'earnings_yield' in df.columns:
            df['payout_ratio'] = df['dividend_yield'] / df['earnings_yield']
        else:
            df['payout_ratio'] = 0.4  # Default 40% payout ratio
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train DDM model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Calculate historical dividend growth rate
            dividend_growth_rate = prepared_data['dividend_growth'].mean()
            
            # Estimate required rate of return
            # Using CAPM: r = rf + beta * (rm - rf)
            risk_free_rate = kwargs.get('risk_free_rate', 0.03)
            market_risk_premium = kwargs.get('market_risk_premium', 0.06)
            beta = kwargs.get('beta', 1.0)
            
            required_return = risk_free_rate + beta * market_risk_premium
            
            # Terminal growth rate (long-term GDP growth)
            terminal_growth_rate = kwargs.get('terminal_growth_rate', 0.025)
            
            # Calculate intrinsic values using Gordon Growth Model
            intrinsic_values = []
            actual_prices = []
            
            for _, row in prepared_data.iterrows():
                if pd.notna(row['dividend_per_share']) and dividend_growth_rate < required_return:
                    # Gordon Growth Model: P = D1 / (r - g)
                    next_dividend = row['dividend_per_share'] * (1 + dividend_growth_rate)
                    intrinsic_value = next_dividend / (required_return - dividend_growth_rate)
                    
                    intrinsic_values.append(intrinsic_value)
                    actual_prices.append(row['close'])
            
            if len(intrinsic_values) > 0:
                # Calculate correlation between intrinsic and actual values
                correlation = np.corrcoef(actual_prices, intrinsic_values)[0, 1]
                accuracy = abs(correlation) * 100
            else:
                correlation = 0
                accuracy = 0
            
            self.parameters = {
                'dividend_growth_rate': dividend_growth_rate,
                'required_return': required_return,
                'terminal_growth_rate': terminal_growth_rate,
                'risk_free_rate': risk_free_rate,
                'beta': beta,
                'correlation': correlation
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'dividend_growth_rate': dividend_growth_rate,
                'required_return': required_return,
                'correlation': correlation,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate DDM-based predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            intrinsic_values = []
            signals = []
            
            for _, row in prepared_data.iterrows():
                if pd.notna(row['dividend_per_share']):
                    # Calculate intrinsic value
                    next_dividend = row['dividend_per_share'] * (1 + self.parameters['dividend_growth_rate'])
                    
                    if self.parameters['dividend_growth_rate'] < self.parameters['required_return']:
                        intrinsic_value = next_dividend / (self.parameters['required_return'] - 
                                                         self.parameters['dividend_growth_rate'])
                    else:
                        intrinsic_value = row['close']  # Fallback to current price
                    
                    intrinsic_values.append(intrinsic_value)
                    
                    # Generate trading signal
                    current_price = row['close']
                    margin_of_safety = 0.15  # 15% margin
                    
                    if current_price < intrinsic_value * (1 - margin_of_safety):
                        signals.append(1)  # Buy - undervalued
                    elif current_price > intrinsic_value * (1 + margin_of_safety):
                        signals.append(-1)  # Sell - overvalued
                    else:
                        signals.append(0)  # Hold
                else:
                    intrinsic_values.append(None)
                    signals.append(0)
            
            return {
                'status': 'success',
                'intrinsic_values': intrinsic_values,
                'signals': signals,
                'dividend_yields': prepared_data['dividend_yield'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class KalmanFilterModel(BaseModel):
    """
    Kalman Filter model for index prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="index_kalman",
            name="Kalman Filters",
            category=AssetCategory.INDEXES,
            model_type=ModelType.STATISTICAL,
            description="Kalman filter for index trend estimation"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Kalman filter.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Kalman Filter model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate log prices for better numerical stability
        df['log_price'] = np.log(df['close'])
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Kalman filter (estimate parameters).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Simplified Kalman filter implementation
            # State: [price_level, trend]
            # Observation: log_price
            
            log_prices = prepared_data['log_price'].values
            n = len(log_prices)
            
            # Initialize parameters
            process_noise = kwargs.get('process_noise', 1e-4)
            observation_noise = kwargs.get('observation_noise', 1e-3)
            
            # State transition matrix (random walk with drift)
            F = np.array([[1, 1], [0, 1]])
            
            # Observation matrix
            H = np.array([[1, 0]])
            
            # Process noise covariance
            Q = np.array([[process_noise, 0], [0, process_noise]])
            
            # Observation noise covariance
            R = np.array([[observation_noise]])
            
            # Initialize state and covariance
            x = np.array([[log_prices[0]], [0]])  # [price, trend]
            P = np.eye(2) * 0.1
            
            # Store filtered states
            filtered_states = []
            predicted_states = []
            
            for i in range(n):
                # Prediction step
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                
                # Update step
                y = log_prices[i] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                
                x = x_pred + K @ y
                P = (np.eye(2) - K @ H) @ P_pred
                
                filtered_states.append(x.copy())
                predicted_states.append(x_pred.copy())
            
            # Calculate accuracy based on prediction error
            predicted_prices = [np.exp(state[0, 0]) for state in predicted_states[1:]]
            actual_prices = prepared_data['close'].values[1:]
            
            mse = np.mean((np.array(predicted_prices) - actual_prices) ** 2)
            mape = np.mean(np.abs((np.array(predicted_prices) - actual_prices) / actual_prices)) * 100
            
            accuracy = max(0, 100 - mape)
            
            self.parameters = {
                'F': F.tolist(),
                'H': H.tolist(),
                'Q': Q.tolist(),
                'R': R.tolist(),
                'process_noise': process_noise,
                'observation_noise': observation_noise,
                'final_state': x.tolist(),
                'final_covariance': P.tolist(),
                'mse': mse,
                'mape': mape
            }
            
            # Store the filter for prediction
            self.filtered_states = filtered_states
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'mse': mse,
                'mape': mape,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Kalman filter predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Get matrices from parameters
            F = np.array(self.parameters['F'])
            H = np.array(self.parameters['H'])
            Q = np.array(self.parameters['Q'])
            R = np.array(self.parameters['R'])
            
            # Start from final state
            x = np.array(self.parameters['final_state'])
            P = np.array(self.parameters['final_covariance'])
            
            log_prices = prepared_data['log_price'].values
            
            predictions = []
            filtered_prices = []
            trends = []
            
            for i, log_price in enumerate(log_prices):
                # Prediction step
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                
                # Store prediction
                predicted_log_price = H @ x_pred
                predictions.append(np.exp(predicted_log_price[0, 0]))
                
                # Update step
                y = log_price - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                
                x = x_pred + K @ y
                P = (np.eye(2) - K @ H) @ P_pred
                
                # Store filtered state
                filtered_prices.append(np.exp(x[0, 0]))
                trends.append(x[1, 0])
            
            # Generate trading signals based on trend
            signals = []
            for trend in trends:
                if trend > 0.001:  # Positive trend
                    signals.append(1)  # Buy
                elif trend < -0.001:  # Negative trend
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'predictions': predictions,
                'filtered_prices': filtered_prices,
                'trends': trends,
                'signals': signals,
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class VECMModel(BaseModel):
    """
    Vector Error Correction Model (VECM) for cointegrated index relationships.
    """
    
    def __init__(self):
        super().__init__(
            model_id="index_vecm",
            name="VECM",
            category=AssetCategory.INDEXES,
            model_type=ModelType.STATISTICAL,
            description="Vector Error Correction Model for cointegrated indexes"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for VECM analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for VECM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # For VECM, we need multiple related series
        # If additional series are provided, use them; otherwise create synthetic ones
        if 'related_index_1' not in df.columns:
            # Create synthetic related series for demonstration
            df['related_index_1'] = df['close'] * (1 + np.random.normal(0, 0.1, len(df)))
            df['related_index_2'] = df['close'] * (1 + np.random.normal(0, 0.15, len(df)))
        
        # Calculate log prices
        df['log_price'] = np.log(df['close'])
        df['log_related_1'] = np.log(df['related_index_1'])
        df['log_related_2'] = np.log(df['related_index_2'])
        
        # Calculate first differences (returns)
        df['d_log_price'] = df['log_price'].diff()
        df['d_log_related_1'] = df['log_related_1'].diff()
        df['d_log_related_2'] = df['log_related_2'].diff()
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train VECM model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Test for cointegration (simplified Engle-Granger test)
            log_prices = prepared_data[['log_price', 'log_related_1', 'log_related_2']].values
            
            from sklearn.linear_model import LinearRegression
            
            # Step 1: Estimate cointegrating relationship
            X = log_prices[:, 1:]  # Related indexes
            y = log_prices[:, 0]   # Main index
            
            coint_reg = LinearRegression()
            coint_reg.fit(X, y)
            
            # Calculate error correction term
            fitted_values = coint_reg.predict(X)
            error_correction_term = y - fitted_values
            
            # Step 2: Estimate VECM
            # Δy_t = α * ECT_{t-1} + Σ(Γ_i * Δy_{t-i}) + ε_t
            
            # Prepare lagged differences
            lags = kwargs.get('lags', 2)
            
            # Create lagged variables
            lagged_data = []
            for lag in range(1, lags + 1):
                lagged_diff = prepared_data[['d_log_price', 'd_log_related_1', 'd_log_related_2']].shift(lag)
                lagged_data.append(lagged_diff.values)
            
            # Combine features: ECT + lagged differences
            ect_lagged = error_correction_term[:-1]  # Lag ECT by 1
            
            X_vecm = [ect_lagged.reshape(-1, 1)]
            for lagged in lagged_data:
                X_vecm.append(lagged[lags:-1])  # Align with ECT
            
            X_vecm = np.hstack(X_vecm)
            y_vecm = prepared_data['d_log_price'].values[lags:]  # Target: current period change
            
            # Remove NaN values
            mask = ~np.isnan(X_vecm).any(axis=1)
            X_vecm = X_vecm[mask]
            y_vecm = y_vecm[mask]
            
            # Estimate VECM
            vecm_reg = LinearRegression()
            vecm_reg.fit(X_vecm, y_vecm)
            
            # Evaluate
            y_pred = vecm_reg.predict(X_vecm)
            from sklearn.metrics import r2_score, mean_squared_error
            
            r2 = r2_score(y_vecm, y_pred)
            mse = mean_squared_error(y_vecm, y_pred)
            
            # Extract coefficients
            alpha = vecm_reg.coef_[0]  # Error correction coefficient
            gamma_coeffs = vecm_reg.coef_[1:]  # Lagged difference coefficients
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'cointegrating_coeffs': coint_reg.coef_.tolist(),
                'cointegrating_intercept': coint_reg.intercept_,
                'alpha': alpha,
                'gamma_coeffs': gamma_coeffs.tolist(),
                'vecm_intercept': vecm_reg.intercept_,
                'lags': lags,
                'r2': r2,
                'mse': mse
            }
            
            # Store models for prediction
            self.coint_model = coint_reg
            self.vecm_model = vecm_reg
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'alpha': alpha,
                'r2': r2,
                'mse': mse,
                'accuracy': accuracy,
                'cointegrating_coeffs': coint_reg.coef_.tolist()
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate VECM predictions.
        """
        if not hasattr(self, 'vecm_model') or self.vecm_model is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            log_prices = prepared_data[['log_price', 'log_related_1', 'log_related_2']].values
            
            # Calculate error correction terms
            X_coint = log_prices[:, 1:]
            y_coint = log_prices[:, 0]
            fitted_values = self.coint_model.predict(X_coint)
            error_correction_terms = y_coint - fitted_values
            
            # Prepare VECM features
            lags = self.parameters['lags']
            
            predictions = []
            signals = []
            
            for i in range(lags, len(prepared_data)):
                # ECT lagged by 1
                ect = error_correction_terms[i-1]
                
                # Lagged differences
                lagged_features = [ect]
                
                for lag in range(1, lags + 1):
                    if i - lag >= 0:
                        lagged_diff = prepared_data[['d_log_price', 'd_log_related_1', 'd_log_related_2']].iloc[i-lag].values
                        lagged_features.extend(lagged_diff)
                
                X_pred = np.array(lagged_features).reshape(1, -1)
                
                # Predict change in log price
                predicted_change = self.vecm_model.predict(X_pred)[0]
                
                # Convert to price prediction
                current_log_price = prepared_data['log_price'].iloc[i]
                predicted_log_price = current_log_price + predicted_change
                predicted_price = np.exp(predicted_log_price)
                
                predictions.append(predicted_price)
                
                # Generate signal based on error correction
                if ect < -0.02:  # Below equilibrium
                    signals.append(1)  # Buy - expect mean reversion
                elif ect > 0.02:  # Above equilibrium
                    signals.append(-1)  # Sell - expect mean reversion
                else:
                    signals.append(0)  # Hold
            
            # Pad with None for alignment
            predictions = [None] * lags + predictions
            signals = [0] * lags + signals
            
            return {
                'status': 'success',
                'predictions': predictions,
                'signals': signals,
                'error_correction_terms': error_correction_terms.tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class ElliottWaveModel(BaseModel):
    """
    Elliott Wave analysis model for index prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="index_elliott_wave",
            name="Elliott Wave",
            category=AssetCategory.INDEXES,
            model_type=ModelType.TECHNICAL,
            description="Elliott Wave pattern analysis for index prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for Elliott Wave analysis.
        """
        required_cols = ['timestamp', 'high', 'low', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Elliott Wave model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Identify swing highs and lows
        window = kwargs.get('swing_window', 5)
        
        df['swing_high'] = df['high'].rolling(window=window*2+1, center=True).max() == df['high']
        df['swing_low'] = df['low'].rolling(window=window*2+1, center=True).min() == df['low']
        
        # Calculate price momentum
        df['momentum'] = df['close'].pct_change(periods=10)
        
        # Calculate volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df.dropna()
    
    def _identify_waves(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify potential Elliott Wave patterns (simplified).
        """
        waves = []
        
        # Find swing points
        swing_highs = data[data['swing_high']]
        swing_lows = data[data['swing_low']]
        
        # Combine and sort swing points
        swing_points = []
        
        for _, row in swing_highs.iterrows():
            swing_points.append({
                'timestamp': row['timestamp'],
                'price': row['high'],
                'type': 'high'
            })
        
        for _, row in swing_lows.iterrows():
            swing_points.append({
                'timestamp': row['timestamp'],
                'price': row['low'],
                'type': 'low'
            })
        
        swing_points.sort(key=lambda x: x['timestamp'])
        
        # Identify 5-wave patterns (simplified)
        if len(swing_points) >= 5:
            for i in range(len(swing_points) - 4):
                pattern = swing_points[i:i+5]
                
                # Check for alternating high-low pattern
                types = [p['type'] for p in pattern]
                if len(set(types)) == 2:  # Has both highs and lows
                    waves.append({
                        'start_time': pattern[0]['timestamp'],
                        'end_time': pattern[-1]['timestamp'],
                        'pattern': pattern,
                        'wave_count': 5
                    })
        
        return waves
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Elliott Wave model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Identify wave patterns
            waves = self._identify_waves(prepared_data)
            
            # Analyze wave characteristics
            wave_stats = {
                'total_waves': len(waves),
                'avg_wave_duration': 0,
                'avg_wave_magnitude': 0
            }
            
            if waves:
                durations = []
                magnitudes = []
                
                for wave in waves:
                    duration = (wave['end_time'] - wave['start_time']).days
                    durations.append(duration)
                    
                    prices = [p['price'] for p in wave['pattern']]
                    magnitude = (max(prices) - min(prices)) / min(prices)
                    magnitudes.append(magnitude)
                
                wave_stats['avg_wave_duration'] = np.mean(durations)
                wave_stats['avg_wave_magnitude'] = np.mean(magnitudes)
            
            # Calculate Fibonacci retracement levels
            fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            # Analyze success rate of Fibonacci levels
            fibonacci_accuracy = self._analyze_fibonacci_accuracy(prepared_data, fibonacci_levels)
            
            accuracy = fibonacci_accuracy * 100
            
            self.parameters = {
                'wave_stats': wave_stats,
                'fibonacci_levels': fibonacci_levels,
                'fibonacci_accuracy': fibonacci_accuracy,
                'swing_window': kwargs.get('swing_window', 5)
            }
            
            self.waves = waves
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'wave_count': len(waves),
                'fibonacci_accuracy': fibonacci_accuracy,
                'accuracy': accuracy,
                'wave_stats': wave_stats
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def _analyze_fibonacci_accuracy(self, data: pd.DataFrame, fib_levels: List[float]) -> float:
        """
        Analyze how often price respects Fibonacci retracement levels.
        """
        swing_highs = data[data['swing_high']]
        swing_lows = data[data['swing_low']]
        
        total_tests = 0
        successful_tests = 0
        
        # Test Fibonacci retracements
        for i in range(len(swing_highs) - 1):
            for j in range(len(swing_lows) - 1):
                high_row = swing_highs.iloc[i]
                low_row = swing_lows.iloc[j]
                
                if high_row['timestamp'] < low_row['timestamp']:
                    # Downtrend retracement
                    high_price = high_row['high']
                    low_price = low_row['low']
                    
                    for fib_level in fib_levels:
                        retracement_price = low_price + (high_price - low_price) * fib_level
                        
                        # Check if price found support/resistance near this level
                        future_data = data[data['timestamp'] > low_row['timestamp']].head(20)
                        
                        if not future_data.empty:
                            tolerance = 0.02  # 2% tolerance
                            near_fib = ((future_data['low'] <= retracement_price * (1 + tolerance)) & 
                                      (future_data['low'] >= retracement_price * (1 - tolerance))).any()
                            
                            total_tests += 1
                            if near_fib:
                                successful_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.5
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Elliott Wave predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Identify current wave position
            current_waves = self._identify_waves(prepared_data.tail(100))  # Recent data
            
            predictions = []
            signals = []
            fibonacci_levels = []
            
            for _, row in prepared_data.iterrows():
                current_price = row['close']
                
                # Calculate Fibonacci retracement levels from recent swing points
                recent_data = prepared_data[prepared_data['timestamp'] <= row['timestamp']].tail(50)
                
                if len(recent_data) > 10:
                    recent_high = recent_data['high'].max()
                    recent_low = recent_data['low'].min()
                    
                    fib_levels = {}
                    for level in self.parameters['fibonacci_levels']:
                        fib_price = recent_low + (recent_high - recent_low) * level
                        fib_levels[f'fib_{level}'] = fib_price
                    
                    fibonacci_levels.append(fib_levels)
                    
                    # Generate prediction based on wave position
                    if current_price < fib_levels['fib_0.382']:
                        predicted_direction = 1  # Expect bounce
                        signal = 1  # Buy
                    elif current_price > fib_levels['fib_0.618']:
                        predicted_direction = -1  # Expect pullback
                        signal = -1  # Sell
                    else:
                        predicted_direction = 0  # Neutral
                        signal = 0  # Hold
                    
                    # Simple prediction: current price + direction * average magnitude
                    avg_magnitude = self.parameters['wave_stats']['avg_wave_magnitude']
                    predicted_price = current_price * (1 + predicted_direction * avg_magnitude * 0.1)
                    
                    predictions.append(predicted_price)
                    signals.append(signal)
                else:
                    predictions.append(current_price)
                    signals.append(0)
                    fibonacci_levels.append({})
            
            return {
                'status': 'success',
                'predictions': predictions,
                'signals': signals,
                'fibonacci_levels': fibonacci_levels,
                'current_waves': len(current_waves),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)