import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class DCFModel(BaseModel):
    """
    Discounted Cash Flow (DCF) model for stock valuation.
    Values stocks based on projected future cash flows.
    """
    
    def __init__(self):
        super().__init__(
            model_id="stock_dcf",
            name="DCF Model",
            category=AssetCategory.STOCKS,
            model_type=ModelType.FUNDAMENTAL,
            description="Discounted Cash Flow valuation model"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare financial data for DCF analysis.
        """
        required_cols = ['timestamp', 'close', 'free_cash_flow', 'revenue', 'shares_outstanding']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for DCF model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate growth rates
        df['fcf_growth'] = df['free_cash_flow'].pct_change()
        df['revenue_growth'] = df['revenue'].pct_change()
        
        # Calculate FCF per share
        df['fcf_per_share'] = df['free_cash_flow'] / df['shares_outstanding']
        
        return df
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train DCF model by estimating growth rates and discount rate.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            prepared_data = prepared_data.dropna()
            
            # Calculate historical growth rates
            fcf_growth_rate = prepared_data['fcf_growth'].mean()
            revenue_growth_rate = prepared_data['revenue_growth'].mean()
            
            # Estimate discount rate (WACC approximation)
            # Simplified: use risk-free rate + equity risk premium
            risk_free_rate = kwargs.get('risk_free_rate', 0.03)  # 3% default
            equity_risk_premium = kwargs.get('equity_risk_premium', 0.06)  # 6% default
            beta = kwargs.get('beta', 1.0)  # Market beta
            
            discount_rate = risk_free_rate + beta * equity_risk_premium
            
            # Terminal growth rate (conservative)
            terminal_growth_rate = kwargs.get('terminal_growth_rate', 0.025)  # 2.5%
            
            self.parameters = {
                'fcf_growth_rate': fcf_growth_rate,
                'revenue_growth_rate': revenue_growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth_rate': terminal_growth_rate,
                'projection_years': kwargs.get('projection_years', 5)
            }
            
            # Calculate intrinsic values for historical data
            intrinsic_values = []
            actual_prices = []
            
            for i, row in prepared_data.iterrows():
                if pd.notna(row['fcf_per_share']):
                    intrinsic_value = self._calculate_dcf_value(
                        row['fcf_per_share'],
                        fcf_growth_rate,
                        discount_rate,
                        terminal_growth_rate
                    )
                    intrinsic_values.append(intrinsic_value)
                    actual_prices.append(row['close'])
            
            if len(intrinsic_values) > 0:
                # Calculate accuracy based on price vs intrinsic value correlation
                correlation = np.corrcoef(actual_prices, intrinsic_values)[0, 1]
                accuracy = abs(correlation) * 100
            else:
                accuracy = 0
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'parameters': self.parameters,
                'accuracy': accuracy,
                'correlation': correlation if 'correlation' in locals() else 0
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_dcf_value(self, current_fcf: float, growth_rate: float, 
                           discount_rate: float, terminal_growth_rate: float) -> float:
        """
        Calculate DCF intrinsic value.
        """
        projection_years = self.parameters.get('projection_years', 5)
        
        # Project future cash flows
        projected_fcf = []
        for year in range(1, projection_years + 1):
            fcf = current_fcf * ((1 + growth_rate) ** year)
            projected_fcf.append(fcf)
        
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        
        # Discount all cash flows to present value
        pv_fcf = sum([fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
        pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
        
        return pv_fcf + pv_terminal
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate DCF-based intrinsic value predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            intrinsic_values = []
            signals = []
            
            for _, row in prepared_data.iterrows():
                if pd.notna(row['fcf_per_share']):
                    intrinsic_value = self._calculate_dcf_value(
                        row['fcf_per_share'],
                        self.parameters['fcf_growth_rate'],
                        self.parameters['discount_rate'],
                        self.parameters['terminal_growth_rate']
                    )
                    
                    intrinsic_values.append(intrinsic_value)
                    
                    # Generate trading signal
                    current_price = row['close']
                    margin_of_safety = 0.2  # 20% margin of safety
                    
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
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class CAPMModel(BaseModel):
    """
    Capital Asset Pricing Model (CAPM) for risk assessment and expected returns.
    """
    
    def __init__(self):
        super().__init__(
            model_id="stock_capm",
            name="CAPM",
            category=AssetCategory.STOCKS,
            model_type=ModelType.FUNDAMENTAL,
            description="Capital Asset Pricing Model for risk assessment"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for CAPM analysis.
        """
        required_cols = ['timestamp', 'close', 'market_return']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for CAPM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate stock returns
        df['stock_return'] = df['close'].pct_change()
        
        # Calculate excess returns
        risk_free_rate = kwargs.get('risk_free_rate', 0.03) / 252  # Daily risk-free rate
        df['excess_stock_return'] = df['stock_return'] - risk_free_rate
        df['excess_market_return'] = df['market_return'] - risk_free_rate
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train CAPM model to estimate beta and alpha.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            X = prepared_data['excess_market_return'].values.reshape(-1, 1)
            y = prepared_data['excess_stock_return'].values
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model_instance = LinearRegression()
            self.model_instance.fit(X_train, y_train)
            
            # Extract CAPM parameters
            beta = self.model_instance.coef_[0]
            alpha = self.model_instance.intercept_
            
            # Calculate R-squared
            r_squared = self.model_instance.score(X_test, y_test)
            
            # Calculate other risk metrics
            stock_volatility = prepared_data['stock_return'].std() * np.sqrt(252)  # Annualized
            market_volatility = prepared_data['market_return'].std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = kwargs.get('risk_free_rate', 0.03)
            mean_return = prepared_data['stock_return'].mean() * 252  # Annualized
            sharpe_ratio = (mean_return - risk_free_rate) / stock_volatility
            
            self.parameters = {
                'beta': beta,
                'alpha': alpha,
                'r_squared': r_squared,
                'stock_volatility': stock_volatility,
                'market_volatility': market_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_free_rate': risk_free_rate
            }
            
            accuracy = r_squared * 100
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'beta': beta,
                'alpha': alpha,
                'r_squared': r_squared,
                'accuracy': accuracy,
                'risk_metrics': {
                    'volatility': stock_volatility,
                    'sharpe_ratio': sharpe_ratio
                }
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate CAPM-based expected returns and risk assessments.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data, **kwargs)
            
            # Calculate expected returns using CAPM
            expected_returns = []
            risk_assessments = []
            
            for _, row in prepared_data.iterrows():
                market_excess_return = row['excess_market_return']
                expected_excess_return = self.parameters['alpha'] + self.parameters['beta'] * market_excess_return
                expected_return = expected_excess_return + self.parameters['risk_free_rate'] / 252
                
                expected_returns.append(expected_return)
                
                # Risk assessment based on beta
                if self.parameters['beta'] > 1.2:
                    risk_level = 'High'
                elif self.parameters['beta'] > 0.8:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
                
                risk_assessments.append(risk_level)
            
            return {
                'status': 'success',
                'expected_returns': expected_returns,
                'risk_assessments': risk_assessments,
                'beta': self.parameters['beta'],
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class StockLSTMModel(BaseModel):
    """
    LSTM Neural Network model for stock price prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="stock_lstm",
            name="LSTM Neural Network",
            category=AssetCategory.STOCKS,
            model_type=ModelType.ML,
            description="Long Short-Term Memory network for stock prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for LSTM training.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for LSTM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate technical indicators
        df['returns'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Normalize features
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd', 'volatility']
        for col in feature_cols:
            if col in df.columns:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> tuple:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 3])  # Close price index
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train LSTM model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Prepare features for LSTM
            feature_cols = [col for col in prepared_data.columns if col.endswith('_norm')]
            features = prepared_data[feature_cols].values
            
            sequence_length = kwargs.get('sequence_length', 60)
            X, y = self._create_sequences(features, sequence_length)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Simplified LSTM using sklearn (for demonstration)
            # In practice, you would use TensorFlow/PyTorch
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            
            # Flatten sequences for sklearn
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            self.model_instance = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_instance.fit(X_train_flat, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test_flat)
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate accuracy (simplified)
            accuracy = max(0, 100 - mse * 100)
            
            self.parameters = {
                'sequence_length': sequence_length,
                'n_features': len(feature_cols),
                'mse': mse
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'mse': mse,
                'accuracy': accuracy,
                'parameters': self.parameters
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate LSTM predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            feature_cols = [col for col in prepared_data.columns if col.endswith('_norm')]
            features = prepared_data[feature_cols].values
            
            sequence_length = self.parameters['sequence_length']
            
            if len(features) < sequence_length:
                return {'status': 'error', 'message': 'Insufficient data for prediction'}
            
            # Create sequences
            X, _ = self._create_sequences(features, sequence_length)
            X_flat = X.reshape(X.shape[0], -1)
            
            # Generate predictions
            predictions = self.model_instance.predict(X_flat)
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'timestamps': prepared_data['timestamp'].iloc[sequence_length:].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class StockXGBoostModel(BaseModel):
    """
    XGBoost model for stock price prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="stock_xgboost",
            name="XGBoost",
            category=AssetCategory.STOCKS,
            model_type=ModelType.ML,
            description="Gradient boosting for stock price prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for XGBoost training.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for XGBoost model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate features
        df['returns'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['price_change'] = df['close'] - df['open']
        
        # Technical indicators
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close'].rolling(window=window).std()
        
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'] = self._calculate_macd(df['close'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target variable (next day's return)
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Select features
            feature_cols = [col for col in prepared_data.columns 
                          if col not in ['timestamp', 'target'] and not col.startswith('close')]
            
            X = prepared_data[feature_cols]
            y = prepared_data['target']
            
            # Remove any remaining NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Use GradientBoostingRegressor as XGBoost alternative
            self.model_instance = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.model_instance.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, self.model_instance.feature_importances_))
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'feature_cols': feature_cols,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'mse': mse,
                'r2': r2,
                'accuracy': accuracy,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate XGBoost predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            X = prepared_data[self.parameters['feature_cols']]
            
            # Generate predictions
            predictions = self.model_instance.predict(X)
            
            # Convert returns to price predictions
            current_prices = prepared_data['close'].values
            price_predictions = current_prices * (1 + predictions)
            
            return {
                'status': 'success',
                'return_predictions': predictions.tolist(),
                'price_predictions': price_predictions.tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from XGBoost model."""
        return self.parameters.get('feature_importance')