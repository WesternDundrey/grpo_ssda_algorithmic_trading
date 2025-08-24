"""State-Space Denoising Autoencoder for Financial Time Series"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from typing import Tuple, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StateSpaceModel:
    """State-space representation for financial time series"""
    
    def __init__(self, state_dim: int = 8, obs_dim: int = 5):
        """
        Initialize state-space model
        
        Args:
            state_dim: Dimension of hidden state space
            obs_dim: Dimension of observations (OHLCV)
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # State transition matrix (A)
        self.A = np.eye(state_dim) * 0.95 + np.random.normal(0, 0.01, (state_dim, state_dim))
        
        # Observation matrix (C) 
        self.C = np.random.normal(0, 0.1, (obs_dim, state_dim))
        
        # Process noise covariance (Q)
        self.Q = np.eye(state_dim) * 0.01
        
        # Observation noise covariance (R)
        self.R = np.eye(obs_dim) * 0.1
        
        # Initial state and covariance
        self.x0 = np.zeros(state_dim)
        self.P0 = np.eye(state_dim)
    
    def predict_state(self, x_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state using transition model"""
        x_pred = self.A @ x_prev
        P_pred = self.A @ self.P0 @ self.A.T + self.Q
        return x_pred, P_pred
    
    def update_state(self, x_pred: np.ndarray, P_pred: np.ndarray, 
                    obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state given observation using Kalman filter"""
        # Innovation
        y = obs - self.C @ x_pred
        
        # Innovation covariance
        S = self.C @ P_pred @ self.C.T + self.R
        
        # Kalman gain
        K = P_pred @ self.C.T @ np.linalg.pinv(S)
        
        # Updated state and covariance
        x_upd = x_pred + K @ y
        P_upd = (np.eye(self.state_dim) - K @ self.C) @ P_pred
        
        return x_upd, P_upd
    
    def filter_sequence(self, observations: np.ndarray) -> np.ndarray:
        """Apply Kalman filtering to observation sequence"""
        T, _ = observations.shape
        states = np.zeros((T, self.state_dim))
        
        # Initialize
        x = self.x0.copy()
        P = self.P0.copy()
        
        for t in range(T):
            # Predict
            x_pred, P_pred = self.predict_state(x)
            
            # Update
            x, P = self.update_state(x_pred, P_pred, observations[t])
            
            states[t] = x
        
        return states


class DenoisingAutoencoder:
    """Neural network-based denoising autoencoder"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16, 32, 64],
                 noise_factor: float = 0.1):
        """
        Initialize denoising autoencoder
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions (symmetric for encoder/decoder)
            noise_factor: Noise level for training
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.noise_factor = noise_factor
        self.scaler = StandardScaler()
        
        # Build encoder and decoder using MLPRegressor
        mid_idx = len(hidden_dims) // 2
        encoder_dims = [input_dim] + hidden_dims[:mid_idx+1]
        decoder_dims = hidden_dims[mid_idx:] + [input_dim]
        
        # Use MLPRegressor as approximation for autoencoder
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            activation='tanh',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.is_fitted = False
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, self.noise_factor, data.shape)
        return data + noise
    
    def fit(self, data: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """
        Train the denoising autoencoder
        
        Args:
            data: Clean training data (T, features)
            epochs: Training epochs (unused with MLPRegressor)
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Training denoising autoencoder on {data.shape[0]} samples")
        
        # Normalize data
        data_normalized = self.scaler.fit_transform(data)
        
        # Create noisy input and clean target pairs
        noisy_data = self.add_noise(data_normalized)
        
        # Train autoencoder to reconstruct clean data from noisy input
        self.autoencoder.fit(noisy_data, data_normalized)
        
        self.is_fitted = True
        
        # Calculate training loss
        reconstructed = self.autoencoder.predict(noisy_data)
        loss = np.mean((reconstructed - data_normalized) ** 2)
        
        return {
            'final_loss': loss,
            'n_samples': len(data),
            'iterations': self.autoencoder.n_iter_
        }
    
    def denoise(self, noisy_data: np.ndarray) -> np.ndarray:
        """Denoise input data"""
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before denoising")
        
        # Normalize and denoise
        noisy_normalized = self.scaler.transform(noisy_data)
        denoised_normalized = self.autoencoder.predict(noisy_normalized)
        
        # Inverse transform
        return self.scaler.inverse_transform(denoised_normalized)
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Get encoded representation (bottleneck features)"""
        # For MLPRegressor, we approximate encoding by using the model
        # In practice, you'd want a proper encoder-decoder architecture
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before encoding")
        
        normalized = self.scaler.transform(data)
        # This is an approximation - in real implementation you'd extract
        # the bottleneck layer activations
        encoded = self.autoencoder.predict(normalized)
        # Safe indexing to avoid out of bounds errors
        if len(self.hidden_dims) > 0:
            bottleneck_size = self.hidden_dims[len(self.hidden_dims)//2]
            return encoded[:, :min(bottleneck_size, encoded.shape[1])]
        else:
            return encoded


class SSDA:
    """State-Space Denoising Autoencoder"""
    
    def __init__(self, 
                 state_dim: int = 8,
                 hidden_dims: list = [64, 32, 16, 32, 64],
                 noise_factor: float = 0.1,
                 lookback_window: int = 20):
        """
        Initialize SSDA
        
        Args:
            state_dim: State space dimension
            hidden_dims: Autoencoder hidden dimensions
            noise_factor: Noise level for training
            lookback_window: Historical window for features
        """
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.noise_factor = noise_factor
        self.lookback_window = lookback_window
        
        self.state_space_model = None
        self.autoencoder = None
        self.feature_scaler = StandardScaler()
        
        self.is_fitted = False
    
    def _create_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """Create technical features from OHLCV data"""
        features_list = []
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in price_data.columns:
                series = price_data[col]
                
                # Price features
                features_list.append(series.values)
                
                # Returns
                returns = series.pct_change().fillna(0)
                features_list.append(returns.values)
                
                # Moving averages
                ma_short = series.rolling(5).mean().fillna(series)
                ma_long = series.rolling(20).mean().fillna(series)
                features_list.append((series / ma_short - 1).fillna(0).values)
                features_list.append((ma_short / ma_long - 1).fillna(0).values)
                
                # Volatility
                volatility = returns.rolling(10).std().fillna(returns.std())
                features_list.append(volatility.values)
        
        # RSI approximation
        if 'close' in price_data.columns:
            close = price_data['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features_list.append(rsi.fillna(50).values)
        
        if not features_list:
            # If no features were created, return a minimal feature set
            return np.zeros((len(price_data), 1))
        
        features = np.column_stack(features_list)
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _create_sequences(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for training"""
        T, n_features = features.shape
        
        if T < self.lookback_window:
            raise ValueError(f"Not enough data points. Need at least {self.lookback_window}, got {T}")
        
        # Create sliding windows
        sequences = []
        targets = []
        
        for i in range(self.lookback_window, T):
            seq = features[i-self.lookback_window:i].flatten()
            target = features[i]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, price_data: pd.DataFrame, epochs: int = 100) -> Dict[str, Any]:
        """
        Train the SSDA model
        
        Args:
            price_data: DataFrame with OHLCV data
            epochs: Training epochs
        
        Returns:
            Training history
        """
        logger.info(f"Training SSDA on {len(price_data)} time steps")
        
        # Create features
        features = self._create_features(price_data)
        logger.info(f"Created {features.shape[1]} features")
        
        # Normalize features
        features_normalized = self.feature_scaler.fit_transform(features)
        
        # Initialize state-space model
        self.state_space_model = StateSpaceModel(
            state_dim=self.state_dim,
            obs_dim=min(5, features.shape[1])  # Use first 5 features for state-space
        )
        
        # Apply Kalman filtering to get state representations
        obs_subset = features_normalized[:, :self.state_space_model.obs_dim]
        states = self.state_space_model.filter_sequence(obs_subset)
        
        # Create sequences for autoencoder training
        sequences, targets = self._create_sequences(features_normalized)
        
        # Initialize and train autoencoder
        seq_dim = sequences.shape[1]
        self.autoencoder = DenoisingAutoencoder(
            input_dim=seq_dim,
            hidden_dims=self.hidden_dims,
            noise_factor=self.noise_factor
        )
        
        # Train autoencoder
        ae_history = self.autoencoder.fit(sequences, epochs=epochs)
        
        self.is_fitted = True
        
        history = {
            'autoencoder_history': ae_history,
            'n_features': features.shape[1],
            'n_sequences': len(sequences),
            'state_dim': self.state_dim
        }
        
        logger.info(f"SSDA training completed. Final loss: {ae_history['final_loss']:.6f}")
        
        return history
    
    def denoise_and_predict(self, price_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denoise price data and predict next values
        
        Args:
            price_data: Recent price data
            
        Returns:
            (denoised_features, state_representation)
        """
        if not self.is_fitted:
            raise ValueError("SSDA must be fitted before prediction")
        
        # Create features
        features = self._create_features(price_data)
        features_normalized = self.feature_scaler.transform(features)
        
        # Get state representation
        obs_subset = features_normalized[:, :self.state_space_model.obs_dim]
        states = self.state_space_model.filter_sequence(obs_subset)
        
        # Denoise recent sequence
        if len(features_normalized) >= self.lookback_window:
            recent_seq = features_normalized[-self.lookback_window:].flatten().reshape(1, -1)
            denoised_seq = self.autoencoder.denoise(recent_seq)
            denoised_features = denoised_seq.reshape(self.lookback_window, -1)
        else:
            # If not enough history, just return normalized features
            denoised_features = features_normalized
        
        return denoised_features, states
    
    def get_trading_signal(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate trading signals from denoised data
        
        Args:
            price_data: Recent price data
            
        Returns:
            Dictionary of trading signals
        """
        if not self.is_fitted:
            return {'signal': 0.0, 'confidence': 0.0}
        
        try:
            denoised_features, states = self.denoise_and_predict(price_data)
            
            # Extract signal from state representation
            if len(states) > 1:
                # Use state change as momentum signal
                state_momentum = np.mean(states[-1] - states[-2])
                
                # Use denoised features for trend detection
                if len(denoised_features) > 1 and denoised_features.shape[1] >= 5:
                    # Safe indexing for features
                    current_features = denoised_features[-1, :min(5, denoised_features.shape[1])]
                    previous_features = denoised_features[-2, :min(5, denoised_features.shape[1])]
                    price_trend = np.mean(current_features) - np.mean(previous_features)
                else:
                    price_trend = 0.0
                
                # Combine signals
                signal = np.tanh(state_momentum + price_trend)  # Bounded between -1 and 1
                confidence = min(abs(state_momentum) + abs(price_trend), 1.0)
                
            else:
                signal = 0.0
                confidence = 0.0
            
            return {
                'signal': float(signal),
                'confidence': float(confidence),
                'state_momentum': float(state_momentum) if len(states) > 1 else 0.0,
                'price_trend': float(price_trend) if 'price_trend' in locals() else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Error generating trading signal: {e}")
            return {'signal': 0.0, 'confidence': 0.0}