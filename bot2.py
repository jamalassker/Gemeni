# deep_lea# d# deep_learning_scalping_bot.py
"""
Deep Learning Scalping Bot with Neural Network Predictions
Monitors 20 Most Volatile Cryptocurrencies
Hourly Profit Targets with $100 Capital
"""

import os
import time
import json
import sqlite3
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import ccxt
from collections import deque, defaultdict
import threading
import sys
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from scipy import stats

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# ULTIMATE CONFIGURATION - DEEP LEARNING SCALPING
# ============================================================================

class UltimateConfig:
    """Ultimate configuration for deep learning scalping"""
    
    # Exchange & Symbols
    EXCHANGE = 'binance'
    
    # Top 20 most volatile cryptocurrencies (based on historical data)
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT',
        'SHIB/USDT', 'TRX/USDT', 'LINK/USDT', 'ATOM/USDT', 'UNI/USDT',
        'LTC/USDT', 'ETC/USDT', 'XLM/USDT', 'ALGO/USDT', 'VET/USDT'
    ]
    
    # Timeframes for multi-timeframe analysis
    TIMEFRAMES = ['1m', '5m', '15m']  # Multi-timeframe analysis
    PRIMARY_TIMEFRAME = '1m'  # Main scalping timeframe
    
    # Capital Management (Starting with $100)
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.008  # 0.8% risk per trade
    MAX_DAILY_LOSS = 0.015  # 1.5% daily loss limit
    HOURLY_TARGET = 0.005   # 0.5% hourly profit target
    DAILY_TARGET = 0.02     # 2% daily target
    
    # Position Sizing
    MAX_POSITION_SIZE = 0.15  # 15% max per symbol
    MAX_OPEN_TRADES = 8       # Max 8 open trades
    MAX_SAME_SYMBOL = 2       # Max 2 trades per symbol
    
    # Scalping Parameters (ULTRA FAST)
    BASE_TAKE_PROFIT = 0.0020  # 0.20% base take profit
    BASE_STOP_LOSS = 0.0015    # 0.15% base stop loss
    TRAILING_STOP = 0.0008     # 0.08% trailing stop
    
    # Dynamic Adjustments based on volatility
    VOLATILITY_MULTIPLIER = {
        'low': 0.7,
        'medium': 1.0,
        'high': 1.3,
        'extreme': 1.6
    }
    
    # Fees & Spread
    MAKER_FEE = 0.00075
    TAKER_FEE = 0.0010
    MIN_SPREAD = 0.00005  # 0.005% minimum
    MAX_SPREAD = 0.0003   # 0.03% maximum
    
    # Deep Learning Parameters
    DL_LOOKBACK = 100           # Look back 100 candles
    DL_TRAIN_INTERVAL = 50      # Retrain every 50 cycles
    DL_MIN_CONFIDENCE = 0.68    # 68% minimum confidence
    DL_ENSEMBLE_SIZE = 3        # Number of models in ensemble
    
    # Feature Engineering
    FEATURE_WINDOWS = [5, 10, 20, 50]
    TECHNICAL_INDICATORS = ['rsi', 'macd', 'bb', 'obv', 'stoch', 'atr', 'adx']
    
    # Volatility Analysis
    VOLATILITY_PERIODS = [5, 15, 30, 60]
    VOLATILITY_THRESHOLDS = {
        'low': 0.001,
        'medium': 0.002,
        'high': 0.004,
        'extreme': 0.008
    }
    
    # Execution Parameters
    CHECK_INTERVAL = 3          # Check every 3 seconds
    TRADE_COOLDOWN = 15         # 15 seconds between trades on same symbol
    MAX_TRADE_DURATION = 120    # Max 2 minutes per trade (ULTRA FAST)
    MIN_TRADE_DURATION = 10     # Minimum 10 seconds
    
    # Market Hours Optimization
    PEAK_HOURS = [8, 9, 10, 14, 15, 16, 20, 21, 22]  # High volatility hours
    PEAK_MULTIPLIER = 1.25
    
    # Telegram
    TELEGRAM_ENABLED = True
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # System
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/deep_learning_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'deep_learning_bot.db'
    
    # Performance Monitoring
    HOURLY_REPORT_INTERVAL = 3600  # 1 hour
    PERFORMANCE_TRACKING = True

# ============================================================================
# ADVANCED LOGGING WITH PERFORMANCE METRICS
# ============================================================================

class PerformanceLogger:
    """Advanced logging with performance metrics"""
    
    @staticmethod
    def setup_logging():
        """Setup advanced logging"""
        logger = logging.getLogger('DeepLearningBot')
        logger.setLevel(getattr(logging, UltimateConfig.LOG_LEVEL))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Color formatter
        class ColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',
                'INFO': '\033[32m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[41m',
                'TRADE': '\033[35m',     # Purple for trades
                'PROFIT': '\033[92m',    # Bright green for profits
                'LOSS': '\033[91m',      # Bright red for losses
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, '\033[37m')
                if hasattr(record, 'trade_type'):
                    if record.trade_type == 'profit':
                        log_color = self.COLORS['PROFIT']
                    elif record.trade_type == 'loss':
                        log_color = self.COLORS['LOSS']
                    elif record.trade_type == 'trade':
                        log_color = self.COLORS['TRADE']
                
                message = super().format(record)
                return f"{log_color}{message}{self.RESET}"
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, UltimateConfig.LOG_LEVEL))
        console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('deep_learning_bot.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        return logger

logger = PerformanceLogger.setup_logging()

# ============================================================================
# DEEP LEARNING PREDICTION ENGINE
# ============================================================================

class DeepLearningPredictor:
    """Deep Learning prediction engine with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = defaultdict(list)
        logger.info("Deep Learning Predictor initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for deep learning"""
        try:
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            df['close_open_pct'] = (df['close'] - df['open']) / df['open']
            
            # Multiple moving averages
            for window in [3, 5, 8, 13, 21, 34]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Price position relative to MAs - تأكد من وجود الأعمدة أولاً
            for window in [5, 10, 20]:
                sma_col = f'sma_{window}'
                ema_col = f'ema_{window}'
                
                # تحقق من وجود العمود قبل استخدامه
                if sma_col in df.columns:
                    # استبدل الصفر بـ NaN لتجنب القسمة على الصفر
                    df[f'price_{sma_col}_ratio'] = df['close'] / df[sma_col].replace(0, np.nan)
                else:
                    # إذا لم يكن موجوداً، أنشئه الآن
                    df[sma_col] = df['close'].rolling(window=window).mean()
                    df[f'price_{sma_col}_ratio'] = df['close'] / df[sma_col].replace(0, np.nan)
                
                if ema_col in df.columns:
                    df[f'price_{ema_col}_ratio'] = df['close'] / df[ema_col].replace(0, np.nan)
                else:
                    df[ema_col] = df['close'].ewm(span=window).mean()
                    df[f'price_{ema_col}_ratio'] = df['close'] / df[ema_col].replace(0, np.nan)
            
            # RSI variations
            for period in [7, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD variations
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands variations
            for window in [10, 20]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{window}'] = bb_middle - (bb_std * 2)
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / bb_middle
                df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['obv'] = self.calculate_obv(df)
            
            # Volatility features
            df['atr'] = self.calculate_atr(df)
            for period in [5, 10, 20]:
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            
            # Statistical features
            df['skewness_20'] = df['returns'].rolling(window=20).skew()
            df['kurtosis_20'] = df['returns'].rolling(window=20).kurt()
            
            # Momentum indicators
            df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
            df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            
            # Price patterns
            df['is_doji'] = self.detect_doji(df)
            df['is_marubozu'] = self.detect_marubozu(df)
            df['is_engulfing'] = self.detect_engulfing(df)
            df['is_hammer'] = self.detect_hammer(df)
            
            # Market structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            df['hhhc'] = (df['high'] > df['high'].shift(1)) & (df['close'] > df['close'].shift(1))
            df['lllc'] = (df['low'] < df['low'].shift(1)) & (df['close'] < df['close'].shift(1))
            
            # Support and Resistance
            df['near_resistance'] = (df['high'] / df['high'].rolling(20).max() > 0.98)
            df['near_support'] = (df['low'] / df['low'].rolling(20).min() < 1.02)
            
            # Time-based features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')
            
            # معالجة إضافية للقيم اللانهائية
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(df))
        obv[0] = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def detect_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Detect Doji candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        return body / total_range < threshold
    
    def detect_marubozu(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
        """Detect Marubozu candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        return (body / total_range > threshold) & (upper_shadow < body * 0.1) & (lower_shadow < body * 0.1)
    
    def detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect engulfing patterns"""
        # Bullish engulfing
        bullish = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        )
        
        # Bearish engulfing
        bearish = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        )
        
        return bullish | bearish
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        is_hammer = (
            (body < (df['high'] - df['low']) * 0.3) &
            (lower_shadow > body * 2) &
            (upper_shadow < body * 0.3)
        )
        
        return is_hammer
    
    def validate_and_clean_data(self, X, y):
        """Validate and clean data before training - FIXED"""
        try:
            # Convert to float arrays first
            X_clean = np.asarray(X, dtype=np.float64)
            y_clean = np.asarray(y, dtype=np.float64)
            
            # Use np.isfinite instead of np.isnan for better compatibility
            mask = np.all(np.isfinite(X_clean), axis=1) & np.isfinite(y_clean)
            
            X_clean = X_clean[mask]
            y_clean = y_clean[mask]
            
            if len(X_clean) < 30:
                return None, None
                
            return X_clean, y_clean
            
        except Exception as e:
            logger.warning(f"Data validation failed: {e}")
            return None, None
    
    def train_ensemble_model(self, symbol: str, features: pd.DataFrame, target: pd.Series):
        """Train ensemble of models for prediction - FIXED"""
        try:
            # Prepare data
            X = features.values
            y = target.values
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return
            
            # Validate and clean data using the fixed method
            X_clean, y_clean = self.validate_and_clean_data(X, y)
            
            if X_clean is None or y_clean is None:
                logger.warning(f"Cannot train model for {symbol}: insufficient or invalid data")
                return
            
            X = X_clean
            y = y_clean
            
            if len(X) < 30:
                logger.warning(f"Insufficient clean data for {symbol} ({len(X)} samples)")
                return
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train multiple models
            models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    subsample=0.8
                )
            }
            
            # Train each model
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_scaled, y)
                    trained_models[name] = model
                except Exception as e:
                    logger.warning(f"Failed to train {name} for {symbol}: {e}")
            
            if not trained_models:
                logger.warning(f"No models trained for {symbol}")
                return
            
            # Store models and scaler
            self.models[symbol] = trained_models
            self.scalers[symbol] = scaler
            
            logger.info(f"Ensemble model trained for {symbol} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
    
    def predict_with_confidence(self, symbol: str, features: pd.DataFrame) -> Dict:
        """Predict with ensemble confidence"""
        try:
            if symbol not in self.models:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'Model not trained'}
            
            # Prepare features
            X = features.values
            
            # Clean features before prediction
            X_clean = np.asarray(X, dtype=np.float64)
            mask = np.all(np.isfinite(X_clean), axis=1)
            
            if not np.any(mask):
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'Invalid feature data'}
            
            X_clean = X_clean[mask]
            
            if len(X_clean) == 0:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'No valid features'}
            
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X_clean)
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model_name, model in self.models[symbol].items():
                try:
                    pred = model.predict(X_scaled)[-1]  # Latest prediction
                    proba = model.predict_proba(X_scaled)[-1]
                    confidence = np.max(proba)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    continue
            
            if not predictions:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'All models failed'}
            
            # Ensemble voting
            unique_preds, counts = np.unique(predictions, return_counts=True)
            majority_pred = unique_preds[np.argmax(counts)]
            
            # Calculate ensemble confidence
            avg_confidence = np.mean(confidences)
            agreement = max(counts) / len(predictions)
            ensemble_confidence = avg_confidence * agreement
            
            # Map prediction to direction
            direction_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
            direction = direction_map.get(majority_pred, 'HOLD')
            
            # Only accept if confidence is high enough
            if ensemble_confidence < UltimateConfig.DL_MIN_CONFIDENCE:
                return {'direction': 'HOLD', 'confidence': ensemble_confidence, 'reason': 'Low confidence'}
            
            # Generate reason based on features
            reason = self.generate_prediction_reason(features.iloc[-1], direction)
            
            return {
                'direction': direction,
                'confidence': ensemble_confidence,
                'reason': reason,
                'model_agreement': agreement,
                'avg_confidence': avg_confidence,
                'predictions': predictions,
                'confidences': confidences
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return {'direction': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}
    
    def generate_prediction_reason(self, features: pd.Series, direction: str) -> str:
        """Generate human-readable reason for prediction"""
        reasons = []
        
        # RSI analysis
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if direction == 'BUY' and rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif direction == 'SELL' and rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.1f})")
        
        # MACD analysis
        if 'macd_hist' in features:
            macd_hist = features['macd_hist']
            if direction == 'BUY' and macd_hist > 0:
                reasons.append("MACD bullish")
            elif direction == 'SELL' and macd_hist < 0:
                reasons.append("MACD bearish")
        
        # Volume analysis
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if vol_ratio > 1.5:
                reasons.append(f"High volume (×{vol_ratio:.1f})")
        
        # Bollinger Bands
        if 'bb_position_20' in features:
            bb_pos = features['bb_position_20']
            if direction == 'BUY' and bb_pos < 0.2:
                reasons.append("Near BB lower band")
            elif direction == 'SELL' and bb_pos > 0.8:
                reasons.append("Near BB upper band")
        
        # Price momentum
        if 'momentum_5' in features:
            momentum = features['momentum_5']
            if direction == 'BUY' and momentum > 0:
                reasons.append(f"Positive momentum ({momentum:.4f})")
            elif direction == 'SELL' and momentum < 0:
                reasons.append(f"Negative momentum ({momentum:.4f})")
        
        # Candlestick patterns
        if features.get('is_hammer', False) and direction == 'BUY':
            reasons.append("Hammer pattern")
        if features.get('is_engulfing', False):
            reasons.append("Engulfing pattern")
        
        # Market structure
        if features.get('higher_high', False) and direction == 'BUY':
            reasons.append("Higher high formation")
        if features.get('lower_low', False) and direction == 'SELL':
            reasons.append("Lower low formation")
        
        return ', '.join(reasons) if reasons else "Technical indicators alignment"

# ============================================================================
# VOLATILITY ANALYZER
# ============================================================================

class VolatilityAnalyzer:
    """Analyze volatility for optimal trading"""
    
    def __init__(self):
        self.volatility_history = defaultdict(list)
        self.symbol_volatility = {}
        logger.info("Volatility Analyzer initialized")
    
    def analyze_volatility(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Analyze volatility at multiple timeframes"""
        try:
            returns = df['close'].pct_change().dropna()
            
            # Calculate volatility at different periods
            volatilities = {}
            for period in UltimateConfig.VOLATILITY_PERIODS:
                if len(returns) >= period:
                    volatilities[f'vol_{period}'] = returns.tail(period).std() * np.sqrt(365 * 24 * 60)  # Annualized
            
            # Calculate current volatility
            current_vol = returns.tail(5).std() * np.sqrt(365 * 24 * 60)
            
            # Determine volatility regime
            vol_regime = 'medium'
            if current_vol < UltimateConfig.VOLATILITY_THRESHOLDS['low']:
                vol_regime = 'low'
            elif current_vol < UltimateConfig.VOLATILITY_THRESHOLDS['medium']:
                vol_regime = 'medium'
            elif current_vol < UltimateConfig.VOLATILITY_THRESHOLDS['high']:
                vol_regime = 'high'
            else:
                vol_regime = 'extreme'
            
            # Calculate volatility ratio (current vs average)
            avg_vol = np.mean(list(volatilities.values())) if volatilities else current_vol
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            # Store history
            self.volatility_history[symbol].append(current_vol)
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol].pop(0)
            
            # Update symbol volatility
            self.symbol_volatility[symbol] = {
                'current': current_vol,
                'regime': vol_regime,
                'ratio': vol_ratio,
                'multiplier': UltimateConfig.VOLATILITY_MULTIPLIER[vol_regime]
            }
            
            return self.symbol_volatility[symbol]
            
        except Exception as e:
            logger.error(f"Error analyzing volatility for {symbol}: {e}")
            return {'current': 0, 'regime': 'medium', 'ratio': 1, 'multiplier': 1.0}
    
    def get_trading_params(self, symbol: str) -> Dict:
        """Get trading parameters based on volatility"""
        vol_data = self.symbol_volatility.get(symbol, {'regime': 'medium', 'multiplier': 1.0})
        multiplier = vol_data['multiplier']
        
        # Adjust parameters based on volatility
        take_profit = UltimateConfig.BASE_TAKE_PROFIT * multiplier
        stop_loss = UltimateConfig.BASE_STOP_LOSS * multiplier
        trailing_stop = UltimateConfig.TRAILING_STOP * multiplier
        
        # Adjust for time of day
        hour = datetime.now().hour
        if hour in UltimateConfig.PEAK_HOURS:
            take_profit *= UltimateConfig.PEAK_MULTIPLIER
            stop_loss *= UltimateConfig.PEAK_MULTIPLIER
        
        return {
            'take_profit_pct': min(take_profit, 0.005),  # Max 0.5%
            'stop_loss_pct': min(stop_loss, 0.003),      # Max 0.3%
            'trailing_stop_pct': trailing_stop,
            'volatility_regime': vol_data['regime'],
            'volatility_multiplier': multiplier,
            'is_peak_hour': hour in UltimateConfig.PEAK_HOURS
        }

# ============================================================================
# ADVANCED EXECUTION ENGINE
# ============================================================================

class AdvancedExecutionEngine:
    """Advanced execution engine with smart order routing"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.ticker_cache = {}
        self.orderbook_cache = {}
        self.cache_time = 1  # 1 second cache
        logger.info("Advanced Execution Engine initialized")
    
    def get_market_intelligence(self, symbol: str) -> Dict:
        """Get comprehensive market intelligence"""
        try:
            # Get ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Get order book
            orderbook = self.exchange.fetch_order_book(symbol, limit=10)
            
            # Get recent trades
            trades = self.exchange.fetch_trades(symbol, limit=10)
            
            # Calculate spread metrics
            bid = ticker['bid']
            ask = ticker['ask']
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            # Order book analysis
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:5]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:5]])
            orderbook_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # Trade flow analysis
            buy_volume = sum([t['amount'] for t in trades if t['side'] == 'buy'])
            sell_volume = sum([t['amount'] for t in trades if t['side'] == 'sell'])
            trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
            
            # Price levels
            support_levels = self.calculate_support_levels(orderbook['bids'])
            resistance_levels = self.calculate_resistance_levels(orderbook['asks'])
            
            return {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': ticker['last'],
                'spread_pct': spread_pct,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'orderbook_imbalance': orderbook_imbalance,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'trade_imbalance': trade_imbalance,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'volume': ticker['quoteVolume'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market intelligence for {symbol}: {e}")
            return None
    
    def calculate_support_levels(self, bids: List) -> List:
        """Calculate support levels from order book"""
        levels = []
        cumulative_volume = 0
        
        for price, volume in bids[:5]:
            cumulative_volume += volume
            levels.append({
                'price': price,
                'volume': volume,
                'cumulative_volume': cumulative_volume,
                'strength': cumulative_volume / sum([bid[1] for bid in bids[:5]])
            })
        
        return levels
    
    def calculate_resistance_levels(self, asks: List) -> List:
        """Calculate resistance levels from order book"""
        levels = []
        cumulative_volume = 0
        
        for price, volume in asks[:5]:
            cumulative_volume += volume
            levels.append({
                'price': price,
                'volume': volume,
                'cumulative_volume': cumulative_volume,
                'strength': cumulative_volume / sum([ask[1] for ask in asks[:5]])
            })
        
        return levels
    
    def calculate_optimal_execution(self, symbol: str, direction: str, size: float) -> Dict:
        """Calculate optimal execution parameters"""
        try:
            market_data = self.get_market_intelligence(symbol)
            if not market_data:
                return None
                        # Calculate optimal price based on direction and order book
            if direction == 'BUY':
                # For buys, look at asks and try to get best price
                optimal_price = market_data['ask']
                price_levels = market_data['resistance_levels']
                
                # Try to get slightly better price if order book allows
                if len(price_levels) > 0 and price_levels[0]['strength'] < 0.3:
                    optimal_price = min(optimal_price * 0.9995, price_levels[0]['price'] * 0.999)
                
            else:  # SELL
                # For sells, look at bids and try to get best price
                optimal_price = market_data['bid']
                price_levels = market_data['support_levels']
                
                # Try to get slightly better price if order book allows
                if len(price_levels) > 0 and price_levels[0]['strength'] < 0.3:
                    optimal_price = max(optimal_price * 1.0005, price_levels[0]['price'] * 1.001)
            
            # Adjust for spread
            spread_impact = market_data['spread_pct'] / 2
            if direction == 'BUY':
                optimal_price *= (1 + spread_impact)
            else:
                optimal_price *= (1 - spread_impact)
            
            # Consider market impact
            market_impact = self.estimate_market_impact(size, market_data)
            
            # Final price with market impact
            if direction == 'BUY':
                execution_price = optimal_price * (1 + market_impact)
            else:
                execution_price = optimal_price * (1 - market_impact)
            
            # Ensure price is within reasonable bounds
            if direction == 'BUY':
                execution_price = min(execution_price, market_data['ask'] * 1.001)
            else:
                execution_price = max(execution_price, market_data['bid'] * 0.999)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'size': size,
                'execution_price': execution_price,
                'market_price': market_data['last'],
                'spread_pct': market_data['spread_pct'],
                'orderbook_imbalance': market_data['orderbook_imbalance'],
                'trade_imbalance': market_data['trade_imbalance'],
                'estimated_slippage': market_impact,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating optimal execution for {symbol}: {e}")
            return None
    
    def estimate_market_impact(self, size: float, market_data: Dict) -> float:
        """Estimate market impact of a trade"""
        try:
            # Calculate order book depth
            total_depth = market_data['bid_volume'] + market_data['ask_volume']
            
            if total_depth == 0:
                return 0.001  # Default 0.1% impact
            
            # Size relative to depth
            size_ratio = size / total_depth
            
            # Estimate impact based on size ratio (exponential decay)
            impact = 0.0005 * (1 - np.exp(-10 * size_ratio))
            
            # Adjust based on order book imbalance
            imbalance = abs(market_data['orderbook_imbalance'])
            impact *= (1 + imbalance * 2)
            
            # Cap the impact
            return min(impact, 0.005)  # Max 0.5% impact
            
        except Exception as e:
            logger.warning(f"Error estimating market impact: {e}")
            return 0.001
    
    def execute_trade(self, trade_signal: Dict, capital_allocated: float) -> Dict:
        """Execute a trade with optimal parameters"""
        try:
            symbol = trade_signal['symbol']
            direction = trade_signal['direction']
            
            # Get optimal execution parameters
            execution_params = self.calculate_optimal_execution(
                symbol, direction, capital_allocated
            )
            
            if not execution_params:
                logger.error(f"Cannot calculate execution parameters for {symbol}")
                return None
            
            # Get current balance (simulated)
            # In production, you would fetch actual balance
            available_balance = self.get_available_balance(symbol)
            
            # Check if we have enough balance
            required_amount = capital_allocated / execution_params['execution_price']
            
            if direction == 'BUY' and required_amount * execution_params['execution_price'] > available_balance:
                logger.warning(f"Insufficient balance for {symbol}: needed {required_amount:.4f}, have {available_balance:.2f}")
                return None
            
            # Simulate order execution (in production, use exchange API)
            order_result = self.simulate_order_execution(
                symbol=symbol,
                side='buy' if direction == 'BUY' else 'sell',
                amount=required_amount,
                price=execution_params['execution_price']
            )
            
            if not order_result:
                logger.error(f"Order execution failed for {symbol}")
                return None
            
            # Create trade record
            trade_record = {
                'trade_id': f"{symbol.replace('/', '')}_{int(time.time())}",
                'symbol': symbol,
                'direction': direction,
                'entry_price': execution_params['execution_price'],
                'size': capital_allocated,
                'quantity': required_amount,
                'stop_loss': trade_signal['stop_loss'],
                'take_profit': trade_signal['take_profit'],
                'trailing_stop': trade_signal['trailing_stop'],
                'entry_time': datetime.now(),
                'status': 'OPEN',
                'current_price': execution_params['market_price'],
                'pnl_pct': 0.0,
                'pnl_usd': 0.0,
                'execution_quality': {
                    'slippage': (execution_params['execution_price'] - execution_params['market_price']) / execution_params['market_price'],
                    'spread_pct': execution_params['spread_pct'],
                    'orderbook_imbalance': execution_params['orderbook_imbalance']
                },
                'prediction_data': trade_signal.get('prediction_data', {}),
                'volatility_data': trade_signal.get('volatility_data', {})
            }
            
            logger.info(f"Trade executed: {direction} {symbol} at ${trade_record['entry_price']:.4f} "
                       f"Size: ${capital_allocated:.2f} TP: {trade_record['take_profit']*100:.2f}% "
                       f"SL: {trade_record['stop_loss']*100:.2f}%")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return None
    
    def get_available_balance(self, symbol: str) -> float:
        """Get available balance for trading"""
        # In production, fetch from exchange
        # For simulation, return a fixed amount
        return UltimateConfig.INITIAL_CAPITAL * 0.8  # 80% of initial capital
    
    def simulate_order_execution(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """Simulate order execution (replace with actual exchange API)"""
        try:
            # Simulate some execution delay and randomness
            time.sleep(0.1)  # Small delay
            
            # Add some random slippage
            slippage = np.random.uniform(-0.0001, 0.0001)
            executed_price = price * (1 + slippage)
            
            # Simulate partial fills
            fill_ratio = np.random.uniform(0.95, 1.0)  # 95-100% fill
            executed_amount = amount * fill_ratio
            
            return {
                'symbol': symbol,
                'side': side,
                'requested_amount': amount,
                'requested_price': price,
                'executed_amount': executed_amount,
                'executed_price': executed_price,
                'fill_ratio': fill_ratio,
                'slippage': slippage,
                'status': 'filled',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error simulating order: {e}")
            return None

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """Advanced portfolio management with risk control"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_trades = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.hourly_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = UltimateConfig.MAX_DAILY_LOSS * initial_capital
        self.hourly_target = UltimateConfig.HOURLY_TARGET * initial_capital
        self.daily_target = UltimateConfig.DAILY_TARGET * initial_capital
        self.last_hour_reset = datetime.now()
        self.last_day_reset = datetime.now().date()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0
        }
        
        logger.info(f"Portfolio Manager initialized with ${initial_capital:.2f}")
    
    def can_open_trade(self, symbol: str, trade_size: float) -> Tuple[bool, str]:
        """Check if we can open a new trade"""
        try:
            # Check max open trades
            if len(self.open_trades) >= UltimateConfig.MAX_OPEN_TRADES:
                return False, f"Max open trades reached ({UltimateConfig.MAX_OPEN_TRADES})"
            
            # Check same symbol limit
            same_symbol_trades = sum(1 for trade in self.open_trades.values() 
                                   if trade['symbol'] == symbol)
            if same_symbol_trades >= UltimateConfig.MAX_SAME_SYMBOL:
                return False, f"Max trades for symbol reached ({UltimateConfig.MAX_SAME_SYMBOL})"
            
            # Check position size limit
            position_size_pct = trade_size / self.current_capital
            if position_size_pct > UltimateConfig.MAX_POSITION_SIZE:
                return False, f"Position size {position_size_pct:.1%} exceeds max {UltimateConfig.MAX_POSITION_SIZE:.1%}"
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
            
            # Check hourly target
            current_hour = datetime.now().hour
            if current_hour != self.last_hour_reset.hour:
                self.hourly_pnl = 0.0
                self.last_hour_reset = datetime.now()
            
            if self.hourly_pnl >= self.hourly_target:
                return False, f"Hourly target reached: ${self.hourly_pnl:.2f}"
            
            # Check daily target
            if self.daily_pnl >= self.daily_target:
                return False, f"Daily target reached: ${self.daily_pnl:.2f}"
            
            # Check if symbol is in cooldown
            if self.is_symbol_in_cooldown(symbol):
                return False, "Symbol in cooldown"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Error checking trade eligibility: {e}")
            return False, f"Error: {str(e)}"
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        for trade_id, trade in self.open_trades.items():
            if trade['symbol'] == symbol:
                trade_age = (datetime.now() - trade['entry_time']).total_seconds()
                if trade_age < UltimateConfig.TRADE_COOLDOWN:
                    return True
        
        # Check recent trade history
        recent_trades = [t for t in self.trade_history 
                        if t['symbol'] == symbol and 
                        (datetime.now() - t['exit_time']).total_seconds() < UltimateConfig.TRADE_COOLDOWN]
        
        return len(recent_trades) > 0
    
    def calculate_position_size(self, symbol: str, volatility_data: Dict) -> float:
        """Calculate position size based on risk and volatility"""
        try:
            # Base risk amount
            base_risk = self.current_capital * UltimateConfig.RISK_PER_TRADE
            
            # Adjust for volatility
            vol_multiplier = volatility_data.get('multiplier', 1.0)
            adjusted_risk = base_risk * vol_multiplier
            
            # Adjust for portfolio concentration
            open_symbols = len(set(trade['symbol'] for trade in self.open_trades.values()))
            concentration_factor = 1.0 / max(open_symbols, 1)
            adjusted_risk *= concentration_factor
            
            # Ensure minimum and maximum sizes
            min_position = self.current_capital * 0.01  # 1% minimum
            max_position = self.current_capital * UltimateConfig.MAX_POSITION_SIZE
            
            position_size = max(min_position, min(adjusted_risk, max_position))
            
            # Round to 2 decimal places
            return round(position_size, 2)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.current_capital * UltimateConfig.RISK_PER_TRADE
    
    def add_trade(self, trade_record: Dict):
        """Add a new trade to portfolio"""
        try:
            self.open_trades[trade_record['trade_id']] = trade_record
            self.daily_trades += 1
            
            # Update capital (subtract trade size)
            self.current_capital -= trade_record['size']
            
            logger.info(f"Trade added: {trade_record['symbol']} | "
                       f"Open trades: {len(self.open_trades)} | "
                       f"Capital: ${self.current_capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
    
    def update_trades(self, market_prices: Dict):
        """Update open trades with current prices"""
        try:
            total_pnl = 0.0
            trades_to_close = []
            
            for trade_id, trade in self.open_trades.items():
                symbol = trade['symbol']
                
                if symbol not in market_prices:
                    continue
                
                current_price = market_prices[symbol]
                trade['current_price'] = current_price
                
                # Calculate P&L
                if trade['direction'] == 'BUY':
                    pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
                else:  # SELL
                    pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                
                # Apply fees
                pnl_pct -= UltimateConfig.TAKER_FEE * 2  # Entry and exit
                
                pnl_usd = pnl_pct * trade['size']
                
                trade['pnl_pct'] = pnl_pct
                trade['pnl_usd'] = pnl_usd
                
                total_pnl += pnl_usd
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(trade)
                if exit_reason:
                    trades_to_close.append((trade_id, exit_reason))
            
            # Close trades that hit exit conditions
            for trade_id, exit_reason in trades_to_close:
                self.close_trade(trade_id, exit_reason)
            
            # Update performance metrics
            self.update_performance_metrics()
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error updating trades: {e}")
            return 0.0
    
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        """Check if trade should be closed"""
        try:
            pnl_pct = trade['pnl_pct']
            trade_age = (datetime.now() - trade['entry_time']).total_seconds()
            
            # Check take profit
            if pnl_pct >= trade['take_profit']:
                return f"Take profit hit: {pnl_pct:.2%}"
            
            # Check stop loss
            if pnl_pct <= -trade['stop_loss']:
                return f"Stop loss hit: {pnl_pct:.2%}"
            
            # Check trailing stop
            if 'trailing_stop' in trade and trade['trailing_stop'] > 0:
                # Calculate highest price since entry
                if 'highest_price' not in trade:
                    trade['highest_price'] = trade['entry_price']
                
                if trade['direction'] == 'BUY':
                    trade['highest_price'] = max(trade['highest_price'], trade['current_price'])
                    trailing_level = trade['highest_price'] * (1 - trade['trailing_stop'])
                    if trade['current_price'] <= trailing_level:
                        return f"Trailing stop hit: {pnl_pct:.2%}"
                else:
                    trade['lowest_price'] = min(trade.get('lowest_price', trade['entry_price']), trade['current_price'])
                    trailing_level = trade['lowest_price'] * (1 + trade['trailing_stop'])
                    if trade['current_price'] >= trailing_level:
                        return f"Trailing stop hit: {pnl_pct:.2%}"
            
            # Check max trade duration
            if trade_age > UltimateConfig.MAX_TRADE_DURATION:
                return f"Max duration reached: {trade_age:.0f}s"
            
            # Check min trade duration
            if trade_age < UltimateConfig.MIN_TRADE_DURATION:
                return None
            
            # Check if prediction changed
            if 'prediction_data' in trade:
                current_time = datetime.now()
                prediction_age = (current_time - trade['entry_time']).total_seconds()
                if prediction_age > 30:  # Re-evaluate every 30 seconds
                    # In production, get new prediction
                    # For now, keep trade open
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def close_trade(self, trade_id: str, exit_reason: str):
        """Close a trade"""
        try:
            if trade_id not in self.open_trades:
                return
            
            trade = self.open_trades[trade_id]
            
            # Calculate final P&L
            exit_price = trade['current_price']
            if trade['direction'] == 'BUY':
                pnl_pct = (exit_price - trade['entry_price']) / trade['entry_price']
            else:
                pnl_pct = (trade['entry_price'] - exit_price) / trade['entry_price']
            
            # Apply fees
            pnl_pct -= UltimateConfig.TAKER_FEE * 2
            pnl_usd = pnl_pct * trade['size']
            
            # Update trade record
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now()
            trade['exit_reason'] = exit_reason
            trade['pnl_pct'] = pnl_pct
            trade['pnl_usd'] = pnl_usd
            trade['status'] = 'CLOSED'
            
            # Update capital
            self.current_capital += trade['size'] + pnl_usd
            
            # Update P&L tracking
            self.daily_pnl += pnl_usd
            self.hourly_pnl += pnl_usd
            
            # Move to history
            self.trade_history.append(trade)
            del self.open_trades[trade_id]
            
            # Log trade result
            log_method = logger.info if pnl_usd >= 0 else logger.warning
            trade_type = 'PROFIT' if pnl_usd >= 0 else 'LOSS'
            
            log_method(f"Trade closed: {trade['symbol']} | "
                      f"Direction: {trade['direction']} | "
                      f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2%}) | "
                      f"Reason: {exit_reason}", extra={'trade_type': trade_type.lower()})
            
            # Update performance metrics
            self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.trade_history:
                return
            
            # Basic metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t['pnl_usd'] > 0)
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(t['pnl_usd'] for t in self.trade_history)
            winning_pnl = sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] > 0)
            losing_pnl = sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] < 0)
            
            # Update metrics
            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'largest_win': max((t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] > 0), default=0),
                'largest_loss': min((t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] < 0), default=0),
                'avg_win': winning_pnl / winning_trades if winning_trades > 0 else 0,
                'avg_loss': losing_pnl / losing_trades if losing_trades > 0 else 0,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'profit_factor': abs(winning_pnl / losing_pnl) if losing_pnl < 0 else float('inf'),
                'current_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            # Calculate open positions P&L
            open_pnl = sum(trade['pnl_usd'] for trade in self.open_trades.values())
            open_exposure = sum(trade['size'] for trade in self.open_trades.values())
            
            # Calculate diversification
            symbols = set(trade['symbol'] for trade in self.open_trades.values())
            diversification = len(symbols) / len(self.open_trades) if self.open_trades else 0
            
            # Calculate daily statistics
            today = datetime.now().date()
            today_trades = [t for t in self.trade_history 
                           if t['exit_time'].date() == today]
            today_pnl = sum(t['pnl_usd'] for t in today_trades)
            
            return {
                'timestamp': datetime.now(),
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'open_trades': len(self.open_trades),
                'open_exposure': open_exposure,
                'open_pnl': open_pnl,
                'daily_pnl': self.daily_pnl,
                'hourly_pnl': self.hourly_pnl,
                'daily_trades': self.daily_trades,
                'today_pnl': today_pnl,
                'diversification': diversification,
                'available_capital': self.current_capital - open_exposure,
                'performance': self.performance_metrics,
                'trade_history_summary': {
                    'total': len(self.trade_history),
                    'today': len(today_trades),
                    'win_rate': self.performance_metrics['win_rate'],
                    'avg_win': self.performance_metrics['avg_win'],
                    'avg_loss': self.performance_metrics['avg_loss']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}

# ============================================================================
# DATA COLLECTOR
# ============================================================================

class DataCollector:
    """Collect and process market data"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        self.data_cache = {}
        self.last_update = {}
        logger.info("Data Collector initialized")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Check cache
            if cache_key in self.data_cache:
                cache_age = time.time() - self.last_update.get(cache_key, 0)
                if cache_age < 10:  # 10 second cache
                    return self.data_cache[cache_key]
            
            # Fetch new data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Create DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache data
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = time.time()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        timeframes = {}
        
        for tf in UltimateConfig.TIMEFRAMES:
            try:
                df = self.fetch_ohlcv(symbol, tf, limit=UltimateConfig.DL_LOOKBACK)
                if not df.empty:
                    timeframes[tf] = df
            except Exception as e:
                logger.warning(f"Error fetching {tf} data for {symbol}: {e}")
        
        return timeframes

# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class DeepLearningScalpingBot:
    """Main trading bot class"""
    
    def __init__(self):
        self.config = UltimateConfig()
        self.data_collector = DataCollector()
        self.predictor = DeepLearningPredictor()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.execution_engine = AdvancedExecutionEngine()
        self.portfolio_manager = PortfolioManager(UltimateConfig.INITIAL_CAPITAL)
        
        # Training counter
        self.training_counter = 0
        
        # Telegram bot (if enabled)
        self.telegram_bot = None
        if UltimateConfig.TELEGRAM_ENABLED and UltimateConfig.TELEGRAM_BOT_TOKEN:
            self.setup_telegram()
        
        logger.info("Deep Learning Scalping Bot initialized")
        logger.info(f"Monitoring {len(self.config.SYMBOLS)} symbols")
        logger.info(f"Initial capital: ${UltimateConfig.INITIAL_CAPITAL:.2f}")
    
    def setup_telegram(self):
        """Setup Telegram bot for notifications"""
        try:
            import telegram
            self.telegram_bot = telegram.Bot(token=UltimateConfig.TELEGRAM_BOT_TOKEN)
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning("Telegram module not installed. Install with: pip install python-telegram-bot")
        except Exception as e:
            logger.error(f"Error setting up Telegram: {e}")
    
    def send_telegram_message(self, message: str):
        """Send message via Telegram"""
        if not self.telegram_bot or not UltimateConfig.TELEGRAM_CHAT_ID:
            return
        
        try:
            self.telegram_bot.send_message(
                chat_id=UltimateConfig.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunities"""
        try:
            # Fetch data for multiple timeframes
            timeframes = self.data_collector.fetch_multiple_timeframes(symbol)
            
            if not timeframes:
                return None
            
            # Get primary timeframe data
            primary_data = timeframes.get(UltimateConfig.PRIMARY_TIMEFRAME)
            if primary_data is None or len(primary_data) < 50:
                return None
            
            # Analyze volatility
            volatility_data = self.volatility_analyzer.analyze_volatility(primary_data, symbol)
            
            # Create features for deep learning
            features = self.predictor.create_features(primary_data.copy())
            
            # Prepare target (next candle direction)
            if len(features) < 2:
                return None
            
            # Create binary target: 1 if next close > current close, else 0
            features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
            features.dropna(inplace=True)
            
            if len(features) < 30:
                return None
            
            # Train model periodically
            self.training_counter += 1
            if self.training_counter % UltimateConfig.DL_TRAIN_INTERVAL == 0:
                self.predictor.train_ensemble_model(
                    symbol,
                    features.iloc[:-1].drop('target', axis=1),
                    features['target'].iloc[:-1]
                )
            
            # Get prediction
            latest_features = features.iloc[-1:].drop('target', axis=1)
            prediction = self.predictor.predict_with_confidence(symbol, latest_features)
            
            if prediction['direction'] == 'HOLD':
                return None
            
            # Get trading parameters based on volatility
            trading_params = self.volatility_analyzer.get_trading_params(symbol)
            
            # Create trade signal
            trade_signal = {
                'symbol': symbol,
                'direction': prediction['direction'],
                'confidence': prediction['confidence'],
                'reason': prediction['reason'],
                'current_price': primary_data['close'].iloc[-1],
                'take_profit': trading_params['take_profit_pct'],
                'stop_loss': trading_params['stop_loss_pct'],
                'trailing_stop': trading_params['trailing_stop_pct'],
                'volatility_regime': trading_params['volatility_regime'],
                'prediction_data': prediction,
                'volatility_data': volatility_data,
                'timestamp': datetime.now()
            }
            
            return trade_signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def process_trade_signal(self, trade_signal: Dict):
        """Process a trade signal"""
        try:
            symbol = trade_signal['symbol']
            
            # Check if we can open a trade
            position_size = self.portfolio_manager.calculate_position_size(
                symbol, trade_signal['volatility_data']
            )
            
            can_trade, reason = self.portfolio_manager.can_open_trade(symbol, position_size)
            
            if not can_trade:
                logger.debug(f"Cannot trade {symbol}: {reason}")
                return
            
            # Execute trade
            trade_record = self.execution_engine.execute_trade(trade_signal, position_size)
            
            if trade_record:
                # Add to portfolio
                self.portfolio_manager.add_trade(trade_record)
                
                # Send notification
                if self.telegram_bot:
                    message = (
                        f"🎯 <b>NEW TRADE</b>\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {trade_signal['direction']}\n"
                        f"Entry: ${trade_record['entry_price']:.4f}\n"
                        f"Size: ${position_size:.2f}\n"
                        f"TP: {trade_signal['take_profit']*100:.2f}%\n"
                        f"SL: {trade_signal['stop_loss']*100:.2f}%\n"
                        f"Confidence: {trade_signal['confidence']:.1%}\n"
                        f"Reason: {trade_signal['reason']}"
                    )
                    self.send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error processing trade signal for {symbol}: {e}")
    
    def update_market_prices(self) -> Dict:
        """Update market prices for all symbols"""
        market_prices = {}
        
        for symbol in self.config.SYMBOLS:
            try:
                ticker = self.execution_engine.exchange.fetch_ticker(symbol)
                market_prices[symbol] = ticker['last']
            except Exception as e:
                logger.warning(f"Error fetching price for {symbol}: {e}")
        
        return market_prices
    
    def generate_hourly_report(self):
        """Generate hourly performance report"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            report = (
                f"\n{'='*60}\n"
                f"HOURLY PERFORMANCE REPORT\n"
                f"{'='*60}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Capital: ${portfolio_summary.get('current_capital', 0):.2f}\n"
                f"Total Return: {portfolio_summary.get('total_return', 0)*100:.2f}%\n"
                f"Open Trades: {portfolio_summary.get('open_trades', 0)}\n"
                f"Daily P&L: ${portfolio_summary.get('daily_pnl', 0):+.2f}\n"
                f"Hourly P&L: ${portfolio_summary.get('hourly_pnl', 0):+.2f}\n"
                f"Today's Trades: {portfolio_summary.get('daily_trades', 0)}\n"
                f"Win Rate: {portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%\n"
                f"Total Trades: {portfolio_summary.get('performance', {}).get('total_trades', 0)}\n"
                f"{'='*60}"
            )
            
            logger.info(report)
            
            # Send Telegram report
            if self.telegram_bot:
                telegram_report = (
                    f"⏰ <b>Hourly Report</b>\n"
                    f"Capital: <code>${portfolio_summary.get('current_capital', 0):.2f}</code>\n"
                    f"Return: <code>{portfolio_summary.get('total_return', 0)*100:+.2f}%</code>\n"
                    f"Daily P&L: <code>${portfolio_summary.get('daily_pnl', 0):+.2f}</code>\n"
                    f"Open Trades: <code>{portfolio_summary.get('open_trades', 0)}</code>\n"
                    f"Win Rate: <code>{portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%</code>"
                )
                self.send_telegram_message(telegram_report)
                
        except Exception as e:
            logger.error(f"Error generating hourly report: {e}")
    
    def run(self):
        """Main bot execution loop"""
        logger.info("Starting Deep Learning Scalping Bot...")
        
        # Send startup notification
        if self.telegram_bot:
            self.send_telegram_message(
                f"🤖 <b>Deep Learning Scalping Bot Started</b>\n"
                f"Capital: ${UltimateConfig.INITIAL_CAPITAL:.2f}\n"
                f"Symbols: {len(self.config.SYMBOLS)}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
        
        last_hourly_report = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Generate hourly report
                if current_time - last_hourly_report >= UltimateConfig.HOURLY_REPORT_INTERVAL:
                    self.generate_hourly_report()
                    last_hourly_report = current_time
                
                # Update market prices and manage existing trades
                market_prices = self.update_market_prices()
                self.portfolio_manager.update_trades(market_prices)
                
                # Analyze each symbol for trading opportunities
                for symbol in self.config.SYMBOLS:
                    try:
                        # Analyze symbol
                        trade_signal = self.analyze_symbol(symbol)
                        
                        if trade_signal and trade_signal['confidence'] >= UltimateConfig.DL_MIN_CONFIDENCE:
                            # Process trade signal
                            self.process_trade_signal(trade_signal)
                        
                        # Small delay between symbols
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Sleep before next iteration
                time.sleep(UltimateConfig.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            # Generate final report
            self.generate_hourly_report()
            
            # Send shutdown notification
            if self.telegram_bot:
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                self.send_telegram_message(
                    f"🛑 <b>Bot Stopped</b>\n"
                    f"Final Capital: <code>${portfolio_summary.get('current_capital', 0):.2f}</code>\n"
                    f"Total Return: <code>{portfolio_summary.get('total_return', 0)*100:+.2f}%</code>\n"
                    f"Total Trades: <code>{portfolio_summary.get('performance', {}).get('total_trades', 0)}</code>\n"
                    f"Win Rate: <code>{portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%</code>"
                )

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create bot instance
    bot = DeepLearningScalpingBot()
    
    # Run bot
    bot.run()
            # Calculate optimal price based on direction and order
