# deep_lea# d# # deep_learning_scalping_bot_fixed.py
"""
Deep Learning Scalping Bot - FIXED VERSION
With Telegram notifications and improved trading logic
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
# ULTIMATE CONFIGURATION - IMPROVED VERSION
# ============================================================================

class UltimateConfig:
    """Ultimate configuration for deep learning scalping - IMPROVED"""
    
    # Exchange & Symbols - REMOVED HIGH-RISK COINS
    EXCHANGE = 'binance'
    
    # Top 15 most suitable cryptocurrencies for scalping
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
        'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT',
        'LTC/USDT', 'AVAX/USDT', 'ATOM/USDT', 'UNI/USDT',
        'ALGO/USDT', 'XLM/USDT', 'VET/USDT'
    ]
    
    # Timeframes for multi-timeframe analysis
    TIMEFRAMES = ['1m', '5m', '15m']
    PRIMARY_TIMEFRAME = '1m'
    
    # Capital Management (Starting with $100) - IMPROVED
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.015      # 1.5% risk per trade (increased)
    MAX_DAILY_LOSS = 0.02       # 2% daily loss limit
    HOURLY_TARGET = 0.01        # 1% hourly profit target
    DAILY_TARGET = 0.04         # 4% daily target
    
    # Position Sizing - IMPROVED
    MAX_POSITION_SIZE = 0.20    # 20% max per symbol
    MIN_POSITION_SIZE = 0.02    # 2% min per symbol
    MAX_OPEN_TRADES = 6         # Reduced to 6
    MAX_SAME_SYMBOL = 1         # Max 1 trade per symbol
    
    # Scalping Parameters - IMPROVED
    BASE_TAKE_PROFIT = 0.0050   # 0.50% base take profit (increased)
    BASE_STOP_LOSS = 0.0030     # 0.30% base stop loss (increased)
    TRAILING_STOP = 0.0015      # 0.15% trailing stop
    
    # Dynamic Adjustments based on volatility
    VOLATILITY_MULTIPLIER = {
        'low': 0.8,
        'medium': 1.0,
        'high': 1.2,
        'extreme': 1.0  # Reduced for extreme volatility
    }
    
    # Fees & Spread
    MAKER_FEE = 0.00075
    TAKER_FEE = 0.0010
    MIN_SPREAD = 0.00005
    MAX_SPREAD = 0.0003
    
    # Deep Learning Parameters - IMPROVED
    DL_LOOKBACK = 150           # Increased to 150
    DL_TRAIN_INTERVAL = 100     # Retrain every 100 cycles
    DL_MIN_CONFIDENCE = 0.75    # 75% minimum confidence (increased)
    DL_ENSEMBLE_SIZE = 3
    
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
    
    # Execution Parameters - IMPROVED
    CHECK_INTERVAL = 5          # Increased to 5 seconds
    TRADE_COOLDOWN = 30         # 30 seconds between trades on same symbol
    MAX_TRADE_DURATION = 300    # 5 minutes per trade (increased)
    MIN_TRADE_DURATION = 30     # Minimum 30 seconds
    
    # Market Hours Optimization
    PEAK_HOURS = [8, 9, 10, 14, 15, 16, 20, 21, 22]
    PEAK_MULTIPLIER = 1.25
    
    # Telegram - FIXED
    TELEGRAM_ENABLED = True
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # System
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/deep_learning_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'deep_learning_bot.db'
    
    # Performance Monitoring
    HOURLY_REPORT_INTERVAL = 3600
    PERFORMANCE_TRACKING = True

# ============================================================================
# TELEGRAM MANAGER - FIXED VERSION
# ============================================================================

class TelegramManager:
    """Fixed Telegram notification manager with retry logic"""
    
    def __init__(self):
        self.bot = None
        self.chat_id = None
        self.initialized = False
        self.initialize_bot()
    
    def initialize_bot(self):
        """Initialize Telegram bot with proper error handling"""
        try:
            # Get credentials from environment
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            logger.info(f"Telegram token exists: {bool(token)}")
            logger.info(f"Telegram chat_id exists: {bool(chat_id)}")
            
            if not token or not chat_id:
                logger.warning("Telegram credentials not found. Disabling Telegram notifications.")
                self.initialized = False
                return
            
            # Import telegram inside try block
            try:
                from telegram import Bot
                from telegram.error import TelegramError
            except ImportError:
                logger.error("python-telegram-bot not installed. Install with: pip install python-telegram-bot")
                self.initialized = False
                return
            
            # Initialize bot
            self.bot = Bot(token=token)
            self.chat_id = str(chat_id)  # Ensure it's string
            
            # Test connection
            bot_info = self.bot.get_me()
            logger.info(f"Telegram bot initialized: @{bot_info.username}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            self.initialized = False
    
    def send_message(self, message: str, parse_mode: str = 'HTML', retry_count: int = 3):
        """Send message with retry logic"""
        if not self.initialized or not self.bot:
            logger.debug("Telegram not initialized, skipping message")
            return False
        
        for attempt in range(retry_count):
            try:
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )
                return True
            except Exception as e:
                logger.warning(f"Telegram send failed (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to send Telegram message after {retry_count} attempts")
                    return False
        
        return False
    
    def send_trade_notification(self, trade_data: Dict, is_open: bool = True):
        """Send trade notification"""
        if not self.initialized:
            return
        
        try:
            if is_open:
                emoji = "🟢" if trade_data['direction'] == 'BUY' else "🔴"
                title = "NEW TRADE"
            else:
                emoji = "💰" if trade_data.get('pnl_usd', 0) >= 0 else "💸"
                title = "TRADE CLOSED"
            
            message = (
                f"{emoji} <b>{title}</b>\n"
                f"• Symbol: {trade_data['symbol']}\n"
                f"• Direction: {trade_data['direction']}\n"
                f"• Entry: ${trade_data.get('entry_price', 0):.4f}\n"
            )
            
            if not is_open:
                message += (
                    f"• Exit: ${trade_data.get('exit_price', 0):.4f}\n"
                    f"• P&L: ${trade_data.get('pnl_usd', 0):+.2f}\n"
                    f"• P&L%: {trade_data.get('pnl_pct', 0):+.2%}\n"
                    f"• Reason: {trade_data.get('exit_reason', 'N/A')}\n"
                )
            else:
                message += (
                    f"• Size: ${trade_data.get('size', 0):.2f}\n"
                    f"• TP: {trade_data.get('take_profit', 0)*100:.2f}%\n"
                    f"• SL: {trade_data.get('stop_loss', 0)*100:.2f}%\n"
                    f"• Confidence: {trade_data.get('confidence', 0):.1%}\n"
                )
            
            self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {str(e)}")
    
    def send_performance_report(self, portfolio_summary: Dict):
        """Send performance report"""
        if not self.initialized:
            return
        
        try:
            message = (
                f"📊 <b>Performance Report</b>\n"
                f"• Capital: ${portfolio_summary.get('current_capital', 0):.2f}\n"
                f"• Total Return: {portfolio_summary.get('total_return', 0)*100:+.2f}%\n"
                f"• Daily P&L: ${portfolio_summary.get('daily_pnl', 0):+.2f}\n"
                f"• Open Trades: {portfolio_summary.get('open_trades', 0)}\n"
                f"• Win Rate: {portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%\n"
                f"• Total Trades: {portfolio_summary.get('performance', {}).get('total_trades', 0)}\n"
            )
            
            self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending performance report: {str(e)}")

# ============================================================================
# LOGGING - IMPROVED
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
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, '\033[37m')
                message = super().format(record)
                return f"{log_color}{message}{self.RESET}"
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, UltimateConfig.LOG_LEVEL))
        console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('trading_bot.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        return logger

logger = PerformanceLogger.setup_logging()

# ============================================================================
# SMART TRADE FILTER
# ============================================================================

class SmartTradeFilter:
    """Filter trades based on market conditions"""
    
    @staticmethod
    def filter_symbol(symbol: str, market_data: Dict) -> bool:
        """Filter symbols based on market conditions"""
        
        # High-risk symbols to avoid
        high_risk_symbols = ['SHIB/USDT', 'DOGE/USDT', 'TRX/USDT']
        if symbol in high_risk_symbols:
            logger.debug(f"Filtered {symbol}: High-risk symbol")
            return False
        
        # Check spread
        spread = market_data.get('spread_pct', 0)
        if spread > 0.0005:  # 0.05% maximum spread
            logger.debug(f"Filtered {symbol}: Spread too high ({spread:.4%})")
            return False
        
        # Check volume (min $1M volume)
        volume = market_data.get('volume', 0)
        if volume < 1000000:
            logger.debug(f"Filtered {symbol}: Volume too low (${volume:,.0f})")
            return False
        
        # Check if price is stagnant
        if 'change_24h' in market_data and abs(market_data['change_24h']) < 0.005:
            logger.debug(f"Filtered {symbol}: Price stagnant ({market_data['change_24h']:.2%})")
            return False
        
        return True
    
    @staticmethod
    def get_trade_quality(symbol: str, prediction: Dict, market_data: Dict) -> float:
        """Calculate trade quality score (0-1)"""
        base_score = prediction.get('confidence', 0)
        
        # Adjust for volatility
        volatility = prediction.get('volatility_regime', 'medium')
        if volatility == 'extreme':
            base_score *= 0.7
        elif volatility == 'high':
            base_score *= 0.9
        
        # Adjust for spread
        spread = market_data.get('spread_pct', 0)
        if spread > 0.0002:
            base_score *= (1 - spread * 100)  # Reduce score with spread
        
        # Adjust for model agreement
        agreement = prediction.get('model_agreement', 0.5)
        if agreement > 0.8:
            base_score *= 1.2
        elif agreement < 0.5:
            base_score *= 0.8
        
        return min(max(base_score, 0), 1.0)

# ============================================================================
# DEEP LEARNING PREDICTOR - IMPROVED
# ============================================================================

class DeepLearningPredictor:
    """Deep Learning prediction engine with improved features"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = defaultdict(list)
        self.prediction_cache = {}
        logger.info("Deep Learning Predictor initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for deep learning"""
        try:
            # Make a copy to avoid warnings
            df = df.copy()
            
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            df['close_open_pct'] = (df['close'] - df['open']) / df['open']
            
            # Moving averages
            for window in [3, 5, 8, 13, 21]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                
                # Price position relative to MAs
                if f'sma_{window}' in df.columns:
                    df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}'].replace(0, np.nan)
                if f'ema_{window}' in df.columns:
                    df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}'].replace(0, np.nan)
            
            # RSI variations
            for period in [7, 14]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss.replace(0, np.nan)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            for window in [10, 20]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                df[f'bb_upper_{window}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{window}'] = bb_middle - (bb_std * 2)
                df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / bb_middle.replace(0, np.nan)
                df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']).replace(0, np.nan)
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['obv'] = self.calculate_obv(df)
            
            # Volatility features
            df['atr'] = self.calculate_atr(df)
            for period in [5, 10, 20]:
                df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            
            # Momentum indicators
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            price_change = df['close'].diff()
            volume = df['volume']
            
            obv = pd.Series(0, index=df.index)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(df)):
                if price_change.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif price_change.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except:
            return pd.Series(0, index=df.index)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(method='bfill').fillna(0)
        except:
            return pd.Series(0, index=df.index)
    
    def validate_and_clean_data(self, X, y):
        """Validate and clean data before training"""
        try:
            X_clean = np.asarray(X, dtype=np.float64)
            y_clean = np.asarray(y, dtype=np.float64)
            
            # Check for finite values
            mask = np.all(np.isfinite(X_clean), axis=1) & np.isfinite(y_clean)
            
            X_clean = X_clean[mask]
            y_clean = y_clean[mask]
            
            if len(X_clean) < 30:
                return None, None
                
            return X_clean, y_clean
            
        except Exception as e:
            logger.warning(f"Data validation failed: {str(e)}")
            return None, None
    
    def train_ensemble_model(self, symbol: str, features: pd.DataFrame, target: pd.Series):
        """Train ensemble of models for prediction"""
        try:
            # Prepare data
            X = features.values
            y = target.values
            
            if len(X) < 50:
                logger.debug(f"Insufficient data for {symbol}: {len(X)} samples")
                return
            
            # Validate and clean data
            X_clean, y_clean = self.validate_and_clean_data(X, y)
            
            if X_clean is None or y_clean is None:
                logger.debug(f"Cannot train model for {symbol}: insufficient clean data")
                return
            
            if len(X_clean) < 30:
                logger.debug(f"Insufficient clean data for {symbol}: {len(X_clean)} samples")
                return
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Train models
            models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
            }
            
            # Train each model
            trained_models = {}
            for name, model in models.items():
                try:
                    model.fit(X_scaled, y_clean)
                    trained_models[name] = model
                except Exception as e:
                    logger.warning(f"Failed to train {name} for {symbol}: {str(e)}")
            
            if not trained_models:
                logger.warning(f"No models trained for {symbol}")
                return
            
            # Store models and scaler
            self.models[symbol] = trained_models
            self.scalers[symbol] = scaler
            
            logger.info(f"Ensemble model trained for {symbol} with {len(X_clean)} samples")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
    
    def predict_with_confidence(self, symbol: str, features: pd.DataFrame) -> Dict:
        """Predict with ensemble confidence"""
        try:
            if symbol not in self.models:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'Model not trained'}
            
            # Prepare features
            X = features.values[-1:]  # Only latest features
            
            # Clean features
            X_clean = np.asarray(X, dtype=np.float64)
            mask = np.all(np.isfinite(X_clean), axis=1)
            
            if not np.any(mask):
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'Invalid feature data'}
            
            X_clean = X_clean[mask]
            
            if len(X_clean) == 0:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'No valid features'}
            
            scaler = self.scalers[symbol]
            X_scaled = scaler.transform(X_clean)
            
            # Get predictions
            predictions = []
            confidences = []
            
            for model_name, model in self.models[symbol].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = np.max(proba)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {str(e)}")
                    continue
            
            if not predictions:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'All models failed'}
            
            # Ensemble voting
            unique_preds, counts = np.unique(predictions, return_counts=True)
            majority_pred = unique_preds[np.argmax(counts)]
            
            # Calculate confidence
            avg_confidence = np.mean(confidences)
            agreement = max(counts) / len(predictions)
            ensemble_confidence = avg_confidence * agreement
            
            # Map prediction
            direction_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
            direction = direction_map.get(majority_pred, 'HOLD')
            
            # Check confidence threshold
            if ensemble_confidence < UltimateConfig.DL_MIN_CONFIDENCE:
                return {'direction': 'HOLD', 'confidence': ensemble_confidence, 'reason': 'Low confidence'}
            
            # Generate reason
            reason = self.generate_prediction_reason(features.iloc[-1], direction)
            
            return {
                'direction': direction,
                'confidence': ensemble_confidence,
                'reason': reason,
                'model_agreement': agreement,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {str(e)}")
            return {'direction': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}
    
    def generate_prediction_reason(self, features: pd.Series, direction: str) -> str:
        """Generate human-readable reason for prediction"""
        reasons = []
        
        # RSI analysis
        for period in [7, 14]:
            rsi_key = f'rsi_{period}'
            if rsi_key in features:
                rsi = features[rsi_key]
                if direction == 'BUY' and rsi < 35:
                    reasons.append(f"RSI{period} oversold")
                elif direction == 'SELL' and rsi > 65:
                    reasons.append(f"RSI{period} overbought")
        
        # MACD analysis
        if 'macd_hist' in features:
            macd_hist = features['macd_hist']
            if direction == 'BUY' and macd_hist > 0:
                reasons.append("MACD bullish")
            elif direction == 'SELL' and macd_hist < 0:
                reasons.append("MACD bearish")
        
        # Bollinger Bands
        if 'bb_position_20' in features:
            bb_pos = features['bb_position_20']
            if direction == 'BUY' and bb_pos < 0.2:
                reasons.append("Near BB support")
            elif direction == 'SELL' and bb_pos > 0.8:
                reasons.append("Near BB resistance")
        
        # Volume
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if vol_ratio > 1.8:
                reasons.append("High volume")
        
        return ', '.join(reasons) if reasons else "Technical alignment"

# ============================================================================
# VOLATILITY ANALYZER - IMPROVED
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
            if len(df) < 20:
                return {'current': 0, 'regime': 'medium', 'ratio': 1, 'multiplier': 1.0}
            
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return {'current': 0, 'regime': 'medium', 'ratio': 1, 'multiplier': 1.0}
            
            # Calculate current volatility (annualized)
            current_vol = returns.tail(10).std() * np.sqrt(365 * 24 * 60)
            
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
            
            # Store history
            self.volatility_history[symbol].append(current_vol)
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol].pop(0)
            
            # Calculate ratio (current vs average)
            if len(self.volatility_history[symbol]) > 0:
                avg_vol = np.mean(self.volatility_history[symbol])
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            else:
                vol_ratio = 1
            
            # Update symbol volatility
            self.symbol_volatility[symbol] = {
                'current': current_vol,
                'regime': vol_regime,
                'ratio': vol_ratio,
                'multiplier': UltimateConfig.VOLATILITY_MULTIPLIER[vol_regime]
            }
            
            return self.symbol_volatility[symbol]
            
        except Exception as e:
            logger.error(f"Error analyzing volatility for {symbol}: {str(e)}")
            return {'current': 0, 'regime': 'medium', 'ratio': 1, 'multiplier': 1.0}
    
    def get_trading_params(self, symbol: str) -> Dict:
        """Get trading parameters based on volatility"""
        vol_data = self.symbol_volatility.get(symbol, {'regime': 'medium', 'multiplier': 1.0})
        multiplier = vol_data['multiplier']
        
        # Adjust parameters
        take_profit = UltimateConfig.BASE_TAKE_PROFIT * multiplier
        stop_loss = UltimateConfig.BASE_STOP_LOSS * multiplier
        trailing_stop = UltimateConfig.TRAILING_STOP * multiplier
        
        # Adjust for time of day
        hour = datetime.now().hour
        if hour in UltimateConfig.PEAK_HOURS:
            take_profit *= UltimateConfig.PEAK_MULTIPLIER
            stop_loss *= UltimateConfig.PEAK_MULTIPLIER
        
        return {
            'take_profit_pct': min(take_profit, 0.008),  # Max 0.8%
            'stop_loss_pct': min(stop_loss, 0.005),      # Max 0.5%
            'trailing_stop_pct': trailing_stop,
            'volatility_regime': vol_data['regime'],
            'volatility_multiplier': multiplier,
            'is_peak_hour': hour in UltimateConfig.PEAK_HOURS
        }

# ============================================================================
# ADVANCED EXECUTION ENGINE - IMPROVED
# ============================================================================

class AdvancedExecutionEngine:
    """Advanced execution engine with smart order routing"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 15000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.market_cache = {}
        self.cache_time = {}
        self.trade_filter = SmartTradeFilter()
        logger.info("Advanced Execution Engine initialized")
    
    def get_market_intelligence(self, symbol: str) -> Dict:
        """Get comprehensive market intelligence"""
        cache_key = symbol
        current_time = time.time()
        
        # Check cache (5 seconds)
        if cache_key in self.market_cache and current_time - self.cache_time.get(cache_key, 0) < 5:
            return self.market_cache[cache_key]
        
        try:
            # Get ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Get order book
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            
            # Calculate metrics
            bid = ticker['bid'] or ticker['last']
            ask = ticker['ask'] or ticker['last']
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            # Order book analysis
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:3]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:3]])
            orderbook_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            market_data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': ticker['last'],
                'spread_pct': spread_pct,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'orderbook_imbalance': orderbook_imbalance,
                'volume': ticker['quoteVolume'] or 0,
                'change_24h': ticker.get('percentage', 0) / 100 if ticker.get('percentage') else 0,
                'timestamp': datetime.now()
            }
            
            # Cache results
            self.market_cache[cache_key] = market_data
            self.cache_time[cache_key] = current_time
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market intelligence for {symbol}: {str(e)}")
            return None
    
    def calculate_optimal_execution(self, symbol: str, direction: str, size: float) -> Dict:
        """Calculate optimal execution parameters"""
        try:
            market_data = self.get_market_intelligence(symbol)
            if not market_data:
                return None
            
            # Apply trade filter
            if not self.trade_filter.filter_symbol(symbol, market_data):
                return None
            
            # Calculate execution price
            if direction == 'BUY':
                execution_price = market_data['ask'] * 1.0002  # Slight premium for buy
            else:
                execution_price = market_data['bid'] * 0.9998  # Slight discount for sell
            
            # Ensure reasonable price
            if direction == 'BUY':
                execution_price = min(execution_price, market_data['ask'] * 1.001)
            else:
                execution_price = max(execution_price, market_data['bid'] * 0.999)
            
            # Calculate quantity
            quantity = size / execution_price
            
            return {
                'symbol': symbol,
                'direction': direction,
                'size': size,
                'execution_price': execution_price,
                'quantity': quantity,
                'market_price': market_data['last'],
                'spread_pct': market_data['spread_pct'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating execution for {symbol}: {str(e)}")
            return None
    
    def execute_trade(self, trade_signal: Dict, capital_allocated: float) -> Dict:
        """Execute a trade with optimal parameters"""
        try:
            symbol = trade_signal['symbol']
            direction = trade_signal['direction']
            
            # Calculate optimal execution
            execution_params = self.calculate_optimal_execution(
                symbol, direction, capital_allocated
            )
            
            if not execution_params:
                logger.debug(f"Cannot execute trade for {symbol}: execution params failed")
                return None
            
            # Check trade quality
            market_data = self.get_market_intelligence(symbol)
            if market_data:
                trade_quality = self.trade_filter.get_trade_quality(
                    symbol, trade_signal, market_data
                )
                if trade_quality < 0.6:
                    logger.debug(f"Trade quality too low for {symbol}: {trade_quality:.2f}")
                    return None
            
            # Create trade record (simulated execution)
            trade_record = {
                'trade_id': f"{symbol.replace('/', '_')}_{int(time.time())}",
                'symbol': symbol,
                'direction': direction,
                'entry_price': execution_params['execution_price'],
                'size': capital_allocated,
                'quantity': execution_params['quantity'],
                'stop_loss': trade_signal['stop_loss'],
                'take_profit': trade_signal['take_profit'],
                'trailing_stop': trade_signal['trailing_stop'],
                'entry_time': datetime.now(),
                'status': 'OPEN',
                'current_price': execution_params['market_price'],
                'pnl_pct': 0.0,
                'pnl_usd': 0.0,
                'execution_data': execution_params,
                'prediction_data': trade_signal.get('prediction_data', {}),
                'volatility_data': trade_signal.get('volatility_data', {}),
                'confidence': trade_signal.get('confidence', 0)
            }
            
            logger.info(f"Trade executed: {direction} {symbol} at ${trade_record['entry_price']:.4f} "
                       f"Size: ${capital_allocated:.2f} TP: {trade_record['take_profit']*100:.2f}% "
                       f"SL: {trade_record['stop_loss']*100:.2f}%")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None

# ============================================================================
# PORTFOLIO MANAGER - IMPROVED
# ============================================================================

class PortfolioManager:
    """Advanced portfolio management with improved risk control"""
    
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
        
        # Improved risk management
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.cooldown_until = None
        self.last_hour_reset = datetime.now()
        self.last_day_reset = datetime.now().date()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info(f"Portfolio Manager initialized with ${initial_capital:.2f}")
    
    def can_open_trade(self, symbol: str, trade_size: float) -> Tuple[bool, str]:
        """Check if we can open a new trade"""
        try:
            # Check cooldown after consecutive losses
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                remaining = (self.cooldown_until - datetime.now()).seconds
                return False, f"In cooldown ({remaining}s remaining)"
            
            # Reset cooldown if time passed
            if self.cooldown_until and datetime.now() >= self.cooldown_until:
                self.cooldown_until = None
                self.consecutive_losses = 0
            
            # Check max open trades
            if len(self.open_trades) >= UltimateConfig.MAX_OPEN_TRADES:
                return False, f"Max open trades ({UltimateConfig.MAX_OPEN_TRADES})"
            
            # Check same symbol limit
            same_symbol_trades = sum(1 for trade in self.open_trades.values() 
                                   if trade['symbol'] == symbol)
            if same_symbol_trades >= UltimateConfig.MAX_SAME_SYMBOL:
                return False, f"Max trades for symbol ({UltimateConfig.MAX_SAME_SYMBOL})"
            
            # Check position size
            position_size_pct = trade_size / self.current_capital
            if position_size_pct > UltimateConfig.MAX_POSITION_SIZE:
                return False, f"Position size {position_size_pct:.1%} > max {UltimateConfig.MAX_POSITION_SIZE:.1%}"
            
            # Check minimum position size
            if position_size_pct < UltimateConfig.MIN_POSITION_SIZE:
                return False, f"Position size {position_size_pct:.1%} < min {UltimateConfig.MIN_POSITION_SIZE:.1%}"
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
            
            # Check hourly/daily targets
            current_hour = datetime.now().hour
            if current_hour != self.last_hour_reset.hour:
                self.hourly_pnl = 0.0
                self.last_hour_reset = datetime.now()
            
            if self.hourly_pnl >= self.hourly_target:
                return False, f"Hourly target reached (${self.hourly_pnl:.2f})"
            
            if self.daily_pnl >= self.daily_target:
                return False, f"Daily target reached (${self.daily_pnl:.2f})"
            
            # Check symbol cooldown
            if self.is_symbol_in_cooldown(symbol):
                return False, "Symbol in cooldown"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Error checking trade eligibility: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        # Check open trades
        for trade in self.open_trades.values():
            if trade['symbol'] == symbol:
                trade_age = (datetime.now() - trade['entry_time']).total_seconds()
                if trade_age < UltimateConfig.TRADE_COOLDOWN:
                    return True
        
        # Check recent trade history
        cutoff_time = datetime.now() - timedelta(seconds=UltimateConfig.TRADE_COOLDOWN)
        recent_trades = [t for t in self.trade_history 
                        if t['symbol'] == symbol and t['exit_time'] > cutoff_time]
        
        return len(recent_trades) > 0
    
    def calculate_position_size(self, symbol: str, volatility_data: Dict, confidence: float) -> float:
        """Calculate position size based on risk, volatility and confidence"""
        try:
            # Base risk amount
            base_risk = self.current_capital * UltimateConfig.RISK_PER_TRADE
            
            # Adjust for volatility
            vol_multiplier = volatility_data.get('multiplier', 1.0)
            
            # Reduce size for high volatility
            if vol_multiplier > 1.2:
                vol_multiplier = 0.7
            
            # Adjust for confidence
            confidence_multiplier = max(confidence, 0.5)
            
            # Calculate size
            position_size = base_risk * vol_multiplier * confidence_multiplier
            
            # Apply limits
            min_position = self.current_capital * UltimateConfig.MIN_POSITION_SIZE
            max_position = self.current_capital * UltimateConfig.MAX_POSITION_SIZE
            
            position_size = max(min_position, min(position_size, max_position))
            
            # Round to nearest $0.10
            return round(position_size, 1)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return self.current_capital * UltimateConfig.MIN_POSITION_SIZE
    
    def add_trade(self, trade_record: Dict):
        """Add a new trade to portfolio"""
        try:
            self.open_trades[trade_record['trade_id']] = trade_record
            self.daily_trades += 1
            
            # Update capital
            self.current_capital -= trade_record['size']
            
            logger.info(f"Trade added: {trade_record['symbol']} | "
                       f"Open trades: {len(self.open_trades)} | "
                       f"Capital: ${self.current_capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
    
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
                else:
                    pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                
                # Apply fees
                pnl_pct -= UltimateConfig.TAKER_FEE * 2
                
                pnl_usd = pnl_pct * trade['size']
                
                trade['pnl_pct'] = pnl_pct
                trade['pnl_usd'] = pnl_usd
                
                total_pnl += pnl_usd
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(trade)
                if exit_reason:
                    trades_to_close.append((trade_id, exit_reason))
            
            # Close trades
            for trade_id, exit_reason in trades_to_close:
                self.close_trade(trade_id, exit_reason)
            
            # Update performance
            self.update_performance_metrics()
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error updating trades: {str(e)}")
            return 0.0
    
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        """Check if trade should be closed"""
        try:
            pnl_pct = trade['pnl_pct']
            trade_age = (datetime.now() - trade['entry_time']).total_seconds()
            
            # Minimum trade duration
            if trade_age < UltimateConfig.MIN_TRADE_DURATION:
                return None
            
            # Take profit
            if pnl_pct >= trade['take_profit']:
                return f"Take profit hit: {pnl_pct:.2%}"
            
            # Stop loss
            if pnl_pct <= -trade['stop_loss']:
                return f"Stop loss hit: {pnl_pct:.2%}"
            
            # Trailing stop
            if pnl_pct > 0 and 'trailing_stop' in trade:
                if 'highest_price' not in trade:
                    trade['highest_price'] = trade['entry_price']
                
                if trade['direction'] == 'BUY':
                    trade['highest_price'] = max(trade['highest_price'], trade['current_price'])
                    trailing_level = trade['highest_price'] * (1 - trade['trailing_stop'])
                    if trade['current_price'] <= trailing_level:
                        return f"Trailing stop hit: {pnl_pct:.2%}"
            
            # Max duration (only close if not profitable)
            if trade_age > UltimateConfig.MAX_TRADE_DURATION:
                if pnl_pct > 0:
                    return f"Max duration reached with profit: {pnl_pct:.2%}"
                else:
                    return f"Max duration reached: {trade_age:.0f}s"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
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
            
            # Update consecutive losses
            if pnl_usd < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.max_consecutive_losses:
                    self.cooldown_until = datetime.now() + timedelta(minutes=15)
                    logger.warning(f"{self.consecutive_losses} consecutive losses, entering 15min cooldown")
            else:
                self.consecutive_losses = 0
            
            # Move to history
            self.trade_history.append(trade)
            del self.open_trades[trade_id]
            
            # Log result
            if pnl_usd >= 0:
                logger.info(f"Trade closed: {trade['symbol']} | Direction: {trade['direction']} | "
                          f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2%}) | Reason: {exit_reason}")
            else:
                logger.warning(f"Trade closed: {trade['symbol']} | Direction: {trade['direction']} | "
                             f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2%}) | Reason: {exit_reason}")
            
            # Update performance
            self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {str(e)}")
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.trade_history:
                return
            
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for t in self.trade_history if t['pnl_usd'] > 0)
            losing_trades = total_trades - winning_trades
            
            total_pnl = sum(t['pnl_usd'] for t in self.trade_history)
            winning_pnl = sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] > 0)
            losing_pnl = sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] < 0)
            
            self.performance_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                'profit_factor': abs(winning_pnl / losing_pnl) if losing_pnl < 0 else float('inf'),
                'current_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            open_pnl = sum(trade['pnl_usd'] for trade in self.open_trades.values())
            open_exposure = sum(trade['size'] for trade in self.open_trades.values())
            
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
                'available_capital': self.current_capital - open_exposure,
                'performance': self.performance_metrics.copy(),
                'consecutive_losses': self.consecutive_losses,
                'in_cooldown': bool(self.cooldown_until and datetime.now() < self.cooldown_until)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
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
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 150) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = time.time()
            
            # Check cache (10 seconds)
            if cache_key in self.data_cache and current_time - self.last_update.get(cache_key, 0) < 10:
                return self.data_cache[cache_key]
            
            # Fetch new data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Create DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache data
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        timeframes = {}
        
        for tf in UltimateConfig.TIMEFRAMES:
            try:
                df = self.fetch_ohlcv(symbol, tf, limit=UltimateConfig.DL_LOOKBACK)
                if not df.empty and len(df) > 50:
                    timeframes[tf] = df
            except Exception as e:
                logger.debug(f"Error fetching {tf} data for {symbol}: {str(e)}")
        
        return timeframes

# ============================================================================
# MAIN TRADING BOT - IMPROVED WITH TELEGRAM
# ============================================================================

class DeepLearningScalpingBot:
    """Main trading bot with improved logic and Telegram"""
    
    def __init__(self):
        self.config = UltimateConfig()
        self.data_collector = DataCollector()
        self.predictor = DeepLearningPredictor()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.execution_engine = AdvancedExecutionEngine()
        self.portfolio_manager = PortfolioManager(UltimateConfig.INITIAL_CAPITAL)
        
        # Initialize Telegram
        self.telegram = TelegramManager()
        
        # Training counter
        self.training_counter = 0
        
        # Start time
        self.start_time = datetime.now()
        
        logger.info("Deep Learning Scalping Bot initialized")
        logger.info(f"Monitoring {len(self.config.SYMBOLS)} symbols")
        logger.info(f"Initial capital: ${UltimateConfig.INITIAL_CAPITAL:.2f}")
        
        # Send startup message
        self.send_startup_message()
    
    def send_startup_message(self):
        """Send startup message via Telegram"""
        if self.telegram.initialized:
            message = (
                f"🤖 <b>Deep Learning Scalping Bot Started</b>\n"
                f"• Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"• Capital: ${UltimateConfig.INITIAL_CAPITAL:.2f}\n"
                f"• Symbols: {len(self.config.SYMBOLS)}\n"
                f"• Risk/Trade: {UltimateConfig.RISK_PER_TRADE*100:.1f}%\n"
                f"• Min Confidence: {UltimateConfig.DL_MIN_CONFIDENCE*100:.0f}%"
            )
            self.telegram.send_message(message)
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunities"""
        try:
            # Fetch data
            timeframes = self.data_collector.fetch_multiple_timeframes(symbol)
            
            if not timeframes:
                return None
            
            primary_data = timeframes.get(UltimateConfig.PRIMARY_TIMEFRAME)
            if primary_data is None or len(primary_data) < 50:
                return None
            
            # Analyze volatility
            volatility_data = self.volatility_analyzer.analyze_volatility(primary_data, symbol)
            
            # Create features
            features = self.predictor.create_features(primary_data.copy())
            
            # Prepare target
            if len(features) < 2:
                return None
            
            features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
            features = features.dropna()
            
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
            
            # Get trading parameters
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
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def process_trade_signal(self, trade_signal: Dict):
        """Process a trade signal"""
        try:
            symbol = trade_signal['symbol']
            
            # Calculate position size
            position_size = self.portfolio_manager.calculate_position_size(
                symbol,
                trade_signal['volatility_data'],
                trade_signal['confidence']
            )
            
            # Check if we can trade
            can_trade, reason = self.portfolio_manager.can_open_trade(symbol, position_size)
            
            if not can_trade:
                logger.debug(f"Cannot trade {symbol}: {reason}")
                return
            
            # Execute trade
            trade_record = self.execution_engine.execute_trade(trade_signal, position_size)
            
            if trade_record:
                # Add to portfolio
                self.portfolio_manager.add_trade(trade_record)
                
                # Send Telegram notification
                self.telegram.send_trade_notification(trade_record, is_open=True)
            
        except Exception as e:
            logger.error(f"Error processing trade signal for {symbol}: {str(e)}")
    
    def update_market_prices(self) -> Dict:
        """Update market prices for all symbols"""
        market_prices = {}
        
        for symbol in self.config.SYMBOLS:
            try:
                market_data = self.execution_engine.get_market_intelligence(symbol)
                if market_data:
                    market_prices[symbol] = market_data['last']
            except Exception as e:
                logger.debug(f"Error fetching price for {symbol}: {str(e)}")
        
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
                f"Uptime: {str(datetime.now() - self.start_time).split('.')[0]}\n"
                f"Capital: ${portfolio_summary.get('current_capital', 0):.2f}\n"
                f"Total Return: {portfolio_summary.get('total_return', 0)*100:+.2f}%\n"
                f"Open Trades: {portfolio_summary.get('open_trades', 0)}\n"
                f"Daily P&L: ${portfolio_summary.get('daily_pnl', 0):+.2f}\n"
                f"Hourly P&L: ${portfolio_summary.get('hourly_pnl', 0):+.2f}\n"
                f"Win Rate: {portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%\n"
                f"Total Trades: {portfolio_summary.get('performance', {}).get('total_trades', 0)}\n"
                f"Consecutive Losses: {portfolio_summary.get('consecutive_losses', 0)}\n"
                f"{'='*60}"
            )
            
            logger.info(report)
            
            # Send Telegram report
            self.telegram.send_performance_report(portfolio_summary)
            
        except Exception as e:
            logger.error(f"Error generating hourly report: {str(e)}")
    
    def run(self):
        """Main bot execution loop"""
        logger.info("Starting Deep Learning Scalping Bot...")
        
        last_hourly_report = time.time()
        last_telegram_update = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                # Hourly report
                if current_time - last_hourly_report >= UltimateConfig.HOURLY_REPORT_INTERVAL:
                    self.generate_hourly_report()
                    last_hourly_report = current_time
                
                # Telegram status update every 30 minutes
                if self.telegram.initialized and current_time - last_telegram_update >= 1800:
                    portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                    status_message = (
                        f"🔄 <b>Bot Status Update</b>\n"
                        f"• Uptime: {str(datetime.now() - self.start_time).split('.')[0]}\n"
                        f"• Capital: ${portfolio_summary.get('current_capital', 0):.2f}\n"
                        f"• Return: {portfolio_summary.get('total_return', 0)*100:+.2f}%\n"
                        f"• Open Trades: {portfolio_summary.get('open_trades', 0)}"
                    )
                    self.telegram.send_message(status_message)
                    last_telegram_update = current_time
                
                # Update market prices and trades
                market_prices = self.update_market_prices()
                if market_prices:
                    self.portfolio_manager.update_trades(market_prices)
                
                # Analyze symbols
                for symbol in self.config.SYMBOLS:
                    try:
                        trade_signal = self.analyze_symbol(symbol)
                        
                        if trade_signal and trade_signal['confidence'] >= UltimateConfig.DL_MIN_CONFIDENCE:
                            self.process_trade_signal(trade_signal)
                        
                        time.sleep(0.3)  # Small delay
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                
                # Sleep before next iteration
                time.sleep(UltimateConfig.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.send_shutdown_message()
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
            self.send_error_message(str(e))
        finally:
            self.send_shutdown_message()
    
    def send_shutdown_message(self):
        """Send shutdown message via Telegram"""
        if self.telegram.initialized:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            runtime = str(datetime.now() - self.start_time).split('.')[0]
            
            message = (
                f"🛑 <b>Bot Stopped</b>\n"
                f"• Runtime: {runtime}\n"
                f"• Final Capital: ${portfolio_summary.get('current_capital', 0):.2f}\n"
                f"• Total Return: {portfolio_summary.get('total_return', 0)*100:+.2f}%\n"
                f"• Total Trades: {portfolio_summary.get('performance', {}).get('total_trades', 0)}\n"
                f"• Win Rate: {portfolio_summary.get('performance', {}).get('win_rate', 0)*100:.1f}%"
            )
            self.telegram.send_message(message)
    
    def send_error_message(self, error: str):
        """Send error message via Telegram"""
        if self.telegram.initialized:
            message = (
                f"⚠️ <b>Bot Error</b>\n"
                f"• Time: {datetime.now().strftime('%H:%M:%S')}\n"
                f"• Error: {error[:100]}..."
            )
            self.telegram.send_message(message)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create bot instance
    bot = DeepLearningScalpingBot()
    
    # Run bot
    bot.run()      # optimal price based on direction and order
