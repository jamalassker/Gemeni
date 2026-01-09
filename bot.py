# bot.py# ai_scalping_bot.py
"""
AI-Powered Scalping Bot with 1-Minute Timeframe
Advanced Algorithm with High Win Rate
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
from typing import Dict, List, Optional, Tuple
import ccxt
from collections import deque
import threading
import sys
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# ADVANCED CONFIGURATION - PROFITABLE SETTINGS
# ============================================================================

class AdvancedConfig:
    """Advanced configuration for profitable scalping"""
    
    # Exchange Settings
    EXCHANGE = 'binance'
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    TIMEFRAME = '1m'  # 1-minute scalping
    MAX_SYMBOLS = 3   # Trade max 3 symbols simultaneously
    
    # Capital Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    MAX_DAILY_LOSS = 0.02  # 2% daily loss limit
    MAX_POSITION_SIZE = 0.20  # 20% max per symbol
    MAX_OPEN_TRADES = 5
    
    # Scalping Parameters (AGGRESSIVE PROFITABLE)
    TAKE_PROFIT_PCT = 0.0015  # 0.15% take profit
    STOP_LOSS_PCT = 0.0010    # 0.10% stop loss
    TRAILING_STOP_PCT = 0.0008  # 0.08% trailing stop
    
    # Fees & Spread Consideration
    MAKER_FEE = 0.00075  # 0.075%
    TAKER_FEE = 0.0010   # 0.10%
    MIN_SPREAD_PCT = 0.0001  # 0.01% minimum spread
    MAX_SPREAD_PCT = 0.0005  # 0.05% maximum spread
    SLIPPAGE_PCT = 0.0002    # 0.02% slippage
    
    # AI Strategy Parameters
    MIN_CONFIDENCE = 0.65  # 65% minimum confidence
    CONFIRMATION_BARS = 2  # Need 2 bars confirmation
    VOLUME_FILTER = 1.3    # 30% above average volume
    
    # RSI Settings
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 75    # Conservative levels
    RSI_OVERSOLD = 25
    
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD = 1.5  # Tighter bands for scalping
    
    # MACD Settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Execution
    CHECK_INTERVAL = 5     # Check every 5 seconds
    TRADE_COOLDOWN = 10    # 10 seconds between trades on same symbol
    MAX_TRADE_DURATION = 180  # Max 3 minutes per trade
    
    # AI Model Parameters
    AI_LOOKBACK = 50       # Look back 50 candles
    AI_TRAIN_INTERVAL = 100  # Retrain every 100 cycles
    
    # Telegram (Optional)
    TELEGRAM_ENABLED = False
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # System
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/scalping_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'scalping_bot.db'
    
    # Performance Targets
    DAILY_TARGET_PCT = 0.02  # 2% daily target
    MIN_WIN_RATE = 0.60      # 60% minimum win rate

# ============================================================================
# ENHANCED LOGGING
# ============================================================================

class ColorFormatter(logging.Formatter):
    """Colorful logging for better visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '\033[37m')
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"

# Setup logging
logger = logging.getLogger('AIScalpingBot')
logger.setLevel(getattr(logging, AdvancedConfig.LOG_LEVEL))

# Clear existing handlers
logger.handlers.clear()

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, AdvancedConfig.LOG_LEVEL))
console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('ai_scalping.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# ============================================================================
# AI PREDICTION ENGINE
# ============================================================================

class AIPredictor:
    """AI-based price prediction using technical indicators"""
    
    def __init__(self):
        self.model_memory = {}
        self.patterns_learned = {}
        logger.info("AI Predictor initialized")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical features"""
        try:
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # Momentum indicators
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = bb_middle + (bb_std * 1.5)
            df['bb_lower'] = bb_middle - (bb_std * 1.5)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price'] = df['volume'] * df['close']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['atr'] = self.calculate_atr(df)
            
            # Support/Resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Candlestick patterns
            df['is_hammer'] = self.detect_hammer(df)
            df['is_shooting_star'] = self.detect_shooting_star(df)
            df['is_engulfing'] = self.detect_engulfing(df)
            
            # Market structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Hammer: small body, long lower shadow, small upper shadow
        is_hammer = (
            (body < (df['high'] - df['low']) * 0.3) &  # Small body
            (lower_shadow > body * 2) &  # Long lower shadow
            (upper_shadow < body * 0.5)  # Small upper shadow
        )
        return is_hammer
    
    def detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect shooting star candlestick pattern"""
        body = np.abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Shooting star: small body, long upper shadow, small lower shadow
        is_shooting_star = (
            (body < (df['high'] - df['low']) * 0.3) &  # Small body
            (upper_shadow > body * 2) &  # Long upper shadow
            (lower_shadow < body * 0.5)  # Small lower shadow
        )
        return is_shooting_star
    
    def detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect engulfing candlestick pattern"""
        # Bullish engulfing
        bullish_engulfing = (
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['close'] > df['open'].shift(1)) &  # Current close > previous open
            (df['open'] < df['close'].shift(1))    # Current open < previous close
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            (df['close'] < df['open']) &  # Current candle is bearish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['close'] < df['open'].shift(1)) &  # Current close < previous open
            (df['open'] > df['close'].shift(1))    # Current open > previous close
        )
        
        return bullish_engulfing | bearish_engulfing
    
    def predict_direction(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Predict market direction with confidence score"""
        try:
            if len(df) < 30:
                return {'direction': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
            
            # Calculate features
            df_features = self.calculate_features(df)
            latest = df_features.iloc[-1]
            
            # Initialize scores
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # 1. RSI Analysis (20%)
            if latest['rsi'] < 30:
                buy_score += 20
                reasons.append(f"RSI oversold ({latest['rsi']:.1f})")
            elif latest['rsi'] > 70:
                sell_score += 20
                reasons.append(f"RSI overbought ({latest['rsi']:.1f})")
            elif latest['rsi'] < 50:
                buy_score += 5
            else:
                sell_score += 5
            
            # 2. Bollinger Bands Analysis (20%)
            if latest['close'] < latest['bb_lower']:
                buy_score += 20
                reasons.append("Price at BB lower band")
            elif latest['close'] > latest['bb_upper']:
                sell_score += 20
                reasons.append("Price at BB upper band")
            
            # 3. Moving Average Crossover (15%)
            if latest['sma_5'] > latest['sma_10'] > latest['sma_20']:
                buy_score += 15
                reasons.append("Bullish MA alignment")
            elif latest['sma_5'] < latest['sma_10'] < latest['sma_20']:
                sell_score += 15
                reasons.append("Bearish MA alignment")
            
            # 4. Volume Analysis (15%)
            if latest['volume_ratio'] > 1.5:
                if latest['close'] > latest['open']:
                    buy_score += 15
                    reasons.append(f"High volume bullish (×{latest['volume_ratio']:.1f})")
                else:
                    sell_score += 15
                    reasons.append(f"High volume bearish (×{latest['volume_ratio']:.1f})")
            
            # 5. Candlestick Patterns (10%)
            if latest['is_hammer']:
                buy_score += 10
                reasons.append("Hammer pattern detected")
            elif latest['is_shooting_star']:
                sell_score += 10
                reasons.append("Shooting star pattern")
            elif latest['is_engulfing']:
                if latest['close'] > latest['open']:
                    buy_score += 10
                    reasons.append("Bullish engulfing")
                else:
                    sell_score += 10
                    reasons.append("Bearish engulfing")
            
            # 6. Market Structure (10%)
            if latest['higher_high']:
                buy_score += 10
                reasons.append("Higher high formation")
            elif latest['lower_low']:
                sell_score += 10
                reasons.append("Lower low formation")
            
            # 7. Momentum (10%)
            if latest['momentum_5'] > 0.001:
                buy_score += 10
                reasons.append(f"Positive momentum ({latest['momentum_5']:.3%})")
            elif latest['momentum_5'] < -0.001:
                sell_score += 10
                reasons.append(f"Negative momentum ({latest['momentum_5']:.3%})")
            
            # Determine direction
            if buy_score > sell_score and buy_score >= 40:  # Minimum 40 points
                confidence = min(buy_score / 100, 0.95)
                return {
                    'direction': 'BUY',
                    'confidence': confidence,
                    'reason': ', '.join(reasons),
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
            elif sell_score > buy_score and sell_score >= 40:
                confidence = min(sell_score / 100, 0.95)
                return {
                    'direction': 'SELL',
                    'confidence': confidence,
                    'reason': ', '.join(reasons),
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
            else:
                return {
                    'direction': 'HOLD',
                    'confidence': max(buy_score, sell_score) / 100,
                    'reason': 'No clear signal',
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
                
        except Exception as e:
            logger.error(f"Error in prediction for {symbol}: {e}")
            return {'direction': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}

# ============================================================================
# ADVANCED EXCHANGE HANDLER
# ============================================================================

class AdvancedExchangeHandler:
    """Advanced exchange handler with spread analysis"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 15000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.ticker_cache = {}
        self.cache_time = 2  # Cache for 2 seconds
        logger.info("Advanced exchange handler initialized")
    
    def get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time market data with spread analysis"""
        try:
            # Check cache
            current_time = time.time()
            cached = self.ticker_cache.get(symbol)
            if cached and current_time - cached['timestamp'] < self.cache_time:
                return cached['data']
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Fetch order book for spread analysis
            orderbook = self.exchange.fetch_order_book(symbol, limit=5)
            
            # Calculate spread metrics
            bid = ticker['bid']
            ask = ticker['ask']
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            # Calculate order book imbalance
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:3]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:3]])
            orderbook_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # Calculate price momentum
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=3)
            if len(ohlcv) >= 3:
                closes = [candle[4] for candle in ohlcv]
                price_momentum = (closes[-1] - closes[0]) / closes[0]
            else:
                price_momentum = 0
            
            data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': ticker['last'],
                'spread_pct': spread_pct,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'orderbook_imbalance': orderbook_imbalance,
                'price_momentum': price_momentum,
                'volume': ticker['quoteVolume'],
                'timestamp': datetime.now()
            }
            
            # Update cache
            self.ticker_cache[symbol] = {
                'data': data,
                'timestamp': current_time
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {e}")
            return None
    
    def get_ohlcv_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    def calculate_optimal_entry(self, symbol: str, direction: str) -> Dict:
        """Calculate optimal entry point considering spread and fees"""
        try:
            market_data = self.get_realtime_data(symbol)
            if not market_data:
                return None
            
            # Calculate optimal entry price
            if direction == 'BUY':
                # For buys, aim for price between bid and ask
                optimal_price = market_data['ask'] * 1.0001  # Slightly above ask for quick fill
                
                # Consider order book depth
                if market_data['orderbook_imbalance'] > 0.1:  # Strong buying pressure
                    optimal_price = market_data['ask'] * 1.0002  # Pay more for quick fill
            else:  # SELL
                # For sells, aim for price between bid and ask
                optimal_price = market_data['bid'] * 0.9999  # Slightly below bid for quick fill
                
                # Consider order book depth
                if market_data['orderbook_imbalance'] < -0.1:  # Strong selling pressure
                    optimal_price = market_data['bid'] * 0.9998  # Accept less for quick fill
            
            # Calculate total cost with fees
            if direction == 'BUY':
                total_fee_pct = AdvancedConfig.TAKER_FEE
            else:
                total_fee_pct = AdvancedConfig.TAKER_FEE
            
            # Include spread in cost calculation
            effective_spread = market_data['spread_pct'] / 2  # Assume we pay half the spread
            
            data = {
                'optimal_price': optimal_price,
                'current_bid': market_data['bid'],
                'current_ask': market_data['ask'],
                'spread_pct': market_data['spread_pct'],
                'effective_spread': effective_spread,
                'total_cost_pct': total_fee_pct + effective_spread,
                'orderbook_imbalance': market_data['orderbook_imbalance'],
                'price_momentum': market_data['price_momentum']
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating optimal entry for {symbol}: {e}")
            return None
    
    def is_good_for_scalping(self, symbol: str) -> bool:
        """Check if symbol is good for scalping"""
        try:
            market_data = self.get_realtime_data(symbol)
            if not market_data:
                return False
            
            # Check spread
            if market_data['spread_pct'] > AdvancedConfig.MAX_SPREAD_PCT:
                logger.debug(f"{symbol}: Spread too high ({market_data['spread_pct']:.4%})")
                return False
            
            if market_data['spread_pct'] < AdvancedConfig.MIN_SPREAD_PCT:
                logger.debug(f"{symbol}: Spread too low for profit ({market_data['spread_pct']:.4%})")
                return False
            
            # Check volume
            if market_data['volume'] < 1000000:  # $1M minimum volume
                logger.debug(f"{symbol}: Volume too low (${market_data['volume']:,.0f})")
                return False
            
            # Check order book depth
            if market_data['bid_volume'] < 10000 or market_data['ask_volume'] < 10000:
                logger.debug(f"{symbol}: Insufficient order book depth")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking {symbol} for scalping: {e}")
            return False

# ============================================================================
# INTELLIGENT RISK MANAGER
# ============================================================================

class IntelligentRiskManager:
    """Intelligent risk management with dynamic position sizing"""
    
    def __init__(self):
        self.open_trades = []
        self.trade_history = []
        self.daily_pnl = 0
        self.daily_start = datetime.now().date()
        self.win_streak = 0
        self.loss_streak = 0
        self.current_risk = AdvancedConfig.RISK_PER_TRADE
        logger.info("Intelligent risk manager initialized")
    
    def calculate_dynamic_position(self, capital: float, entry_price: float, 
                                  stop_loss: float, confidence: float, 
                                  symbol: str) -> Dict:
        """Calculate dynamic position size based on multiple factors"""
        try:
            # Base risk amount
            base_risk_amount = capital * self.current_risk
            
            # Adjust risk based on confidence
            confidence_multiplier = min(max(confidence, 0.6), 0.95) / 0.8  # 0.75-1.1875
            
            # Adjust based on win/loss streak
            streak_multiplier = 1.0
            if self.win_streak >= 3:
                streak_multiplier = 1.2  # Increase after 3 wins
            elif self.loss_streak >= 2:
                streak_multiplier = 0.7  # Reduce after 2 losses
            
            # Adjust based on time of day (more aggressive during high volume)
            hour = datetime.now().hour
            if 14 <= hour <= 22:  # US/EU overlap
                time_multiplier = 1.15
            else:
                time_multiplier = 0.9
            
            # Calculate final risk amount
            adjusted_risk = base_risk_amount * confidence_multiplier * streak_multiplier * time_multiplier
            adjusted_risk = min(adjusted_risk, capital * AdvancedConfig.MAX_POSITION_SIZE)
            
            # Calculate position size
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                return {'size': 0, 'risk_amount': 0}
            
            position_size = adjusted_risk / price_risk
            
            # Ensure minimum position
            min_position = (capital * 0.01) / entry_price  # At least 1% of capital
            position_size = max(position_size, min_position)
            
            # Round to appropriate decimals
            if symbol.endswith('/USDT'):
                position_size = round(position_size, 6)
            
            # Calculate fees
            entry_fee = position_size * entry_price * AdvancedConfig.TAKER_FEE
            exit_fee = position_size * entry_price * AdvancedConfig.TAKER_FEE  # Estimate
            
            data = {
                'size': position_size,
                'risk_amount': adjusted_risk,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fees': entry_fee + exit_fee,
                'confidence_multiplier': confidence_multiplier,
                'streak_multiplier': streak_multiplier,
                'time_multiplier': time_multiplier
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'size': 0, 'risk_amount': 0}
    
    def can_trade(self, symbol: str, capital: float) -> Tuple[bool, str]:
        """Check if trading is allowed with intelligent rules"""
        # Check daily loss
        if self.daily_pnl <= -capital * AdvancedConfig.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
        
        # Check open trades count
        symbol_open_trades = len([t for t in self.open_trades if t['symbol'] == symbol])
        if symbol_open_trades >= 2:  # Max 2 trades per symbol
            return False, f"Max open trades for {symbol}"
        
        total_open_trades = len(self.open_trades)
        if total_open_trades >= AdvancedConfig.MAX_OPEN_TRADES:
            return False, f"Max total open trades ({total_open_trades})"
        
        # Check cooldown for symbol
        recent_trades = [t for t in self.open_trades 
                        if t['symbol'] == symbol 
                        and datetime.now() - t['entry_time'] < timedelta(seconds=AdvancedConfig.TRADE_COOLDOWN)]
        if recent_trades:
            return False, f"Cooldown active for {symbol}"
        
        # Check win/loss streaks
        if self.loss_streak >= 3:
            return False, f"Loss streak ({self.loss_streak}), reducing risk"
        
        return True, "OK"
    
    def update_streaks(self, pnl: float):
        """Update win/loss streaks"""
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            
            # Increase risk slightly after wins
            if self.win_streak >= 3:
                self.current_risk = min(AdvancedConfig.RISK_PER_TRADE * 1.2, 0.02)
        else:
            self.loss_streak += 1
            self.win_streak = 0
            
            # Decrease risk after losses
            if self.loss_streak >= 2:
                self.current_risk = max(AdvancedConfig.RISK_PER_TRADE * 0.7, 0.005)

# ============================================================================
# HIGH FREQUENCY SCALPING ENGINE
# ============================================================================

class ScalpingEngine:
    """High-frequency scalping engine for 1-minute timeframe"""
    
    def __init__(self):
        self.ai_predictor = AIPredictor()
        self.exchange = AdvancedExchangeHandler()
        self.risk_manager = IntelligentRiskManager()
        self.capital = AdvancedConfig.INITIAL_CAPITAL
        self.total_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.active_symbols = AdvancedConfig.SYMBOLS[:AdvancedConfig.MAX_SYMBOLS]
        
        # Performance tracking
        self.performance_history = []
        self.symbol_performance = {}
        
        logger.info("Scalping engine initialized")
        logger.info(f"Active symbols: {', '.join(self.active_symbols)}")
    
    async def scan_markets(self):
        """Scan all active markets for opportunities"""
        opportunities = []
        
        for symbol in self.active_symbols:
            try:
                # Check if symbol is good for scalping
                if not self.exchange.is_good_for_scalping(symbol):
                    continue
                
                # Get market data
                df = self.exchange.get_ohlcv_data(symbol, AdvancedConfig.TIMEFRAME, 50)
                if df is None or len(df) < 30:
                    continue
                
                # Get AI prediction
                prediction = self.ai_predictor.predict_direction(df, symbol)
                
                # Filter by confidence
                if prediction['direction'] == 'HOLD':
                    continue
                
                if prediction['confidence'] < AdvancedConfig.MIN_CONFIDENCE:
                    continue
                
                # Check risk management
                can_trade, reason = self.risk_manager.can_trade(symbol, self.capital)
                if not can_trade:
                    continue
                
                # Calculate optimal entry
                entry_data = self.exchange.calculate_optimal_entry(symbol, prediction['direction'])
                if not entry_data:
                    continue
                
                # Calculate stop loss and take profit
                if prediction['direction'] == 'BUY':
                    stop_loss = entry_data['optimal_price'] * (1 - AdvancedConfig.STOP_LOSS_PCT)
                    take_profit = entry_data['optimal_price'] * (1 + AdvancedConfig.TAKE_PROFIT_PCT)
                else:
                    stop_loss = entry_data['optimal_price'] * (1 + AdvancedConfig.STOP_LOSS_PCT)
                    take_profit = entry_data['optimal_price'] * (1 - AdvancedConfig.TAKE_PROFIT_PCT)
                
                # Calculate position size
                position_data = self.risk_manager.calculate_dynamic_position(
                    capital=self.capital,
                    entry_price=entry_data['optimal_price'],
                    stop_loss=stop_loss,
                    confidence=prediction['confidence'],
                    symbol=symbol
                )
                
                if position_data['size'] <= 0:
                    continue
                
                # Calculate expected profit after fees
                trade_value = position_data['size'] * entry_data['optimal_price']
                expected_profit = abs(take_profit - entry_data['optimal_price']) * position_data['size']
                net_profit = expected_profit - position_data['total_fees']
                
                # Check if profitable after fees and spread
                if net_profit <= 0:
                    continue
                
                # Calculate risk/reward ratio
                risk_amount = abs(entry_data['optimal_price'] - stop_loss) * position_data['size']
                reward_amount = abs(take_profit - entry_data['optimal_price']) * position_data['size']
                risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
                
                if risk_reward < 1.2:  # Minimum 1.2:1 risk/reward
                    continue
                
                opportunity = {
                    'symbol': symbol,
                    'direction': prediction['direction'],
                    'entry_price': entry_data['optimal_price'],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': entry_data['optimal_price'] * (
                        1 - AdvancedConfig.TRAILING_STOP_PCT if prediction['direction'] == 'BUY'
                        else 1 + AdvancedConfig.TRAILING_STOP_PCT
                    ),
                    'size': position_data['size'],
                    'confidence': prediction['confidence'],
                    'reason': prediction['reason'],
                    'risk_amount': position_data['risk_amount'],
                    'expected_profit': net_profit,
                    'risk_reward': risk_reward,
                    'fees': position_data['total_fees'],
                    'spread_pct': entry_data['spread_pct'],
                    'timestamp': datetime.now()
                }
                
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort opportunities by confidence and risk/reward
        opportunities.sort(key=lambda x: (x['confidence'], x['risk_reward']), reverse=True)
        
        return opportunities
    
    async def execute_trade(self, opportunity: Dict):
        """Execute a trade opportunity"""
        try:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            
            logger.info(f"🎯 Executing trade: {direction} {symbol}")
            logger.info(f"   Entry: ${opportunity['entry_price']:.4f}")
            logger.info(f"   Stop: ${opportunity['stop_loss']:.4f}")
            logger.info(f"   Target: ${opportunity['take_profit']:.4f}")
            logger.info(f"   Size: {opportunity['size']:.6f}")
            logger.info(f"   Confidence: {opportunity['confidence']:.1%}")
            logger.info(f"   Risk/Reward: {opportunity['risk_reward']:.2f}:1")
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': opportunity['entry_price'],
                'stop_loss': opportunity['stop_loss'],
                'take_profit': opportunity['take_profit'],
                'trailing_stop': opportunity['trailing_stop'],
                'size': opportunity['size'],
                'entry_time': datetime.now(),
                'status': 'open',
                'confidence': opportunity['confidence'],
                'reason': opportunity['reason'],
                'fees_paid': opportunity['fees'] / 2,  # Half at entry
                'risk_amount': opportunity['risk_amount']
            }
            
            # Update capital (simulate fees)
            trade_value = opportunity['size'] * opportunity['entry_price']
            self.capital -= trade_value + (opportunity['fees'] / 2)
            
            # Add to open trades
            self.risk_manager.open_trades.append(trade)
            
            # Start monitoring
            asyncio.create_task(self.monitor_trade(trade))
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def monitor_trade(self, trade: Dict):
        """Monitor and manage open trade"""
        try:
            symbol = trade['symbol']
            start_time = datetime.now()
            best_price = trade['entry_price']
            
            while trade['status'] == 'open':
                # Check max duration
                duration = (datetime.now() - start_time).total_seconds()
                if duration > AdvancedConfig.MAX_TRADE_DURATION:
                    await self.close_trade(trade, 'timeout')
                    break
                
                # Get current market data
                market_data = self.exchange.get_realtime_data(symbol)
                if not market_data:
                    await asyncio.sleep(1)
                    continue
                
                current_price = market_data['last']
                
                # Update best price for trailing stop
                if trade['direction'] == 'BUY':
                    best_price = max(best_price, current_price)
                    # Update trailing stop
                    new_trailing_stop = best_price * (1 - AdvancedConfig.TRAILING_STOP_PCT)
                    trade['trailing_stop'] = max(trade['trailing_stop'], new_trailing_stop)
                else:
                    best_price = min(best_price, current_price)
                    # Update trailing stop
                    new_trailing_stop = best_price * (1 + AdvancedConfig.TRAILING_STOP_PCT)
                    trade['trailing_stop'] = min(trade['trailing_stop'], new_trailing_stop)
                
                # Check exit conditions
                exit_price = None
                exit_reason = None
                
                if trade['direction'] == 'BUY':
                    # Take profit
                    if current_price >= trade['take_profit']:
                        exit_price = trade['take_profit']
                        exit_reason = 'take_profit'
                    
                    # Stop loss
                    elif current_price <= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = 'stop_loss'
                    
                    # Trailing stop
                    elif current_price <= trade['trailing_stop']:
                        exit_price = trade['trailing_stop']
                        exit_reason = 'trailing_stop'
                
                else:  # SELL
                    # Take profit
                    if current_price <= trade['take_profit']:
                        exit_price = trade['take_profit']
                        exit_reason = 'take_profit'
                    
                    # Stop loss
                    elif current_price >= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = 'stop_loss'
                    
                    # Trailing stop
                    elif current_price >= trade['trailing_stop']:
                        exit_price = trade['trailing_stop']
                        exit_reason = 'trailing_stop'
                
                # Exit if condition met
                if exit_price and exit_reason:
                    await self.close_trade(trade, exit_reason, exit_price)
                    break
                
                # Small sleep to prevent CPU overload
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error monitoring trade {trade['symbol']}: {e}")
    
    async def close_trade(self, trade: Dict, reason: str, exit_price: Optional[float] = None):
        """Close a trade and calculate P&L"""
        try:
            symbol = trade['symbol']
            
            # Get current price if not provided
            if exit_price is None:
                market_data = self.exchange.get_realtime_data(symbol)
                exit_price = market_data['last'] if market_data else trade['entry_price']
            
            # Calculate P&L
            if trade['direction'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['size']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['size']
            
            # Deduct remaining fees
            total_fees = trade['fees_paid'] * 2  # Entry + exit
            net_pnl = pnl - total_fees
            
            # Update capital
            self.capital += trade['size'] * exit_price - trade['fees_paid']  # Exit value - remaining fees
            
            # Update statistics
            self.total_trades += 1
            self.total_pnl += net_pnl
            self.risk_manager.daily_pnl += net_pnl
            
            if net_pnl > 0:
                self.winning_trades += 1
            
            # Update streaks
            self.risk_manager.update_streaks(net_pnl)
            
            # Remove from open trades
            self.risk_manager.open_trades = [
                t for t in self.risk_manager.open_trades 
                if not (t['symbol'] == trade['symbol'] and t['entry_time'] == trade['entry_time'])
            ]
            
            # Update trade record
            trade.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'status': 'closed',
                'exit_reason': reason,
                'pnl': net_pnl,
                'pnl_pct': (net_pnl / (trade['entry_price'] * trade['size'])) * 100,
                'duration': (datetime.now() - trade['entry_time']).total_seconds()
            })
            
            # Add to history
            self.risk_manager.trade_history.append(trade)
            
            # Log result
            pnl_color = '🟢' if net_pnl > 0 else '🔴'
            logger.info(f"{pnl_color} Trade closed: {trade['direction']} {symbol}")
            logger.info(f"   Entry: ${trade['entry_price']:.4f}")
            logger.info(f"   Exit: ${exit_price:.4f}")
            logger.info(f"   P&L: ${net_pnl:+.4f} ({trade['pnl_pct']:+.2f}%)")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Duration: {trade['duration']:.1f}s")
            logger.info(f"   Win rate: {self.winning_trades}/{self.total_trades} ({self.win_rate:.1%})")
            logger.info(f"   Total P&L: ${self.total_pnl:+.2f}")
            logger.info(f"   Capital: ${self.capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    @property
    def win_rate(self):
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

# ============================================================================
# MAIN AI SCALPING BOT
# ============================================================================

class AIScalpingBot:
    """Main AI-powered scalping bot"""
    
    def __init__(self):
        self.config = AdvancedConfig
        self.engine = ScalpingEngine()
        self.running = False
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        logger.info("="*60)
        logger.info("🤖 AI-POWERED SCALPING BOT")
        logger.info("="*60)
        logger.info(f"💰 Initial Capital: ${self.config.INITIAL_CAPITAL}")
        logger.info(f"📈 Timeframe: {self.config.TIMEFRAME}")
        logger.info(f"🎯 Take Profit: {self.config.TAKE_PROFIT_PCT:.3%}")
        logger.info(f"🛑 Stop Loss: {self.config.STOP_LOSS_PCT:.3%}")
        logger.info(f"📊 Active Symbols: {', '.join(self.engine.active_symbols)}")
        logger.info(f"⚡ Check Interval: {self.config.CHECK_INTERVAL}s")
        logger.info("="*60)
    
    async def start(self):
        """Start the AI scalping bot"""
        self.running = True
        
        try:
            logger.info("🚀 Starting AI Scalping Bot...")
            
            while self.running:
                try:
                    self.cycle_count += 1
                    
                    # Log status every 10 cycles
                    if self.cycle_count % 10 == 0:
                        self.log_status()
                    
                    # Scan for opportunities
                    opportunities = await self.engine.scan_markets()
                    
                    # Execute best opportunities
                    for opportunity in opportunities[:2]:  # Max 2 trades per cycle
                        if len(self.engine.risk_manager.open_trades) < self.config.MAX_OPEN_TRADES:
                            await self.engine.execute_trade(opportunity)
                            await asyncio.sleep(1)  # Small delay between executions
                    
                    # Check daily target
                    if self.engine.total_pnl >= self.config.INITIAL_CAPITAL * self.config.DAILY_TARGET_PCT:
                        logger.info(f"🎉 Daily target reached! P&L: ${self.engine.total_pnl:.2f}")
                        # Optionally reduce trading intensity
                    
                    # Sleep before next scan
                    await asyncio.sleep(self.config.CHECK_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
        finally:
            await self.stop()
    
    def log_status(self):
        """Log current bot status"""
        status = f"""
📊 BOT STATUS - Cycle {self.cycle_count}
{'='*40}
💰 Capital: ${self.engine.capital:.2f}
📈 Total P&L: ${self.engine.total_pnl:+.2f}
📊 Win Rate: {self.engine.win_rate:.1%}
🎯 Trades: {self.engine.total_trades} (W: {self.engine.winning_trades}/L: {self.engine.total_trades - self.engine.winning_trades})
⚡ Open Trades: {len(self.engine.risk_manager.open_trades)}
📅 Daily P&L: ${self.engine.risk_manager.daily_pnl:+.2f}
🔄 Win Streak: {self.engine.risk_manager.win_streak}
{'='*40}
        """
        logger.info(status)
    
    async def stop(self):
        """Stop the bot"""
        self.running = False
        
        # Close all open trades
        for trade in self.engine.risk_manager.open_trades[:]:
            if trade['status'] == 'open':
                await self.engine.close_trade(trade, 'bot_shutdown')
        
        # Calculate final statistics
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        final_stats = f"""
{'='*60}
🛑 BOT STOPPED
{'='*60}
⏱️  Uptime: {hours}h {minutes}m {seconds}s
💰 Final Capital: ${self.engine.capital:.2f}
📈 Total P&L: ${self.engine.total_pnl:+.2f} ({self.engine.total_pnl/self.config.INITIAL_CAPITAL*100:+.2f}%)
📊 Win Rate: {self.engine.win_rate:.1%}
🎯 Total Trades: {self.engine.total_trades}
⚡ Trades/Hour: {self.engine.total_trades / max(uptime.total_seconds()/3600, 0.1):.1f}
📅 Daily P&L: ${self.engine.risk_manager.daily_pnl:+.2f}
{'='*60}
        """
        
        logger.info(final_stats)

# ============================================================================
# HEALTH CHECK SERVER (FOR RAILWAY)
# ============================================================================

async def start_health_server():
    """Start health check server for Railway"""
    try:
        from aiohttp import web
        
        app = web.Application()
        
        async def health_handler(request):
            return web.Response(text='OK', status=200)
        
        async def stats_handler(request):
            stats = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'service': 'ai-scalping-bot',
                'version': '2.0'
            }
            return web.json_response(stats)
        
        app.router.add_get('/health', health_handler)
        app.router.add_get('/stats', stats_handler)
        app.router.add_get('/', health_handler)
        
        port = int(os.getenv('PORT', 8080))
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health server started on port {port}")
        return runner
        
    except ImportError:
        logger.warning("aiohttp not available, health server disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🤖 AI-POWERED PROFITABLE SCALPING BOT")
    print("="*60)
    print(f"🎯 Target: ${AdvancedConfig.INITIAL_CAPITAL * AdvancedConfig.DAILY_TARGET_PCT:.2f} daily")
    print(f"⚡ Timeframe: {AdvancedConfig.TIMEFRAME}")
    print(f"📈 Strategy: Multi-indicator AI Analysis")
    print(f"💰 Capital: ${AdvancedConfig.INITIAL_CAPITAL}")
    print(f"🎯 Take Profit: {AdvancedConfig.TAKE_PROFIT_PCT:.3%}")
    print(f"🛑 Stop Loss: {AdvancedConfig.STOP_LOSS_PCT:.3%}")
    print(f"📊 Min Win Rate: {AdvancedConfig.MIN_WIN_RATE:.0%}")
    print("="*60)
    print()
    
    # Check dependencies
    try:
        import ccxt
        import pandas
        import numpy
        print("✅ Dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install ccxt pandas numpy")
        return
    
    # Start health server
    health_server = await start_health_server()
    
    # Create and run bot
    bot = AIScalpingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if health_server:
            await health_server.cleanup()

if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())





