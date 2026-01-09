
# ultimate_scalping_bot_fixed.py
"""
Ultimate Scalping Bot - Fixed Version for Railway
Fixed Telegram and NaN issues
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
import sys
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# STABLE CONFIGURATION - OPTIMIZED FOR RAILWAY
# ============================================================================

class StableConfig:
    """Stable configuration optimized for Railway"""
    
    # Exchange & Symbols
    EXCHANGE = 'binance'
    
    # Top 10 most volatile cryptocurrencies (reduced for stability)
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
        'ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
        'AVAX/USDT'
    ]
    
    # Timeframes
    PRIMARY_TIMEFRAME = '1m'
    
    # Capital Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.006  # 0.6% risk per trade
    MAX_DAILY_LOSS = 0.01   # 1% daily loss limit
    HOURLY_TARGET = 0.003   # 0.3% hourly profit target
    
    # Position Sizing
    MAX_POSITION_SIZE = 0.10  # 10% max per symbol
    MAX_OPEN_TRADES = 4       # Max 4 open trades
    
    # Scalping Parameters
    TAKE_PROFIT_PCT = 0.0015  # 0.15% take profit
    STOP_LOSS_PCT = 0.0010    # 0.10% stop loss
    TRAILING_STOP = 0.0006    # 0.06% trailing stop
    
    # Fees & Spread
    TAKER_FEE = 0.0010
    MIN_SPREAD = 0.0001
    MAX_SPREAD = 0.0004
    
    # Strategy Parameters
    MIN_CONFIDENCE = 0.60    # 60% minimum confidence
    MIN_VOLUME = 1000000     # $1M minimum volume
    CHECK_INTERVAL = 10      # Check every 10 seconds
    TRADE_COOLDOWN = 30      # 30 seconds cooldown
    MAX_TRADE_DURATION = 180 # Max 3 minutes
    
    # Telegram (Fixed)
    TELEGRAM_ENABLED = True
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Verify Telegram configuration
    if TELEGRAM_ENABLED:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            TELEGRAM_ENABLED = False
            print("⚠️ Telegram disabled: Missing token or chat ID")
        elif str(TELEGRAM_CHAT_ID) == str(TELEGRAM_BOT_TOKEN).split(':')[0]:
            TELEGRAM_ENABLED = False
            print("⚠️ Telegram disabled: Chat ID appears to be a bot ID")
    
    # System
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/scalping_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'scalping_bot.db'
    PAPER_TRADING = True

# ============================================================================
# ENHANCED LOGGING
# ============================================================================

class EnhancedLogger:
    """Enhanced logging with colors and file output"""
    
    @staticmethod
    def setup():
        logger = logging.getLogger('StableScalpingBot')
        logger.setLevel(getattr(logging, StableConfig.LOG_LEVEL))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, StableConfig.LOG_LEVEL))
        
        class ColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',
                'INFO': '\033[32m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[41m',
                'TRADE': '\033[35m',
                'PROFIT': '\033[92m',
                'LOSS': '\033[91m',
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
        
        console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # File handler (only if directory exists)
        try:
            file_handler = logging.FileHandler('stable_bot.log')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass  # Skip file handler if not possible
        
        return logger

logger = EnhancedLogger.setup()

# ============================================================================
# SIMPLE YET EFFECTIVE PREDICTOR (FIXED NaN ISSUES)
# ============================================================================

class SimplePredictor:
    """Simple but effective predictor without NaN issues"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        logger.info("Simple Predictor initialized")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reliable indicators without NaN issues"""
        try:
            # Copy dataframe to avoid modifying original
            result = df.copy()
            
            # Ensure we have enough data
            if len(result) < 20:
                return result
            
            # Simple moving averages (with NaN handling)
            for window in [5, 10, 20]:
                result[f'sma_{window}'] = result['close'].rolling(
                    window=window, min_periods=int(window/2)
                ).mean()
            
            # RSI calculation (robust)
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14, min_periods=7).mean()
            avg_loss = loss.rolling(window=14, min_periods=7).mean()
            
            rs = avg_gain / avg_loss
            result['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_middle = result['close'].rolling(window=20, min_periods=10).mean()
            bb_std = result['close'].rolling(window=20, min_periods=10).std()
            
            result['bb_upper'] = bb_middle + (bb_std * 2)
            result['bb_lower'] = bb_middle - (bb_std * 2)
            
            # Volume indicators
            result['volume_sma'] = result['volume'].rolling(window=20, min_periods=10).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma']
            
            # Price momentum
            result['price_change'] = result['close'].pct_change(periods=3)
            result['high_low_range'] = (result['high'] - result['low']) / result['close']
            
            # Fill NaN values safely
            result = result.fillna(method='bfill').fillna(method='ffill')
            
            # Ensure no infinite values
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.fillna(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def predict_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Generate trading signal with confidence"""
        try:
            if len(df) < 30:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
            
            # Calculate indicators
            df_indicators = self.calculate_indicators(df)
            latest = df_indicators.iloc[-1]
            prev = df_indicators.iloc[-2]
            
            # Initialize scores
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # 1. RSI Analysis (25%)
            if 'rsi' in latest:
                if latest['rsi'] < 35:
                    buy_score += 25
                    reasons.append(f"RSI oversold ({latest['rsi']:.1f})")
                elif latest['rsi'] > 65:
                    sell_score += 25
                    reasons.append(f"RSI overbought ({latest['rsi']:.1f})")
            
            # 2. Price Position (20%)
            if all(col in latest for col in ['bb_upper', 'bb_lower']):
                bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                
                if bb_position < 0.2:
                    buy_score += 20
                    reasons.append("Near BB lower band")
                elif bb_position > 0.8:
                    sell_score += 20
                    reasons.append("Near BB upper band")
            
            # 3. Volume Analysis (20%)
            if 'volume_ratio' in latest and latest['volume_ratio'] > 1.3:
                if latest['close'] > latest['open']:
                    buy_score += 20
                    reasons.append(f"High volume bullish (×{latest['volume_ratio']:.1f})")
                else:
                    sell_score += 20
                    reasons.append(f"High volume bearish (×{latest['volume_ratio']:.1f})")
            
            # 4. Trend Analysis (15%)
            if all(col in latest for col in ['sma_5', 'sma_10']):
                if latest['sma_5'] > latest['sma_10']:
                    buy_score += 15
                    reasons.append("Bullish MA crossover")
                else:
                    sell_score += 15
                    reasons.append("Bearish MA crossover")
            
            # 5. Price Action (10%)
            if 'price_change' in latest:
                if latest['price_change'] > 0.001:
                    buy_score += 10
                    reasons.append(f"Positive momentum ({latest['price_change']:.2%})")
                elif latest['price_change'] < -0.001:
                    sell_score += 10
                    reasons.append(f"Negative momentum ({latest['price_change']:.2%})")
            
            # 6. Candle Pattern (10%)
            candle_body = abs(latest['close'] - latest['open'])
            candle_range = latest['high'] - latest['low']
            
            if candle_body / candle_range < 0.3:  # Doji-like
                if latest['close'] > prev['close']:
                    buy_score += 5
                else:
                    sell_score += 5
            
            # Determine final signal
            if buy_score > sell_score and buy_score >= 40:
                confidence = min(buy_score / 100, 0.9)
                return {
                    'signal': 'BUY',
                    'confidence': confidence,
                    'reason': ', '.join(reasons),
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
            elif sell_score > buy_score and sell_score >= 40:
                confidence = min(sell_score / 100, 0.9)
                return {
                    'signal': 'SELL',
                    'confidence': confidence,
                    'reason': ', '.join(reasons),
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
            else:
                return {
                    'signal': 'HOLD',
                    'confidence': max(buy_score, sell_score) / 100,
                    'reason': 'No clear signal',
                    'buy_score': buy_score,
                    'sell_score': sell_score
                }
                
        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}

# ============================================================================
# STABLE EXCHANGE HANDLER (FIXED DATA ISSUES)
# ============================================================================

class StableExchangeHandler:
    """Stable exchange handler with error handling"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.data_cache = {}
        self.cache_timeout = 5  # 5 seconds cache
        logger.info("Stable Exchange Handler initialized")
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with error handling"""
        try:
            # Check cache
            cache_key = f"{symbol}_data"
            current_time = time.time()
            
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if current_time - cached_data['timestamp'] < self.cache_timeout:
                    return cached_data['data']
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Validate data
            if not ticker or 'bid' not in ticker or 'ask' not in ticker:
                logger.warning(f"Invalid ticker data for {symbol}")
                return None
            
            bid = float(ticker['bid']) if ticker['bid'] else 0
            ask = float(ticker['ask']) if ticker['ask'] else 0
            
            if bid <= 0 or ask <= 0:
                logger.warning(f"Invalid prices for {symbol}: bid={bid}, ask={ask}")
                return None
            
            # Calculate spread
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            # Get OHLCV data
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=50)
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df = None
            except Exception as e:
                logger.warning(f"Could not fetch OHLCV for {symbol}: {e}")
                df = None
            
            data = {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'last': float(ticker['last']) if ticker['last'] else (bid + ask) / 2,
                'spread_pct': spread_pct,
                'volume': float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0,
                'ohlcv_data': df,
                'timestamp': datetime.now()
            }
            
            # Update cache
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': current_time
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable"""
        try:
            data = self.get_market_data(symbol)
            if not data:
                return False
            
            # Check spread
            if data['spread_pct'] > StableConfig.MAX_SPREAD:
                logger.debug(f"{symbol}: Spread too high ({data['spread_pct']:.4%})")
                return False
            
            if data['spread_pct'] < StableConfig.MIN_SPREAD:
                logger.debug(f"{symbol}: Spread too low ({data['spread_pct']:.4%})")
                return False
            
            # Check volume
            if data['volume'] < StableConfig.MIN_VOLUME:
                logger.debug(f"{symbol}: Volume too low (${data['volume']:,.0f})")
                return False
            
            # Check data quality
            if data['ohlcv_data'] is None or len(data['ohlcv_data']) < 20:
                logger.debug(f"{symbol}: Insufficient OHLCV data")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking tradability for {symbol}: {e}")
            return False

# ============================================================================
# ROBUST RISK MANAGER
# ============================================================================

class RobustRiskManager:
    """Robust risk management system"""
    
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.open_trades = []
        self.trade_history = []
        self.daily_pnl = 0
        self.hourly_pnl = 0
        self.hourly_start = datetime.now()
        self.daily_start = datetime.now().date()
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        # Risk control
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        logger.info(f"Robust Risk Manager initialized with ${initial_capital:.2f}")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              confidence: float, symbol: str) -> Dict:
        """Calculate position size with risk management"""
        try:
            # Base risk amount
            risk_amount = self.capital * StableConfig.RISK_PER_TRADE
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + confidence  # 0.5-1.5 range
            
            # Adjust based on performance
            performance_multiplier = 1.0
            if self.consecutive_losses >= 2:
                performance_multiplier = 0.7  # Reduce after 2 losses
            elif self.consecutive_wins >= 3:
                performance_multiplier = 1.2  # Increase after 3 wins
            
            # Calculate final risk
            adjusted_risk = risk_amount * confidence_multiplier * performance_multiplier
            
            # Apply maximum position size
            max_risk = self.capital * StableConfig.MAX_POSITION_SIZE
            adjusted_risk = min(adjusted_risk, max_risk)
            
            # Calculate position size
            price_risk = abs(entry_price - stop_loss)
            if price_risk == 0:
                return {'size': 0, 'risk_amount': 0}
            
            position_size = adjusted_risk / price_risk
            
            # Minimum position check
            min_position = (self.capital * 0.005) / entry_price  # 0.5% minimum
            position_size = max(position_size, min_position)
            
            # Round appropriately
            position_size = round(position_size, 6)
            
            # Calculate fees
            entry_fee = position_size * entry_price * StableConfig.TAKER_FEE
            exit_fee = position_size * entry_price * StableConfig.TAKER_FEE
            
            return {
                'size': position_size,
                'risk_amount': adjusted_risk,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fees': entry_fee + exit_fee,
                'confidence_multiplier': confidence_multiplier,
                'performance_multiplier': performance_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {'size': 0, 'risk_amount': 0}
    
    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        # Check daily loss limit
        if self.daily_pnl <= -self.initial_capital * StableConfig.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
        
        # Check open trades count
        symbol_trades = len([t for t in self.open_trades if t['symbol'] == symbol])
        if symbol_trades >= 2:  # Max 2 trades per symbol
            return False, f"Max trades for {symbol} ({symbol_trades})"
        
        # Check total open trades
        if len(self.open_trades) >= StableConfig.MAX_OPEN_TRADES:
            return False, f"Max open trades ({len(self.open_trades)})"
        
        # Check cooldown
        recent_trades = [t for t in self.open_trades 
                        if t['symbol'] == symbol 
                        and datetime.now() - t['entry_time'] < timedelta(seconds=StableConfig.TRADE_COOLDOWN)]
        if recent_trades:
            return False, f"Cooldown active for {symbol}"
        
        # Check loss streak
        if self.consecutive_losses >= 3:
            return False, f"Loss streak ({self.consecutive_losses}), cooling down"
        
        return True, "OK"
    
    def update_performance(self, pnl: float):
        """Update performance metrics"""
        # Update streaks
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.winning_trades += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update totals
        self.total_trades += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.hourly_pnl += pnl
        
        # Update capital
        self.capital += pnl
        
        # Reset hourly counter if needed
        if datetime.now() - self.hourly_start >= timedelta(hours=1):
            self.hourly_pnl = 0
            self.hourly_start = datetime.now()
    
    @property
    def win_rate(self):
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

# ============================================================================
# FIXED TELEGRAM NOTIFIER
# ============================================================================

class FixedTelegramNotifier:
    """Fixed Telegram notifier with proper validation"""
    
    def __init__(self):
        self.token = StableConfig.TELEGRAM_BOT_TOKEN
        self.chat_id = StableConfig.TELEGRAM_CHAT_ID
        self.enabled = StableConfig.TELEGRAM_ENABLED
        
        # Additional validation
        if self.enabled:
            if not self.token or not self.chat_id:
                self.enabled = False
                logger.warning("Telegram disabled: Missing token or chat ID")
            elif str(self.chat_id) == str(self.token).split(':')[0]:
                self.enabled = False
                logger.warning("Telegram disabled: Chat ID appears to be a bot ID")
            else:
                logger.info("Telegram Notifier initialized and validated")
        else:
            logger.info("Telegram Notifier disabled by config")
    
    async def send_message(self, message: str):
        """Send Telegram message"""
        if not self.enabled:
            return
        
        try:
            import aiohttp
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            
            # Truncate message if too long
            if len(message) > 4000:
                message = message[:4000] + "..."
            
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_notification': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent successfully")
                    elif response.status == 400:
                        error_data = await response.json()
                        logger.error(f"Telegram bad request: {error_data}")
                    elif response.status == 403:
                        logger.error("Telegram error: Bot cannot send messages to this chat")
                        self.enabled = False
                    else:
                        error_text = await response.text()
                        logger.error(f"Telegram error {response.status}: {error_text}")
                        
        except ImportError:
            logger.warning("aiohttp not installed, Telegram disabled")
            self.enabled = False
        except asyncio.TimeoutError:
            logger.warning("Telegram request timeout")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

# ============================================================================
# STABLE SCALPING BOT (MAIN ENGINE)
# ============================================================================

class StableScalpingBot:
    """Main scalping bot - stable and reliable"""
    
    def __init__(self):
        # Initialize components
        self.predictor = SimplePredictor()
        self.exchange = StableExchangeHandler()
        self.risk_manager = RobustRiskManager(StableConfig.INITIAL_CAPITAL)
        self.telegram = FixedTelegramNotifier()
        
        # State
        self.running = False
        self.cycle_count = 0
        self.start_time = datetime.now()
        
        # Performance tracking
        self.best_trade = 0
        self.worst_trade = 0
        
        logger.info("="*60)
        logger.info("🤖 STABLE SCALPING BOT")
        logger.info("="*60)
        logger.info(f"💰 Capital: ${StableConfig.INITIAL_CAPITAL}")
        logger.info(f"🎯 Hourly Target: ${StableConfig.INITIAL_CAPITAL * StableConfig.HOURLY_TARGET:.2f}")
        logger.info(f"📊 Monitoring {len(StableConfig.SYMBOLS)} Cryptocurrencies")
        logger.info(f"⚡ Timeframe: {StableConfig.PRIMARY_TIMEFRAME}")
        logger.info(f"🎯 Take Profit: {StableConfig.TAKE_PROFIT_PCT:.3%}")
        logger.info(f"🛑 Stop Loss: {StableConfig.STOP_LOSS_PCT:.3%}")
        logger.info(f"📈 Min Confidence: {StableConfig.MIN_CONFIDENCE:.0%}")
        logger.info("="*60)
    
    async def start(self):
        """Start the bot"""
        self.running = True
        
        # Send startup message
        if self.telegram.enabled:
            startup_msg = f"""
🚀 <b>STABLE SCALPING BOT STARTED</b>

💰 <b>Capital:</b> ${StableConfig.INITIAL_CAPITAL}
🎯 <b>Hourly Target:</b> ${StableConfig.INITIAL_CAPITAL * StableConfig.HOURLY_TARGET:.2f}
📊 <b>Cryptocurrencies:</b> {len(StableConfig.SYMBOLS)}
⚡ <b>Timeframe:</b> {StableConfig.PRIMARY_TIMEFRAME}

📈 <b>Trading Parameters:</b>
• Take Profit: {StableConfig.TAKE_PROFIT_PCT:.3%}
• Stop Loss: {StableConfig.STOP_LOSS_PCT:.3%}
• Risk/Trade: {StableConfig.RISK_PER_TRADE:.2%}
• Min Confidence: {StableConfig.MIN_CONFIDENCE:.0%}

Bot is now scanning for opportunities...
            """
            await self.telegram.send_message(startup_msg)
        
        logger.info("🚀 Starting Stable Scalping Bot...")
        
        try:
            while self.running:
                self.cycle_count += 1
                
                # Log status periodically
                if self.cycle_count % 30 == 0:
                    await self.log_status()
                
                # Scan and execute trades
                await self.execute_trading_cycle()
                
                # Check hourly performance
                await self.check_hourly_performance()
                
                # Sleep before next cycle
                await asyncio.sleep(StableConfig.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            if self.telegram.enabled:
                await self.telegram.send_message(f"❌ Bot crashed: {str(e)[:100]}")
        finally:
            await self.stop()
    
    async def execute_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            opportunities = []
            
            for symbol in StableConfig.SYMBOLS:
                try:
                    # Check if tradable
                    if not self.exchange.is_tradable(symbol):
                        continue
                    
                    # Get market data
                    market_data = self.exchange.get_market_data(symbol)
                    if not market_data or market_data['ohlcv_data'] is None:
                        continue
                    
                    # Get prediction
                    prediction = self.predictor.predict_signal(market_data['ohlcv_data'], symbol)
                    
                    if prediction['signal'] == 'HOLD':
                        continue
                    
                    if prediction['confidence'] < StableConfig.MIN_CONFIDENCE:
                        continue
                    
                    # Check risk management
                    can_trade, reason = self.risk_manager.can_trade(symbol)
                    if not can_trade:
                        continue
                    
                    # Calculate entry price
                    if prediction['signal'] == 'BUY':
                        entry_price = market_data['ask'] * 1.00005  # Slightly above ask
                        stop_loss = entry_price * (1 - StableConfig.STOP_LOSS_PCT)
                        take_profit = entry_price * (1 + StableConfig.TAKE_PROFIT_PCT)
                    else:  # SELL
                        entry_price = market_data['bid'] * 0.99995  # Slightly below bid
                        stop_loss = entry_price * (1 + StableConfig.STOP_LOSS_PCT)
                        take_profit = entry_price * (1 - StableConfig.TAKE_PROFIT_PCT)
                    
                    # Calculate position size
                    position_data = self.risk_manager.calculate_position_size(
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        confidence=prediction['confidence'],
                        symbol=symbol
                    )
                    
                    if position_data['size'] <= 0:
                        continue
                    
                    # Calculate expected profit
                    trade_value = position_data['size'] * entry_price
                    expected_profit = abs(take_profit - entry_price) * position_data['size']
                    net_profit = expected_profit - position_data['total_fees']
                    
                    if net_profit <= 0:
                        continue
                    
                    # Calculate risk/reward
                    risk_amount = abs(entry_price - stop_loss) * position_data['size']
                    reward_amount = abs(take_profit - entry_price) * position_data['size']
                    risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
                    
                    if risk_reward < 1.2:
                        continue
                    
                    # Create opportunity
                    opportunity = {
                        'symbol': symbol,
                        'signal': prediction['signal'],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': position_data['size'],
                        'confidence': prediction['confidence'],
                        'reason': prediction['reason'],
                        'risk_amount': position_data['risk_amount'],
                        'expected_profit': net_profit,
                        'risk_reward': risk_reward,
                        'fees': position_data['total_fees'],
                        'spread_pct': market_data['spread_pct'],
                        'timestamp': datetime.now()
                    }
                    
                    opportunities.append(opportunity)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Sort and execute best opportunities
            opportunities.sort(key=lambda x: (x['confidence'], x['risk_reward']), reverse=True)
            
            for opportunity in opportunities[:2]:  # Max 2 per cycle
                if len(self.risk_manager.open_trades) < StableConfig.MAX_OPEN_TRADES:
                    await self.execute_trade(opportunity)
                    await asyncio.sleep(2)  # Small delay between executions
                    
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def execute_trade(self, opportunity: Dict):
        """Execute a trade"""
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            
            # Log trade
            extra = {'trade_type': 'trade'}
            logger.log(21, f"🎯 Executing: {signal} {symbol}", extra=extra)
            logger.log(21, f"   Entry: ${opportunity['entry_price']:.4f}", extra=extra)
            logger.log(21, f"   Stop: ${opportunity['stop_loss']:.4f}", extra=extra)
            logger.log(21, f"   Target: ${opportunity['take_profit']:.4f}", extra=extra)
            logger.log(21, f"   Size: {opportunity['size']:.6f}", extra=extra)
            logger.log(21, f"   Confidence: {opportunity['confidence']:.1%}", extra=extra)
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'signal': signal,
                'entry_price': opportunity['entry_price'],
                'stop_loss': opportunity['stop_loss'],
                'take_profit': opportunity['take_profit'],
                'size': opportunity['size'],
                'entry_time': datetime.now(),
                'status': 'open',
                'confidence': opportunity['confidence'],
                'reason': opportunity['reason'],
                'fees_paid': opportunity['fees'] / 2,
                'risk_amount': opportunity['risk_amount']
            }
            
            # Update capital (paper trading)
            trade_value = opportunity['size'] * opportunity['entry_price']
            self.risk_manager.capital -= trade_value + (opportunity['fees'] / 2)
            
            # Add to open trades
            self.risk_manager.open_trades.append(trade)
            
            # Start monitoring
            asyncio.create_task(self.monitor_trade(trade))
            
            # Send Telegram notification
            if self.telegram.enabled:
                telegram_msg = f"""
⚡ <b>TRADE EXECUTED</b>

📊 <b>Symbol:</b> {symbol}
🎯 <b>Signal:</b> {signal}
💰 <b>Entry Price:</b> ${opportunity['entry_price']:.4f}
📈 <b>Target:</b> ${opportunity['take_profit']:.4f}
🛑 <b>Stop Loss:</b> ${opportunity['stop_loss']:.4f}
📊 <b>Size:</b> {opportunity['size']:.6f}
🎯 <b>Confidence:</b> {opportunity['confidence']:.1%}
💰 <b>Capital:</b> ${self.risk_manager.capital:.2f}
                """
                await self.telegram.send_message(telegram_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def monitor_trade(self, trade: Dict):
        """Monitor and manage open trade"""
        try:
            symbol = trade['symbol']
            start_time = trade['entry_time']
            
            while trade['status'] == 'open':
                # Check max duration
                duration = (datetime.now() - start_time).total_seconds()
                if duration > StableConfig.MAX_TRADE_DURATION:
                    await self.close_trade(trade, 'timeout')
                    break
                
                # Get current price
                market_data = self.exchange.get_market_data(symbol)
                if not market_data:
                    await asyncio.sleep(2)
                    continue
                
                current_price = market_data['last']
                
                # Check exit conditions
                exit_price = None
                exit_reason = None
                
                if trade['signal'] == 'BUY':
                    if current_price >= trade['take_profit']:
                        exit_price = trade['take_profit']
                        exit_reason = 'take_profit'
                    elif current_price <= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = 'stop_loss'
                else:  # SELL
                    if current_price <= trade['take_profit']:
                        exit_price = trade['take_profit']
                        exit_reason = 'take_profit'
                    elif current_price >= trade['stop_loss']:
                        exit_price = trade['stop_loss']
                        exit_reason = 'stop_loss'
                
                # Exit if condition met
                if exit_price and exit_reason:
                    await self.close_trade(trade, exit_reason, exit_price)
                    break
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error monitoring trade {trade['symbol']}: {e}")
    
    async def close_trade(self, trade: Dict, reason: str, exit_price: Optional[float] = None):
        """Close a trade"""
        try:
            symbol = trade['symbol']
            
            # Get exit price if not provided
            if exit_price is None:
                market_data = self.exchange.get_market_data(symbol)
                exit_price = market_data['last'] if market_data else trade['entry_price']
            
            # Calculate P&L
            if trade['signal'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['size']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['size']
            
            # Deduct fees
            total_fees = trade['fees_paid'] * 2
            net_pnl = pnl - total_fees
            
            # Update capital
            self.risk_manager.capital += trade['size'] * exit_price - trade['fees_paid']
            
            # Update performance
            self.risk_manager.update_performance(net_pnl)
            
            # Update best/worst trade
            if net_pnl > self.best_trade:
                self.best_trade = net_pnl
            if net_pnl < self.worst_trade:
                self.worst_trade = net_pnl
            
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
            if net_pnl > 0:
                extra = {'trade_type': 'profit'}
                logger.log(22, f"💰 Profit: {trade['signal']} {symbol}", extra=extra)
            else:
                extra = {'trade_type': 'loss'}
                logger.log(23, f"📉 Loss: {trade['signal']} {symbol}", extra=extra)
            
            logger.log(21, f"   P&L: ${net_pnl:+.4f} ({trade['pnl_pct']:+.2f}%)", extra=extra)
            logger.log(21, f"   Duration: {trade['duration']:.1f}s", extra=extra)
            logger.log(21, f"   Win Rate: {self.risk_manager.win_rate:.1%}", extra=extra)
            
            # Send Telegram notification for significant trades
            if self.telegram.enabled and abs(net_pnl) > 0.5:
                if net_pnl > 0:
                    telegram_msg = f"""
💰 <b>PROFIT!</b>

📊 <b>Symbol:</b> {symbol}
🎯 <b>Signal:</b> {trade['signal']}
💰 <b>P&L:</b> +${net_pnl:.4f} (+{trade['pnl_pct']:.2f}%)
⏱️ <b>Duration:</b> {trade['duration']:.1f}s
📈 <b>Reason:</b> {reason}
💰 <b>Capital:</b> ${self.risk_manager.capital:.2f}
📊 <b>Win Rate:</b> {self.risk_manager.win_rate:.1%}
                    """
                else:
                    telegram_msg = f"""
📉 <b>LOSS</b>

📊 <b>Symbol:</b> {symbol}
🎯 <b>Signal:</b> {trade['signal']}
💰 <b>P&L:</b> -${abs(net_pnl):.4f} (-{abs(trade['pnl_pct']):.2f}%)
⏱️ <b>Duration:</b> {trade['duration']:.1f}s
📈 <b>Reason:</b> {reason}
💰 <b>Capital:</b> ${self.risk_manager.capital:.2f}
📊 <b>Win Rate:</b> {self.risk_manager.win_rate:.1%}
                    """
                await self.telegram.send_message(telegram_msg)
            
        except Exception as e:
            logger.error(f"Error closing trade: {e}")
    
    async def log_status(self):
        """Log current status"""
        status = f"""
📊 STATUS - Cycle {self.cycle_count}
{'='*40}
💰 Capital: ${self.risk_manager.capital:.2f}
📈 Total P&L: ${self.risk_manager.total_pnl:+.2f}
🎯 Trades: {self.risk_manager.total_trades}
📊 Win Rate: {self.risk_manager.win_rate:.1%}
⚡ Open Trades: {len(self.risk_manager.open_trades)}
⏰ Hourly P&L: ${self.risk_manager.hourly_pnl:+.2f}
📅 Daily P&L: ${self.risk_manager.daily_pnl:+.2f}
🎯 Best Trade: ${self.best_trade:+.4f}
📉 Worst Trade: ${self.worst_trade:+.4f}
{'='*40}
        """
        logger.info(status)
    
    async def check_hourly_performance(self):
        """Check hourly performance"""
        if datetime.now() - self.risk_manager.hourly_start >= timedelta(hours=1):
            hourly_msg = f"""
⏰ HOURLY REPORT
{'='*40}
💰 Capital: ${self.risk_manager.capital:.2f}
📈 Hourly P&L: ${self.risk_manager.hourly_pnl:+.2f}
🎯 Trades: {self.risk_manager.total_trades}
📊 Win Rate: {self.risk_manager.win_rate:.1%}
{'='*40}
            """
            logger.info(hourly_msg)
            
            # Send Telegram hourly report
            if self.telegram.enabled:
                telegram_msg = f"""
⏰ <b>HOURLY REPORT</b>

💰 <b>Capital:</b> ${self.risk_manager.capital:.2f}
📈 <b>Hourly P&L:</b> ${self.risk_manager.hourly_pnl:+.2f}
🎯 <b>Total Trades:</b> {self.risk_manager.total_trades}
📊 <b>Win Rate:</b> {self.risk_manager.win_rate:.1%}
⏱️ <b>Uptime:</b> {(datetime.now() - self.start_time).total_seconds()/3600:.1f}h
                """
                await self.telegram.send_message(telegram_msg)
            
            # Reset hourly counter
            self.risk_manager.hourly_pnl = 0
            self.risk_manager.hourly_start = datetime.now()
    
    async def stop(self):
        """Stop the bot"""
        self.running = False
        
        # Close all open trades
        for trade in self.risk_manager.open_trades[:]:
            if trade['status'] == 'open':
                await self.close_trade(trade, 'shutdown')
        
        # Calculate final statistics
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        final_stats = f"""
{'='*60}
🏁 BOT STOPPED
{'='*60}
⏱️  Runtime: {hours}h {minutes}m {seconds}s
💰 Final Capital: ${self.risk_manager.capital:.2f}
📈 Total P&L: ${self.risk_manager.total_pnl:+.2f}
🎯 Total Trades: {self.risk_manager.total_trades}
📊 Win Rate: {self.risk_manager.win_rate:.1%}
⚡ Trades/Hour: {self.risk_manager.total_trades / max(uptime.total_seconds()/3600, 0.1):.1f}
📅 Daily P&L: ${self.risk_manager.daily_pnl:+.2f}
🎯 Best Trade: ${self.best_trade:+.4f}
📉 Worst Trade: ${self.worst_trade:+.4f}
{'='*60}
        """
        
        logger.info(final_stats)
        
        # Send final Telegram message
        if self.telegram.enabled:
            final_telegram_msg = f"""
🏁 <b>BOT STOPPED</b>

⏱️ <b>Runtime:</b> {hours}h {minutes}m {seconds}s
💰 <b>Final Capital:</b> ${self.risk_manager.capital:.2f}
📈 <b>Total P&L:</b> ${self.risk_manager.total_pnl:+.2f}
🎯 <b>Total Trades:</b> {self.risk_manager.total_trades}
📊 <b>Win Rate:</b> {self.risk_manager.win_rate:.1%}
⚡ <b>Trades/Hour:</b> {self.risk_manager.total_trades / max(uptime.total_seconds()/3600, 0.1):.1f}
📅 <b>Daily P&L:</b> ${self.risk_manager.daily_pnl:+.2f}

Thank you for using Stable Scalping Bot!
            """
            await self.telegram.send_message(final_telegram_msg)

# ============================================================================
# HEALTH CHECK SERVER
# ============================================================================

async def start_health_server():
    """Start health check server"""
    try:
        from aiohttp import web
        
        app = web.Application()
        
        async def health_handler(request):
            return web.Response(text='OK', status=200)
        
        async def status_handler(request):
            stats = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'service': 'stable-scalping-bot',
                'version': '1.0'
            }
            return web.json_response(stats)
        
        app.router.add_get('/health', health_handler)
        app.router.add_get('/status', status_handler)
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
    print("🤖 STABLE SCALPING BOT - FIXED VERSION")
    print("="*60)
    print(f"💰 Capital: ${StableConfig.INITIAL_CAPITAL}")
    print(f"🎯 Hourly Target: ${StableConfig.INITIAL_CAPITAL * StableConfig.HOURLY_TARGET:.2f}")
    print(f"📊 Monitoring {len(StableConfig.SYMBOLS)} Cryptocurrencies")
    print(f"⚡ Timeframe: {StableConfig.PRIMARY_TIMEFRAME}")
    print(f"🎯 Take Profit: {StableConfig.TAKE_PROFIT_PCT:.3%}")
    print(f"🛑 Stop Loss: {StableConfig.STOP_LOSS_PCT:.3%}")
    print("="*60)
    print()
    
    # Check Telegram configuration
    if StableConfig.TELEGRAM_ENABLED:
        if StableConfig.TELEGRAM_BOT_TOKEN and StableConfig.TELEGRAM_CHAT_ID:
            print(f"✅ Telegram configured")
        else:
            print(f"⚠️ Telegram disabled: Missing configuration")
    else:
        print(f"ℹ️ Telegram disabled by configuration")
    
    # Check dependencies
    try:
        import ccxt
        import pandas
        import numpy
        print("✅ Core dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install ccxt pandas numpy")
        return
    
    # Start health server
    health_server = await start_health_server()
    
    # Create and run bot
    bot = StableScalpingBot()
    
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
