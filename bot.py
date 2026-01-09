# scalping_bot.py
"""
Complete Scalping Trading Bot for Binance with Telegram Integration
Author: Trading Bot System
Version: 1.0
"""

import os
import time
import json
import sqlite3
import asyncio
import logging
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import ccxt
import ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackContext, ContextTypes
from dotenv import load_dotenv
import threading
import signal
import sys

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All configuration settings in one place"""
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID',  '5665906172')
    
    # Exchange Configuration
    EXCHANGE = 'binance'
    SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    TIMEFRAME = '1m'
    
    # Risk Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.02  # 2%
    MAX_DAILY_LOSS = 0.05  # 5%
    MAX_POSITION_SIZE = 0.95  # 95% of capital
    MAX_OPEN_TRADES = 3
    
    # Fees & Spread
    MAKER_FEE = 0.001  # 0.1%
    TAKER_FEE = 0.001  # 0.1%
    MIN_SPREAD_PCT = 0.0002  # 0.02%
    MAX_SPREAD_PCT = 0.001  # 0.1%
    SLIPPAGE_PCT = 0.0005  # 0.05%
    
    # Strategy Parameters
    TAKE_PROFIT_PCT = 0.003  # 0.3%
    STOP_LOSS_PCT = 0.002  # 0.2%
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    BB_PERIOD = 20
    BB_STD = 2.0
    VOLUME_MA_PERIOD = 20
    MIN_VOLUME_RATIO = 1.5
    
    # Execution
    TRADE_COOLDOWN = 30  # seconds
    CHECK_INTERVAL = 30  # seconds
    MAX_TRADE_DURATION = 600  # seconds (10 minutes)
    
    # Bot Settings
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = 'trading_bot.db'
    PAPER_TRADING = True  # Set to False for real trading
    
# ============================================================================
# DATA CLASSES & ENUMS
# ============================================================================

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradeSignal:
    symbol: str
    signal: SignalType
    price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    timestamp: datetime

@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    entry_time: datetime
    exit_time: Optional[datetime]
    fees: float
    pnl: float = 0
    pnl_pct: float = 0
    status: str = 'open'
    reason: Optional[str] = None

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    spread_pct: float
    timestamp: datetime

# ============================================================================
# LOGGING SETUP
# ============================================================================

class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors"""
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Setup logging
logger = logging.getLogger('ScalpingBot')
logger.setLevel(getattr(logging, Config.LOG_LEVEL))

# Console handler
ch = logging.StreamHandler()
ch.setLevel(getattr(logging, Config.LOG_LEVEL))
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# File handler
fh = logging.FileHandler('scalping_bot.log')
fh.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(file_formatter)
logger.addHandler(fh)

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class Database:
    """SQLite database manager"""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    fees REAL NOT NULL,
                    pnl REAL,
                    pnl_pct REAL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    status TEXT NOT NULL,
                    reason TEXT,
                    metadata TEXT
                )
            ''')
            
            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    price REAL NOT NULL,
                    confidence REAL,
                    reason TEXT,
                    timestamp TIMESTAMP NOT NULL,
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    best_trade REAL,
                    worst_trade REAL,
                    avg_trade_duration TEXT
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized")
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades 
                    (symbol, side, entry_price, exit_price, size, fees, pnl, pnl_pct, 
                     entry_time, exit_time, status, reason, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['entry_price'],
                    trade_data.get('exit_price'),
                    trade_data['size'],
                    trade_data['fees'],
                    trade_data.get('pnl'),
                    trade_data.get('pnl_pct'),
                    trade_data['entry_time'],
                    trade_data.get('exit_time'),
                    trade_data['status'],
                    trade_data.get('reason'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                conn.commit()
                logger.debug(f"Trade saved to database: {trade_data['symbol']}")
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades 
                    ORDER BY entry_time DESC 
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching trades from database: {e}")
            return []
    
    def get_daily_stats(self, date: datetime = None) -> Dict:
        """Get daily statistics"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades 
                    WHERE DATE(entry_time) = ? 
                    AND status = 'closed'
                ''', (date_str,))
                
                trades = cursor.fetchall()
                
                if not trades:
                    return {}
                
                # Calculate statistics
                winning_trades = [t for t in trades if t[7] and t[7] > 0]
                
                return {
                    'date': date_str,
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'win_rate': len(winning_trades) / len(trades) if trades else 0,
                    'total_pnl': sum(t[7] or 0 for t in trades),
                    'best_trade': max((t[7] or 0 for t in trades), default=0),
                    'worst_trade': min((t[7] or 0 for t in trades), default=0)
                }
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return {}

# ============================================================================
# EXCHANGE HANDLER
# ============================================================================

class ExchangeHandler:
    """Handle all exchange operations using CCXT"""
    
    def __init__(self, exchange_id: str = Config.EXCHANGE):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.symbols = Config.SYMBOLS
        logger.info(f"Exchange handler initialized for {exchange_id}")
    
    def get_ticker(self, symbol: str) -> Optional[MarketData]:
        """Get real-time ticker data"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            bid = ticker['bid']
            ask = ticker['ask']
            last = ticker['last']
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=ticker['quoteVolume'] or 0,
                spread_pct=spread_pct,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = Config.TIMEFRAME, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data"""
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
    
    def calculate_fees(self, symbol: str, amount: float, price: float, is_buy: bool) -> Dict:
        """Calculate trading fees with spread consideration"""
        ticker = self.get_ticker(symbol)
        if not ticker:
            return {'total_fee': 0, 'effective_price': price}
        
        # Consider spread for execution price
        if is_buy:
            execution_price = ticker.ask  # Buying at ask price
        else:
            execution_price = ticker.bid  # Selling at bid price
        
        # Calculate fees
        fee_rate = Config.TAKER_FEE
        fee_amount = amount * execution_price * fee_rate
        
        # Consider slippage
        slippage = execution_price * Config.SLIPPAGE_PCT
        effective_price = execution_price + (slippage if is_buy else -slippage)
        
        return {
            'execution_price': execution_price,
            'effective_price': effective_price,
            'fee_amount': fee_amount,
            'fee_rate': fee_rate,
            'spread_pct': ticker.spread_pct,
            'slippage': slippage
        }
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """Get order book for spread analysis"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def check_market_conditions(self, symbol: str) -> Dict:
        """Check if market conditions are suitable for scalping"""
        ticker = self.get_ticker(symbol)
        if not ticker:
            return {'suitable': False, 'reason': 'No ticker data'}
        
        # Check spread
        if ticker.spread_pct > Config.MAX_SPREAD_PCT:
            return {'suitable': False, 'reason': f'Spread too high: {ticker.spread_pct:.4%}'}
        
        # Check minimum spread
        if ticker.spread_pct < Config.MIN_SPREAD_PCT:
            return {'suitable': False, 'reason': f'Spread too low for profit: {ticker.spread_pct:.4%}'}
        
        # Check volume
        if ticker.volume < 1000000:  # Minimum $1M volume
            return {'suitable': False, 'reason': f'Volume too low: ${ticker.volume:,.0f}'}
        
        # Check order book depth
        orderbook = self.get_order_book(symbol, 5)
        if orderbook:
            bid_depth = sum([bid[1] for bid in orderbook['bids'][:3]])
            ask_depth = sum([ask[1] for ask in orderbook['asks'][:3]])
            if bid_depth < 1 or ask_depth < 1:
                return {'suitable': False, 'reason': 'Insufficient order book depth'}
        
        return {
            'suitable': True,
            'spread': ticker.spread_pct,
            'volume': ticker.volume,
            'bid': ticker.bid,
            'ask': ticker.ask
        }

# ============================================================================
# TRADING STRATEGY
# ============================================================================

class ScalpingStrategy:
    """Mean reversion scalping strategy"""
    
    def __init__(self):
        self.signals_history = []
        logger.info("Scalping strategy initialized")
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSignal]:
        """Analyze market data and generate signals"""
        if len(df) < 50:
            return None
        
        # Calculate indicators
        df = self._add_indicators(df)
        
        # Get latest data
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check multiple conditions
        signals = []
        
        # RSI strategy
        if latest['rsi'] < Config.RSI_OVERSOLD:
            signals.append(('RSI Oversold', 0.3))
        elif latest['rsi'] > Config.RSI_OVERBOUGHT:
            signals.append(('RSI Overbought', 0.3))
        
        # Bollinger Bands strategy
        if latest['close'] < latest['bb_lower']:
            signals.append(('BB Lower Band', 0.4))
        elif latest['close'] > latest['bb_upper']:
            signals.append(('BB Upper Band', 0.4))
        
        # Volume spike
        volume_ma = df['volume'].rolling(window=Config.VOLUME_MA_PERIOD).mean().iloc[-1]
        if latest['volume'] > volume_ma * Config.MIN_VOLUME_RATIO:
            signals.append(('Volume Spike', 0.2))
        
        # Price action - crossing BB middle
        bb_middle = df['close'].rolling(window=Config.BB_PERIOD).mean().iloc[-2]
        if latest['close'] > bb_middle and prev['close'] <= bb_middle:
            signals.append(('Above BB Middle', 0.3))
        
        # Generate final signal
        if not signals:
            return None
        
        # Calculate weighted signal
        total_weight = sum([weight for _, weight in signals])
        avg_confidence = total_weight / len(signals)
        
        # Determine signal type
        buy_conditions = ['RSI Oversold', 'BB Lower Band', 'Above BB Middle']
        sell_conditions = ['RSI Overbought', 'BB Upper Band']
        
        buy_score = sum([weight for reason, weight in signals if reason in buy_conditions])
        sell_score = sum([weight for reason, weight in signals if reason in sell_conditions])
        
        if buy_score > sell_score and buy_score > 0.3:
            signal_type = SignalType.BUY
            price = latest['close']
            stop_loss = price * (1 - Config.STOP_LOSS_PCT)
            take_profit = price * (1 + Config.TAKE_PROFIT_PCT)
            reason = ', '.join([r for r, _ in signals if r in buy_conditions])
        elif sell_score > buy_score and sell_score > 0.3:
            signal_type = SignalType.SELL
            price = latest['close']
            stop_loss = price * (1 + Config.STOP_LOSS_PCT)
            take_profit = price * (1 - Config.TAKE_PROFIT_PCT)
            reason = ', '.join([r for r, _ in signals if r in sell_conditions])
        else:
            return None
        
        # Ensure minimum profit after fees
        potential_profit = abs(take_profit - price)
        if potential_profit / price < Config.MAKER_FEE * 3:  # Need 3x fees as profit
            return None
        
        signal = TradeSignal(
            symbol=symbol,
            signal=signal_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=avg_confidence,
            reason=reason,
            timestamp=latest.name
        )
        
        self.signals_history.append(signal)
        return signal
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=Config.RSI_PERIOD).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=Config.BB_PERIOD, window_dev=Config.BB_STD)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=Config.VOLUME_MA_PERIOD).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        return df
    
    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = capital * Config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        # Apply max position size limit
        max_size = capital * Config.MAX_POSITION_SIZE / entry_price
        position_size = min(position_size, max_size)
        
        # Consider fees
        estimated_fees = position_size * entry_price * Config.TAKER_FEE * 2
        if estimated_fees > risk_amount * 0.1:
            position_size *= 0.9
        
        return round(position_size, 6)  # Round to 6 decimal places

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Manage trading risk and position sizing"""
    
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.daily_pnl = 0
        self.daily_trades = 0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info("Risk manager initialized")
    
    def can_trade(self, symbol: str, capital: float) -> Dict:
        """Check if trading is allowed"""
        # Check daily loss limit
        daily_loss_limit = Config.INITIAL_CAPITAL * Config.MAX_DAILY_LOSS
        if self.daily_pnl <= -daily_loss_limit:
            return {
                'allowed': False,
                'reason': f'Daily loss limit reached: ${self.daily_pnl:.2f}'
            }
        
        # Check open trades
        open_trades = [t for t in self.trades if t.status == 'open']
        if len(open_trades) >= Config.MAX_OPEN_TRADES:
            return {
                'allowed': False,
                'reason': f'Max open trades reached: {len(open_trades)}'
            }
        
        # Check if same symbol already traded recently
        recent_trades = [t for t in self.trades 
                        if t.symbol == symbol 
                        and t.entry_time > datetime.now() - timedelta(seconds=Config.TRADE_COOLDOWN)]
        if recent_trades:
            return {
                'allowed': False,
                'reason': f'Recent trade on {symbol} within cooldown period'
            }
        
        return {'allowed': True, 'reason': 'OK'}
    
    def record_trade(self, trade: TradeRecord):
        """Record a new trade"""
        self.trades.append(trade)
        self.daily_trades += 1
        logger.info(f"Trade recorded: {trade.symbol} {trade.side} at ${trade.entry_price:.2f}")
    
    def update_trade(self, symbol: str, exit_price: float, exit_time: datetime, reason: str = 'manual'):
        """Update trade when closed"""
        open_trades = [t for t in self.trades if t.symbol == symbol and t.status == 'open']
        
        if not open_trades:
            return None
        
        trade = open_trades[0]
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = 'closed'
        trade.reason = reason
        
        # Calculate P&L with fees
        if trade.side == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.size - trade.fees
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.size - trade.fees
        
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size)
        
        # Update daily P&L
        self.daily_pnl += trade.pnl
        
        logger.info(f"Trade closed: {symbol} {trade.side} P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        return trade
    
    def get_daily_stats(self) -> Dict:
        """Get daily trading statistics"""
        today_trades = [t for t in self.trades if t.entry_time.date() == datetime.now().date()]
        
        if not today_trades:
            return {}
        
        closed_trades = [t for t in today_trades if t.status == 'closed']
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # Calculate average trade duration
        durations = []
        for trade in closed_trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_trades': len(today_trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'pnl_pct': total_pnl / Config.INITIAL_CAPITAL,
            'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
            'best_trade': max((t.pnl for t in closed_trades), default=0),
            'worst_trade': min((t.pnl for t in closed_trades), default=0),
            'avg_duration': f"{avg_duration:.0f}s",
            'daily_pnl': self.daily_pnl
        }

# ============================================================================
# TELEGRAM BOT
# ============================================================================

class TelegramBot:
    """Telegram bot for alerts and control"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.application = None
        self.bot_running = False
        logger.info("Telegram bot initialized")
    
    async def start(self):
        """Start the Telegram bot"""
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat ID not configured")
            return
        
        try:
            self.application = Application.builder().token(self.token).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("trades", self.trades_command))
            self.application.add_handler(CommandHandler("pnl", self.pnl_command))
            self.application.add_handler(CommandHandler("stop", self.stop_command))
            self.application.add_handler(CommandHandler("resume", self.resume_command))
            
            # Start bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            self.bot_running = True
            logger.info("Telegram bot started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    async def stop(self):
        """Stop the Telegram bot"""
        if self.application:
            await self.application.stop()
            self.bot_running = False
            logger.info("Telegram bot stopped")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = """
🤖 *Scalping Bot Activated* 🤖

*Available Commands:*
/status - Bot status & metrics
/trades - Recent trades
/pnl - Profit & Loss summary
/stop - Pause trading
/resume - Resume trading

*Alerts will be sent for:*
✅ Trade signals
📊 Trade executions
💰 Profit/Loss updates
⚠️ System warnings

Bot is now monitoring markets...
        """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        # This would be populated with real data from the main bot
        status_msg = """
📊 *Bot Status*

*Exchange:* Binance
*Status:* ✅ Active
*Mode:* Paper Trading
*Symbols:* BTC/USDT, ETH/USDT, BNB/USDT

*Risk Settings:*
Capital: $100.00
Risk/Trade: 2%
Daily Loss Limit: 5%

*Last Signal:* No recent signals
        """
        await update.message.reply_text(status_msg, parse_mode='Markdown')
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        # This would show real trades from database
        trades_msg = """
📈 *Recent Trades*

*No trades executed yet.*

*Note:* Run the bot to see trade history.
        """
        await update.message.reply_text(trades_msg, parse_mode='Markdown')
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        pnl_msg = """
💰 *Profit & Loss Report*

*No trading activity yet.*

*Note:* Start trading to see P&L reports.
        """
        await update.message.reply_text(pnl_msg, parse_mode='Markdown')
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        # This would pause the trading bot
        await update.message.reply_text("⏸️ Trading paused (not implemented in this example)")
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        # This would resume the trading bot
        await update.message.reply_text("▶️ Trading resumed (not implemented in this example)")
    
    async def send_alert(self, alert_type: str, message: str, data: Dict = None):
        """Send alert to Telegram"""
        if not self.bot_running:
            return
        
        try:
            emoji = {
                'trade': '📊',
                'signal': '🚨',
                'profit': '💰',
                'loss': '📉',
                'error': '⚠️',
                'info': 'ℹ️',
                'warning': '⚠️'
            }.get(alert_type, '📢')
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"{emoji} *{alert_type.upper()} Alert* [{timestamp}]\n\n{message}"
            
            if data:
                formatted_msg += f"\n\n*Details:*\n"
                for key, value in data.items():
                    formatted_msg += f"• {key}: {value}\n"
            
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_msg,
                parse_mode='Markdown'
            )
            logger.info(f"Telegram alert sent: {alert_type}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    async def send_trade_signal(self, signal_data: Dict):
        """Send trade signal alert"""
        signal = signal_data.get('signal')
        symbol = signal_data.get('symbol')
        price = signal_data.get('price')
        confidence = signal_data.get('confidence')
        
        emoji = '🟢' if signal == 'BUY' else '🔴'
        message = f"{emoji} *TRADE SIGNAL*\n\n"
        message += f"*Symbol:* {symbol}\n"
        message += f"*Signal:* {signal}\n"
        message += f"*Price:* ${price:,.2f}\n"
        message += f"*Confidence:* {confidence:.1%}\n"
        message += f"*Reason:* {signal_data.get('reason', 'N/A')}"
        
        await self.send_alert('signal', message)
    
    async def send_trade_execution(self, trade_data: Dict):
        """Send trade execution alert"""
        message = f"⚡ *TRADE EXECUTED*\n\n"
        message += f"*Symbol:* {trade_data['symbol']}\n"
        message += f"*Side:* {trade_data['side']}\n"
        message += f"*Entry:* ${trade_data['entry_price']:,.2f}\n"
        message += f"*Size:* {trade_data['size']:.6f}\n"
        message += f"*Stop Loss:* ${trade_data['stop_loss']:,.2f}\n"
        message += f"*Take Profit:* ${trade_data['take_profit']:,.2f}\n"
        message += f"*Fees:* ${trade_data['fees']:.4f}"
        
        await self.send_alert('trade', message)
    
    async def send_pnl_update(self, pnl_data: Dict):
        """Send P&L update"""
        pnl = pnl_data.get('pnl', 0)
        pnl_pct = pnl_data.get('pnl_pct', 0)
        
        if pnl > 0:
            emoji = '💰'
            alert_type = 'profit'
            message = f"*PROFIT!* +${pnl:.2f} ({pnl_pct:.2%})"
        else:
            emoji = '📉'
            alert_type = 'loss'
            message = f"*LOSS* -${abs(pnl):.2f} ({abs(pnl_pct):.2%})"
        
        message += f"\n\n*Trade Details:*\n"
        message += f"Symbol: {pnl_data.get('symbol')}\n"
        message += f"Side: {pnl_data.get('side')}\n"
        message += f"Duration: {pnl_data.get('duration')}\n"
        message += f"Exit Reason: {pnl_data.get('reason')}"
        
        await self.send_alert(alert_type, message)
    
    async def send_daily_summary(self, summary_data: Dict):
        """Send daily summary"""
        total_pnl = summary_data.get('total_pnl', 0)
        
        message = f"📊 *DAILY SUMMARY*\n\n"
        message += f"*Date:* {datetime.now().strftime('%Y-%m-%d')}\n"
        message += f"*Total Trades:* {summary_data.get('total_trades', 0)}\n"
        message += f"*Win Rate:* {summary_data.get('win_rate', 0):.1%}\n"
        message += f"*Total P&L:* ${total_pnl:+.2f}\n"
        message += f"*P&L %:* {summary_data.get('pnl_pct', 0):+.2%}\n"
        message += f"*Best Trade:* ${summary_data.get('best_trade', 0):.2f}\n"
        message += f"*Worst Trade:* ${summary_data.get('worst_trade', 0):.2f}\n"
        message += f"*Avg Trade Duration:* {summary_data.get('avg_duration', 'N/A')}"
        
        await self.send_alert('info', message)

# ============================================================================
# MAIN SCALPING BOT
# ============================================================================

class ScalpingBot:
    """Main scalping bot class"""
    
    def __init__(self):
        self.config = Config()
        self.capital = Config.INITIAL_CAPITAL
        self.running = False
        
        # Initialize components
        self.exchange = ExchangeHandler()
        self.strategy = ScalpingStrategy()
        self.risk_manager = RiskManager()
        self.database = Database()
        self.telegram_bot = TelegramBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.total_pnl = 0
        
        logger.info(f"Scalping Bot initialized with ${self.capital:.2f} capital")
        logger.info(f"Trading symbols: {', '.join(Config.SYMBOLS)}")
        logger.info(f"Paper trading mode: {Config.PAPER_TRADING}")
    
    async def start(self):
        """Start the trading bot"""
        self.running = True
        logger.info("Starting Scalping Bot...")
        
        # Start Telegram bot
        await self.telegram_bot.start()
        
        # Send startup message
        if self.telegram_bot.bot_running:
            await self.telegram_bot.send_alert(
                'info',
                f"🚀 *Scalping Bot Started*\n\n"
                f"*Capital:* ${self.capital:.2f}\n"
                f"*Symbols:* {', '.join(Config.SYMBOLS)}\n"
                f"*Strategy:* Mean Reversion Scalping\n"
                f"*Mode:* {'Paper Trading' if Config.PAPER_TRADING else 'Live Trading'}\n"
                f"*Risk/Trade:* {Config.RISK_PER_TRADE:.1%}\n"
                f"*Daily Loss Limit:* {Config.MAX_DAILY_LOSS:.1%}"
            )
        
        try:
            # Main trading loop
            while self.running:
                try:
                    await self.check_markets()
                    
                    # Sleep before next scan
                    await asyncio.sleep(Config.CHECK_INTERVAL)
                    
                    # Log heartbeat
                    self._log_heartbeat()
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    if self.telegram_bot.bot_running:
                        await self.telegram_bot.send_alert(
                            'error',
                            "Error in main loop",
                            {'error': str(e)}
                        )
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            if self.telegram_bot.bot_running:
                await self.telegram_bot.send_alert(
                    'error',
                    "Bot crashed!",
                    {'error': str(e)}
                )
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot"""
        self.running = False
        
        # Send shutdown message
        if self.telegram_bot.bot_running:
            uptime = datetime.now() - self.start_time
            await self.telegram_bot.send_alert(
                'info',
                f"🛑 *Scalping Bot Stopped*\n\n"
                f"*Uptime:* {str(uptime).split('.')[0]}\n"
                f"*Total Trades:* {self.total_trades}\n"
                f"*Final Capital:* ${self.capital:.2f}\n"
                f"*Total P&L:* ${self.total_pnl:+.2f}"
            )
        
        # Stop Telegram bot
        await self.telegram_bot.stop()
        
        logger.info("Scalping Bot shutdown complete")
    
    async def check_markets(self):
        """Check all markets for trading opportunities"""
        logger.debug("Scanning markets for opportunities...")
        
        for symbol in Config.SYMBOLS:
            try:
                # Check market conditions
                market_cond = self.exchange.check_market_conditions(symbol)
                if not market_cond['suitable']:
                    logger.debug(f"Market not suitable for {symbol}: {market_cond['reason']}")
                    continue
                
                # Get market data
                df = self.exchange.get_ohlcv(symbol, Config.TIMEFRAME, limit=100)
                if df is None or len(df) < 50:
                    continue
                
                # Generate signal
                signal = self.strategy.analyze(df, symbol)
                if signal is None:
                    continue
                
                # Check risk management
                risk_check = self.risk_manager.can_trade(symbol, self.capital)
                if not risk_check['allowed']:
                    logger.debug(f"Trade not allowed for {symbol}: {risk_check['reason']}")
                    continue
                
                # Calculate position size
                position_size = self.strategy.calculate_position_size(
                    self.capital,
                    signal.price,
                    signal.stop_loss
                )
                
                if position_size <= 0:
                    continue
                
                # Calculate fees
                is_buy = signal.signal == SignalType.BUY
                fees_data = self.exchange.calculate_fees(
                    symbol,
                    position_size,
                    signal.price,
                    is_buy
                )
                
                # Create trade record
                trade = TradeRecord(
                    symbol=symbol,
                    side=signal.signal.value,
                    entry_price=fees_data['effective_price'],
                    exit_price=None,
                    size=position_size,
                    entry_time=datetime.now(),
                    exit_time=None,
                    fees=fees_data['fee_amount'],
                    status='open'
                )
                
                # Record trade
                self.risk_manager.record_trade(trade)
                self.database.save_trade(asdict(trade))
                
                # Update capital (paper trading)
                trade_value = position_size * fees_data['effective_price']
                self.capital -= trade_value + fees_data['fee_amount']
                
                # Send Telegram alerts
                if self.telegram_bot.bot_running:
                    await self.telegram_bot.send_trade_signal({
                        'symbol': symbol,
                        'signal': signal.signal.value,
                        'price': signal.price,
                        'confidence': signal.confidence,
                        'reason': signal.reason
                    })
                    
                    await self.telegram_bot.send_trade_execution({
                        'symbol': symbol,
                        'side': signal.signal.value,
                        'entry_price': fees_data['effective_price'],
                        'size': position_size,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'fees': fees_data['fee_amount']
                    })
                
                logger.info(f"Trade executed: {signal.signal.value} {symbol} "
                          f"at ${fees_data['effective_price']:.2f}, "
                          f"size: {position_size:.6f}")
                
                # Monitor trade in background
                asyncio.create_task(self.monitor_trade(trade, signal))
                
                # Cooldown between trades
                await asyncio.sleep(Config.TRADE_COOLDOWN)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                if self.telegram_bot.bot_running:
                    await self.telegram_bot.send_alert(
                        'error',
                        f"Error processing {symbol}",
                        {'error': str(e)}
                    )
    
    async def monitor_trade(self, trade: TradeRecord, signal: TradeSignal):
        """Monitor open trade for exit conditions"""
        try:
            entry_time = trade.entry_time
            
            while trade.status == 'open' and self.running:
                # Get current price
                ticker = self.exchange.get_ticker(trade.symbol)
                if not ticker:
                    await asyncio.sleep(5)
                    continue
                
                current_price = ticker.last
                duration = datetime.now() - entry_time
                
                # Check exit conditions
                exit_reason = None
                exit_price = None
                
                if trade.side == 'BUY':
                    # Take profit
                    if current_price >= signal.take_profit:
                        exit_reason = 'Take Profit'
                        exit_price = signal.take_profit
                    
                    # Stop loss
                    elif current_price <= signal.stop_loss:
                        exit_reason = 'Stop Loss'
                        exit_price = signal.stop_loss
                    
                    # Time-based exit
                    elif duration.total_seconds() > Config.MAX_TRADE_DURATION:
                        exit_reason = 'Time Exit'
                        exit_price = current_price
                
                else:  # SELL
                    # Take profit
                    if current_price <= signal.take_profit:
                        exit_reason = 'Take Profit'
                        exit_price = signal.take_profit
                    
                    # Stop loss
                    elif current_price >= signal.stop_loss:
                        exit_reason = 'Stop Loss'
                        exit_price = signal.stop_loss
                    
                    # Time-based exit
                    elif duration.total_seconds() > Config.MAX_TRADE_DURATION:
                        exit_reason = 'Time Exit'
                        exit_price = current_price
                
                # Exit trade if condition met
                if exit_reason:
                    # Calculate exit fees
                    exit_fees = current_price * trade.size * Config.TAKER_FEE
                    
                    # Update trade
                    trade.exit_price = exit_price
                    trade.exit_time = datetime.now()
                    trade.status = 'closed'
                    trade.fees += exit_fees
                    
                    # Calculate P&L
                    if trade.side == 'BUY':
                        trade.pnl = (exit_price - trade.entry_price) * trade.size - trade.fees
                    else:
                        trade.pnl = (trade.entry_price - exit_price) * trade.size - trade.fees
                    
                    trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size)
                    
                    # Update capital
                    self.capital += trade.size * exit_price - exit_fees
                    self.total_trades += 1
                    self.total_pnl += trade.pnl
                    
                    # Save to database
                    trade_dict = asdict(trade)
                    trade_dict['reason'] = exit_reason
                    self.database.save_trade(trade_dict)
                    
                    # Send Telegram alert
                    if self.telegram_bot.bot_running:
                        await self.telegram_bot.send_pnl_update({
                            'symbol': trade.symbol,
                            'side': trade.side,
                            'pnl': trade.pnl,
                            'pnl_pct': trade.pnl_pct,
                            'duration': str(duration).split('.')[0],
                            'reason': exit_reason,
                            'entry_price': trade.entry_price,
                            'exit_price': exit_price
                        })
                    
                    logger.info(f"Trade closed: {trade.symbol} {trade.side} "
                              f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) "
                              f"Reason: {exit_reason}")
                    break
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring trade {trade.symbol}: {e}")
            if self.telegram_bot.bot_running:
                await self.telegram_bot.send_alert(
                    'error',
                    f"Error monitoring trade {trade.symbol}",
                    {'error': str(e)}
                )
    
    def _log_heartbeat(self):
        """Log heartbeat status"""
        open_trades = len([t for t in self.risk_manager.trades if t.status == 'open'])
        closed_trades = len([t for t in self.risk_manager.trades if t.status == 'closed'])
        
        logger.debug(f"Heartbeat - Capital: ${self.capital:.2f}, "
                    f"Open trades: {open_trades}, "
                    f"Closed trades: {closed_trades}, "
                    f"Total P&L: ${self.total_pnl:.2f}")
    
    async def send_daily_report(self):
        """Send daily trading report"""
        if not self.telegram_bot.bot_running:
            return
        
        stats = self.risk_manager.get_daily_stats()
        if stats:
            await self.telegram_bot.send_daily_summary(stats)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🤖 SCALPING TRADING BOT 🤖")
    print("="*60)
    print(f"Initial Capital: ${Config.INITIAL_CAPITAL}")
    print(f"Trading Symbols: {', '.join(Config.SYMBOLS)}")
    print(f"Paper Trading: {Config.PAPER_TRADING}")
    print(f"Risk per Trade: {Config.RISK_PER_TRADE:.1%}")
    print(f"Daily Loss Limit: {Config.MAX_DAILY_LOSS:.1%}")
    print("="*60 + "\n")
    
    # Create and run bot
    bot = ScalpingBot()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down bot...")
        bot.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Ensure bot is stopped
        bot.running = False

if __name__ == "__main__":
    # Check for required packages
    try:
        import ccxt
        import pandas
        import numpy
        import ta
        import telegram
        import dotenv
        import schedule
    except ImportError as e:
        print(f"\n❌ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install ccxt pandas numpy ta python-telegram-bot python-dotenv schedule")
        sys.exit(1)
    
    # Run the bot
    asyncio.run(main())







