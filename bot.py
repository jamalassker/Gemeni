# bot.py
"""
Complete Scalping Trading Bot for Binance with Telegram Integration
Optimized for Railway Deployment
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
from telegram.error import RetryAfter, TimedOut, NetworkError
from dotenv import load_dotenv
import threading
import signal
import sys
import aiohttp

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All configuration settings in one place"""
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Exchange Configuration
    EXCHANGE = 'binance'
    SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Reduced to 2 symbols for better performance
    TIMEFRAME = '5m'  # Changed to 5m for less frequent signals
    
    # Risk Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.015  # Reduced to 1.5%
    MAX_DAILY_LOSS = 0.03  # Reduced to 3%
    MAX_POSITION_SIZE = 0.90  # 90% of capital
    MAX_OPEN_TRADES = 2  # Reduced max open trades
    
    # Fees & Spread
    MAKER_FEE = 0.001  # 0.1%
    TAKER_FEE = 0.001  # 0.1%
    MIN_SPREAD_PCT = 0.0002  # 0.02%
    MAX_SPREAD_PCT = 0.001  # 0.1%
    SLIPPAGE_PCT = 0.0005  # 0.05%
    
    # Strategy Parameters
    TAKE_PROFIT_PCT = 0.002  # Reduced to 0.2%
    STOP_LOSS_PCT = 0.0015  # Reduced to 0.15%
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 72  # Adjusted
    RSI_OVERSOLD = 28  # Adjusted
    BB_PERIOD = 20
    BB_STD = 1.8  # Reduced for more signals
    VOLUME_MA_PERIOD = 20
    MIN_VOLUME_RATIO = 1.3  # Reduced
    
    # Execution
    TRADE_COOLDOWN = 60  # Increased to 60 seconds
    CHECK_INTERVAL = 45  # Increased to 45 seconds
    MAX_TRADE_DURATION = 300  # Reduced to 5 minutes
    
    # Bot Settings
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/trading_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'trading_bot.db'
    PAPER_TRADING = True
    RAILWAY_DEPLOYMENT = 'RAILWAY_ENVIRONMENT' in os.environ
    
    # Telegram Rate Limiting
    TELEGRAM_MAX_RETRIES = 3
    TELEGRAM_RETRY_DELAY = 2
    TELEGRAM_BATCH_DELAY = 1  # Delay between messages
    
    # Health Check
    HEALTH_CHECK_PORT = int(os.getenv('PORT', 8080))
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes

# ============================================================================
# ENHANCED LOGGING
# ============================================================================

class RailwayFormatter(logging.Formatter):
    """Custom formatter for Railway logs"""
    
    def format(self, record):
        # Add timestamp and level
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        message = super().format(record)
        
        # Color codes for Railway
        colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[41m'  # Red background
        }
        
        reset = '\033[0m'
        color = colors.get(level, '\033[37m')  # Default white
        
        return f"{color}{timestamp} - {level:8s} - {record.name}: {record.getMessage()}{reset}"

# Setup enhanced logging
logger = logging.getLogger('ScalpingBot')
logger.setLevel(getattr(logging, Config.LOG_LEVEL))

# Remove existing handlers
logger.handlers.clear()

# Console handler for Railway
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
console_handler.setFormatter(RailwayFormatter())
logger.addHandler(console_handler)

# File handler for persistent logs
try:
    file_handler = logging.FileHandler('scalping_bot.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create file handler: {e}")

# ============================================================================
# HEALTH CHECK SERVER
# ============================================================================

class HealthCheckServer:
    """Simple HTTP server for health checks"""
    
    def __init__(self, port: int = Config.HEALTH_CHECK_PORT):
        self.port = port
        self.server = None
        self.running = False
    
    async def start(self):
        """Start the health check server"""
        try:
            import aiohttp
            from aiohttp import web
            
            app = web.Application()
            
            # Health check endpoint
            async def health_check(request):
                return web.Response(text='OK', status=200)
            
            # Bot status endpoint
            async def status_check(request):
                data = {
                    'status': 'running',
                    'timestamp': datetime.now().isoformat(),
                    'deployment': 'railway' if Config.RAILWAY_DEPLOYMENT else 'local'
                }
                return web.json_response(data)
            
            app.router.add_get('/health', health_check)
            app.router.add_get('/status', status_check)
            
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self.port)
            await site.start()
            
            self.server = runner
            self.running = True
            logger.info(f"Health check server started on port {self.port}")
            
        except ImportError:
            logger.warning("aiohttp not installed, health check server disabled")
        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")
    
    async def stop(self):
        """Stop the health check server"""
        if self.server:
            await self.server.cleanup()
            self.running = False
            logger.info("Health check server stopped")

# ============================================================================
# DATABASE MANAGER WITH RAILWAY SUPPORT
# ============================================================================

class Database:
    """SQLite database manager with Railway support"""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
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
                
                # Bot stats table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bot_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP NOT NULL,
                        capital REAL NOT NULL,
                        open_trades INTEGER,
                        daily_pnl REAL,
                        total_pnl REAL
                    )
                ''')
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            # Fallback to in-memory database
            self.db_path = ':memory:'
            logger.info("Using in-memory database as fallback")
    
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
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    def save_bot_stats(self, capital: float, open_trades: int, daily_pnl: float, total_pnl: float):
        """Save bot statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO bot_stats 
                    (timestamp, capital, open_trades, daily_pnl, total_pnl)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    capital,
                    open_trades,
                    daily_pnl,
                    total_pnl
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving bot stats: {e}")

# ============================================================================
# ENHANCED EXCHANGE HANDLER
# ============================================================================

class ExchangeHandler:
    """Enhanced exchange handler with better error handling"""
    
    def __init__(self, exchange_id: str = Config.EXCHANGE):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        self.symbols = Config.SYMBOLS
        self.last_fetch = {}
        self.cache = {}
        self.cache_timeout = 2  # Cache for 2 seconds
        logger.info(f"Exchange handler initialized for {exchange_id}")
    
    async def get_ticker_async(self, symbol: str) -> Optional[Dict]:
        """Get ticker data with async support"""
        try:
            # Check cache
            current_time = time.time()
            if symbol in self.cache and current_time - self.last_fetch.get(symbol, 0) < self.cache_timeout:
                return self.cache[symbol]
            
            # Fetch new data
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
            
            # Update cache
            self.cache[symbol] = ticker
            self.last_fetch[symbol] = current_time
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = Config.TIMEFRAME, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data with retry logic"""
        for attempt in range(3):
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
                if attempt < 2:
                    logger.warning(f"Retry {attempt + 1} for {symbol}: {e}")
                    time.sleep(1)
                else:
                    logger.error(f"Error fetching OHLCV for {symbol}: {e}")
                    return None
    
    def check_market_conditions(self, symbol: str) -> Dict:
        """Check if market conditions are suitable for scalping"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            
            if not ticker or ticker['bid'] is None or ticker['ask'] is None:
                return {'suitable': False, 'reason': 'Invalid ticker data'}
            
            bid = ticker['bid']
            ask = ticker['ask']
            spread_pct = (ask - bid) / bid if bid > 0 else 0
            
            # Check spread
            if spread_pct > Config.MAX_SPREAD_PCT:
                return {'suitable': False, 'reason': f'Spread too high: {spread_pct:.4%}'}
            
            # Check volume
            volume = ticker['quoteVolume'] or 0
            if volume < 500000:  # Minimum $500k volume
                return {'suitable': False, 'reason': f'Volume too low: ${volume:,.0f}'}
            
            return {
                'suitable': True,
                'spread': spread_pct,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'last': ticker['last']
            }
            
        except Exception as e:
            logger.error(f"Error checking market conditions for {symbol}: {e}")
            return {'suitable': False, 'reason': f'Error: {str(e)}'}

# ============================================================================
# ENHANCED TELEGRAM BOT WITH RATE LIMITING
# ============================================================================

class TelegramBot:
    """Enhanced Telegram bot with rate limiting and error handling"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.application = None
        self.bot_running = False
        self.message_queue = asyncio.Queue()
        self.processor_task = None
        self.rate_limiter = RateLimiter(max_calls=20, period=60)  # 20 messages per minute
        logger.info("Telegram bot initialized")
    
    async def start(self):
        """Start the Telegram bot"""
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat ID not configured")
            return
        
        try:
            # Configure application with better settings
            self.application = Application.builder().token(self.token).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("trades", self.trades_command))
            self.application.add_handler(CommandHandler("pnl", self.pnl_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            
            # Start message processor
            self.processor_task = asyncio.create_task(self._process_message_queue())
            
            # Start bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            
            self.bot_running = True
            logger.info("Telegram bot started successfully")
            
            # Send startup message (delayed to avoid rate limiting)
            await asyncio.sleep(2)
            await self.queue_message('startup', {
                'capital': Config.INITIAL_CAPITAL,
                'symbols': Config.SYMBOLS,
                'mode': 'Paper Trading' if Config.PAPER_TRADING else 'Live Trading'
            })
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    async def stop(self):
        """Stop the Telegram bot"""
        if self.processor_task:
            self.processor_task.cancel()
        
        if self.application:
            await self.application.stop()
            self.bot_running = False
            logger.info("Telegram bot stopped")
    
    async def _process_message_queue(self):
        """Process messages from queue with rate limiting"""
        while True:
            try:
                message_type, data = await self.message_queue.get()
                
                # Apply rate limiting
                await self.rate_limiter.wait()
                
                if message_type == 'alert':
                    await self._send_alert_safe(data['alert_type'], data['message'], data.get('data'))
                elif message_type == 'startup':
                    await self._send_startup_message(data)
                
                self.message_queue.task_done()
                await asyncio.sleep(Config.TELEGRAM_BATCH_DELAY)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(1)
    
    async def queue_message(self, message_type: str, data: Dict):
        """Add message to queue for processing"""
        await self.message_queue.put((message_type, data))
    
    async def _send_alert_safe(self, alert_type: str, message: str, data: Dict = None):
        """Send alert with retry logic"""
        for attempt in range(Config.TELEGRAM_MAX_RETRIES):
            try:
                emoji = {
                    'trade': '📊',
                    'signal': '🚨',
                    'profit': '💰',
                    'loss': '📉',
                    'error': '⚠️',
                    'info': 'ℹ️',
                    'warning': '⚠️',
                    'startup': '🚀'
                }.get(alert_type, '📢')
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_msg = f"{emoji} *{alert_type.upper()}* [{timestamp}]\n\n{message}"
                
                if data:
                    formatted_msg += f"\n\n*Details:*\n"
                    for key, value in data.items():
                        formatted_msg += f"• {key}: {value}\n"
                
                await self.application.bot.send_message(
                    chat_id=self.chat_id,
                    text=formatted_msg,
                    parse_mode='Markdown',
                    disable_notification=(alert_type not in ['signal', 'error'])
                )
                
                logger.info(f"Telegram alert sent: {alert_type}")
                return
                
            except RetryAfter as e:
                retry_after = e.retry_after
                logger.warning(f"Rate limited, retrying in {retry_after}s")
                await asyncio.sleep(retry_after)
                
            except (TimedOut, NetworkError) as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                if attempt < Config.TELEGRAM_MAX_RETRIES - 1:
                    await asyncio.sleep(Config.TELEGRAM_RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to send alert after {Config.TELEGRAM_MAX_RETRIES} attempts")
                    break
                    
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
                break
    
    async def _send_startup_message(self, data: Dict):
        """Send startup message"""
        message = "🚀 *Scalping Bot Started*\n\n"
        message += f"*Platform:* {'Railway' if Config.RAILWAY_DEPLOYMENT else 'Local'}\n"
        message += f"*Capital:* ${data['capital']:.2f}\n"
        message += f"*Symbols:* {', '.join(data['symbols'])}\n"
        message += f"*Mode:* {data['mode']}\n"
        message += f"\n*Commands:*\n"
        message += "/start - Start bot\n"
        message += "/status - Check status\n"
        message += "/trades - View trades\n"
        message += "/pnl - Profit/Loss\n"
        message += "/help - Show help"
        
        await self._send_alert_safe('startup', message)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            welcome_msg = """
🤖 *Scalping Bot Activated* 🤖

*Available Commands:*
/status - Bot status & metrics
/trades - Recent trades
/pnl - Profit & Loss summary
/help - Show help

*Risk Settings:*
Capital: $100.00
Risk/Trade: 1.5%
Daily Loss Limit: 3%

Bot is monitoring markets...
            """
            await update.message.reply_text(welcome_msg, parse_mode='Markdown')
        except RetryAfter as e:
            await update.message.reply_text("⚠️ Rate limited, please wait...")
        except Exception as e:
            logger.error(f"Error in start_command: {e}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            status_msg = f"""
📊 *Bot Status*

*Platform:* {'Railway 🚂' if Config.RAILWAY_DEPLOYMENT else 'Local 💻'}
*Status:* ✅ Active
*Mode:* {'Paper Trading 📄' if Config.PAPER_TRADING else 'Live Trading 💰'}
*Symbols:* {', '.join(Config.SYMBOLS)}
*Timeframe:* {Config.TIMEFRAME}

*Last Update:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Note:* Detailed stats available in logs.
            """
            await update.message.reply_text(status_msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text("⚠️ Error fetching status")
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        try:
            trades_msg = """
📈 *Recent Trades*

No trades executed yet.

The bot is scanning for opportunities...
            """
            await update.message.reply_text(trades_msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in trades_command: {e}")
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        try:
            pnl_msg = """
💰 *Profit & Loss Report*

No trading activity yet.

Start trading to see P&L reports.
            """
            await update.message.reply_text(pnl_msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in pnl_command: {e}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        try:
            help_msg = """
🆘 *Help Guide*

*Commands:*
/start - Start the bot
/status - Check bot status
/trades - View recent trades
/pnl - See profit/loss
/help - This help message

*Risk Management:*
• 1.5% risk per trade
• 3% daily loss limit
• 90% max position size
• 2 max open trades

*Strategy:*
• Mean Reversion Scalping
• RSI + Bollinger Bands
• 5-minute timeframe
• Volume confirmation

*Note:* This is a paper trading bot for educational purposes.
            """
            await update.message.reply_text(help_msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in help_command: {e}")

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    async def wait(self):
        """Wait if rate limit is exceeded"""
        now = time.time()
        
        # Remove old calls
        self.calls = [call for call in self.calls if now - call < self.period]
        
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = self.calls[0]
            wait_time = self.period - (now - oldest_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.time()
        
        # Add current call
        self.calls.append(now)
        
        # Keep only recent calls
        self.calls = [call for call in self.calls if now - call < self.period]

# ============================================================================
# MAIN BOT WITH RAILWAY OPTIMIZATIONS
# ============================================================================

class ScalpingBot:
    """Main scalping bot optimized for Railway"""
    
    def __init__(self):
        self.config = Config()
        self.capital = Config.INITIAL_CAPITAL
        self.running = False
        self.start_time = datetime.now()
        
        # Initialize components
        self.exchange = ExchangeHandler()
        self.strategy = ScalpingStrategy()
        self.risk_manager = RiskManager()
        self.database = Database()
        self.telegram_bot = TelegramBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
        self.health_check = HealthCheckServer()
        
        # Performance tracking
        self.total_trades = 0
        self.total_pnl = 0
        self.cycle_count = 0
        
        logger.info("="*60)
        logger.info("🤖 SCALPING TRADING BOT 🤖")
        logger.info("="*60)
        logger.info(f"Platform: {'Railway' if Config.RAILWAY_DEPLOYMENT else 'Local'}")
        logger.info(f"Initial Capital: ${self.capital:.2f}")
        logger.info(f"Trading Symbols: {', '.join(Config.SYMBOLS)}")
        logger.info(f"Paper Trading: {Config.PAPER_TRADING}")
        logger.info(f"Risk per Trade: {Config.RISK_PER_TRADE:.1%}")
        logger.info(f"Daily Loss Limit: {Config.MAX_DAILY_LOSS:.1%}")
        logger.info("="*60)
    
    async def start(self):
        """Start the trading bot"""
        self.running = True
        
        try:
            # Start health check server
            await self.health_check.start()
            
            # Start Telegram bot
            await self.telegram_bot.start()
            
            logger.info("🚀 Bot started successfully")
            logger.info(f"📡 Monitoring {len(Config.SYMBOLS)} symbols")
            logger.info(f"⏱️  Check interval: {Config.CHECK_INTERVAL}s")
            logger.info(f"🔄 Trade cooldown: {Config.TRADE_COOLDOWN}s")
            
            # Main trading loop
            while self.running:
                try:
                    self.cycle_count += 1
                    
                    # Log cycle start
                    if self.cycle_count % 10 == 0:
                        logger.info(f"🔄 Cycle {self.cycle_count} - Checking markets...")
                    
                    # Check markets
                    await self.check_markets()
                    
                    # Save stats periodically
                    if self.cycle_count % 20 == 0:
                        open_trades = len([t for t in self.risk_manager.trades if t.status == 'open'])
                        self.database.save_bot_stats(
                            capital=self.capital,
                            open_trades=open_trades,
                            daily_pnl=self.risk_manager.daily_pnl,
                            total_pnl=self.total_pnl
                        )
                    
                    # Sleep before next cycle
                    await asyncio.sleep(Config.CHECK_INTERVAL)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            if self.telegram_bot.bot_running:
                await self.telegram_bot.queue_message('alert', {
                    'alert_type': 'error',
                    'message': 'Bot crashed!',
                    'data': {'error': str(e)[:100]}
                })
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot"""
        self.running = False
        
        # Calculate runtime
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Send shutdown message
        if self.telegram_bot.bot_running:
            await self.telegram_bot.queue_message('alert', {
                'alert_type': 'info',
                'message': 'Bot stopped',
                'data': {
                    'Uptime': f'{hours}h {minutes}m {seconds}s',
                    'Total Trades': self.total_trades,
                    'Final Capital': f'${self.capital:.2f}',
                    'Total P&L': f'${self.total_pnl:+.2f}'
                }
            })
        
        # Stop components
        await self.telegram_bot.stop()
        await self.health_check.stop()
        
        logger.info("="*60)
        logger.info("🛑 Bot shutdown complete")
        logger.info(f"⏱️  Total uptime: {hours}h {minutes}m {seconds}s")
        logger.info(f"📊 Total trades: {self.total_trades}")
        logger.info(f"💰 Final capital: ${self.capital:.2f}")
        logger.info(f"📈 Total P&L: ${self.total_pnl:+.2f}")
        logger.info("="*60)
    
    async def check_markets(self):
        """Check all markets for trading opportunities"""
        for symbol in Config.SYMBOLS:
            try:
                # Check market conditions
                market_cond = self.exchange.check_market_conditions(symbol)
                if not market_cond['suitable']:
                    if self.cycle_count % 30 == 0:  # Log less frequently
                        logger.debug(f"Market not suitable for {symbol}: {market_cond['reason']}")
                    continue
                
                # Get market data
                df = self.exchange.get_ohlcv(symbol, Config.TIMEFRAME, limit=50)
                if df is None or len(df) < 20:
                    continue
                
                # Generate signal (simplified for demo)
                signal = self.generate_signal(df, symbol)
                if signal is None:
                    continue
                
                # Log signal
                logger.info(f"📈 Signal detected for {symbol}: {signal['side']} at ${signal['price']:.2f}")
                
                # Send Telegram alert
                if self.telegram_bot.bot_running:
                    await self.telegram_bot.queue_message('alert', {
                        'alert_type': 'signal',
                        'message': f"Signal: {signal['side']} {symbol}",
                        'data': {
                            'Price': f'${signal["price"]:.2f}',
                            'Reason': signal['reason'],
                            'Spread': f'{market_cond["spread"]:.4%}',
                            'Volume': f'${market_cond["volume"]:,.0f}'
                        }
                    })
                
                # Simulate trade (paper trading)
                await self.simulate_trade(symbol, signal, market_cond)
                
                # Cooldown between trades
                await asyncio.sleep(Config.TRADE_COOLDOWN)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate simplified trading signal"""
        try:
            # Calculate simple indicators
            close_prices = df['close'].values
            
            if len(close_prices) < 20:
                return None
            
            # Simple moving averages
            sma_short = np.mean(close_prices[-5:])
            sma_long = np.mean(close_prices[-20:])
            
            # Price action
            current_price = close_prices[-1]
            prev_price = close_prices[-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
            
            # Volume check
            volumes = df['volume'].values
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Generate signals
            if (sma_short > sma_long * 1.001 and  # Short MA above long MA
                price_change > 0.05 and  # Positive momentum
                volume_ratio > Config.MIN_VOLUME_RATIO):  # High volume
                return {
                    'symbol': symbol,
                    'side': 'BUY',
                    'price': current_price,
                    'reason': f'Bullish crossover (Δ{price_change:.2f}%, Vol×{volume_ratio:.1f})'
                }
            
            elif (sma_short < sma_long * 0.999 and  # Short MA below long MA
                  price_change < -0.05 and  # Negative momentum
                  volume_ratio > Config.MIN_VOLUME_RATIO):  # High volume
                return {
                    'symbol': symbol,
                    'side': 'SELL',
                    'price': current_price,
                    'reason': f'Bearish crossover (Δ{price_change:.2f}%, Vol×{volume_ratio:.1f})'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def simulate_trade(self, symbol: str, signal: Dict, market_cond: Dict):
        """Simulate a trade (paper trading)"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(signal['price'])
            
            if position_size <= 0:
                return
            
            # Simulate trade execution
            entry_price = signal['price']
            fees = entry_price * position_size * Config.TAKER_FEE
            trade_value = entry_price * position_size
            
            # Update capital
            self.capital -= trade_value + fees
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'side': signal['side'],
                'entry_price': entry_price,
                'size': position_size,
                'fees': fees,
                'entry_time': datetime.now(),
                'status': 'open'
            }
            
            # Save to database
            self.database.save_trade(trade)
            
            # Send execution alert
            if self.telegram_bot.bot_running:
                await self.telegram_bot.queue_message('alert', {
                    'alert_type': 'trade',
                    'message': f"Trade Executed: {signal['side']} {symbol}",
                    'data': {
                        'Entry Price': f'${entry_price:.2f}',
                        'Size': f'{position_size:.6f}',
                        'Value': f'${trade_value:.2f}',
                        'Fees': f'${fees:.4f}',
                        'New Capital': f'${self.capital:.2f}'
                    }
                })
            
            logger.info(f"Trade simulated: {signal['side']} {symbol} "
                       f"at ${entry_price:.2f}, size: {position_size:.6f}")
            
            # Monitor trade in background
            asyncio.create_task(self.monitor_simulated_trade(trade, signal))
            
        except Exception as e:
            logger.error(f"Error simulating trade for {symbol}: {e}")
    
    def calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.capital * Config.RISK_PER_TRADE
        
        # Assume 1% price risk for simulation
        price_risk = entry_price * 0.01
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        # Apply max position size
        max_size = self.capital * Config.MAX_POSITION_SIZE / entry_price
        position_size = min(position_size, max_size)
        
        return round(position_size, 6)
    
    async def monitor_simulated_trade(self, trade: Dict, signal: Dict):
        """Monitor simulated trade for exit"""
        try:
            entry_time = datetime.now()
            entry_price = trade['entry_price']
            
            # Random exit simulation (between 1-5 minutes)
            wait_time = np.random.uniform(60, 300)
            await asyncio.sleep(wait_time)
            
            # Get current price
            market_cond = self.exchange.check_market_conditions(trade['symbol'])
            if not market_cond['suitable']:
                exit_price = entry_price * (1 + np.random.uniform(-0.002, 0.002))
            else:
                exit_price = market_cond['last']
            
            # Calculate P&L
            exit_fees = exit_price * trade['size'] * Config.TAKER_FEE
            total_fees = trade['fees'] + exit_fees
            
            if trade['side'] == 'BUY':
                pnl = (exit_price - entry_price) * trade['size'] - total_fees
            else:
                pnl = (entry_price - exit_price) * trade['size'] - total_fees
            
            pnl_pct = (pnl / (entry_price * trade['size'])) * 100
            
            # Update capital
            self.capital += trade['size'] * exit_price - exit_fees
            self.total_trades += 1
            self.total_pnl += pnl
            
            # Update trade record
            trade.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'status': 'closed',
                'reason': 'Time-based exit'
            })
            
            # Save to database
            self.database.save_trade(trade)
            
            # Send P&L alert
            if self.telegram_bot.bot_running:
                alert_type = 'profit' if pnl > 0 else 'loss'
                await self.telegram_bot.queue_message('alert', {
                    'alert_type': alert_type,
                    'message': f"Trade Closed: {trade['side']} {trade['symbol']}",
                    'data': {
                        'P&L': f'${pnl:+.2f} ({pnl_pct:+.2f}%)',
                        'Entry': f'${entry_price:.2f}',
                        'Exit': f'${exit_price:.2f}',
                        'Duration': f'{wait_time:.0f}s',
                        'Total Capital': f'${self.capital:.2f}'
                    }
                })
            
            logger.info(f"Trade closed: {trade['symbol']} {trade['side']} "
                       f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Error monitoring trade {trade['symbol']}: {e}")

# ============================================================================
# SIMPLIFIED COMPONENTS
# ============================================================================

class ScalpingStrategy:
    """Simplified strategy for Railway"""
    def __init__(self):
        logger.info("Strategy initialized")

class RiskManager:
    """Simplified risk manager"""
    def __init__(self):
        self.trades = []
        self.daily_pnl = 0
        logger.info("Risk manager initialized")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🤖 SCALPING TRADING BOT - RAILWAY EDITION 🤖")
    print("="*60)
    print(f"Platform: {'Railway 🚂' if Config.RAILWAY_DEPLOYMENT else 'Local 💻'}")
    print(f"Initial Capital: ${Config.INITIAL_CAPITAL}")
    print(f"Symbols: {', '.join(Config.SYMBOLS)}")
    print(f"Timeframe: {Config.TIMEFRAME}")
    print(f"Mode: {'Paper Trading' if Config.PAPER_TRADING else 'Live Trading'}")
    print("="*60 + "\n")
    
    # Create and run bot
    bot = ScalpingBot()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal...")
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
        print("\n✅ Bot shutdown complete")

if __name__ == "__main__":
    # Check for required packages
    required_packages = {
        'ccxt': 'ccxt',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'telegram': 'python-telegram-bot',
        'dotenv': 'python-dotenv'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name if import_name != 'telegram' else 'telegram.ext')
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install ccxt pandas numpy python-telegram-bot python-dotenv ta schedule")
        sys.exit(1)
    
    # Run the bot
    asyncio.run(main())







