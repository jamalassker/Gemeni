# bot.py
"""
Simplified Scalping Bot for Railway
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
from typing import Dict, List, Optional
import ccxt
import requests
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    
    # Trading Configuration
    EXCHANGE = 'binance'
    SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Reduced symbols
    TIMEFRAME = '5m'
    
    # Risk Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.015  # 1.5%
    MAX_DAILY_LOSS = 0.03  # 3%
    MAX_POSITION_SIZE = 0.90  # 90%
    MAX_OPEN_TRADES = 2
    
    # Strategy Parameters
    TAKE_PROFIT_PCT = 0.002  # 0.2%
    STOP_LOSS_PCT = 0.0015  # 0.15%
    
    # Execution
    CHECK_INTERVAL = 60  # 60 seconds
    TRADE_COOLDOWN = 120  # 120 seconds
    MAX_TRADE_DURATION = 300  # 5 minutes
    
    # Bot Settings
    LOG_LEVEL = 'INFO'
    DATABASE_PATH = '/data/trading_bot.db' if 'RAILWAY_ENVIRONMENT' in os.environ else 'trading_bot.db'
    
    # Telegram Configuration (Optional)
    TELEGRAM_ENABLED = True  # Disabled to simplify
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Health Check
    PORT = int(os.getenv('PORT', 8080))

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger('ScalpingBot')

# ============================================================================
# SIMPLE DATABASE
# ============================================================================

class SimpleDatabase:
    """Simple SQLite database"""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    pnl REAL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    status TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    capital REAL,
                    total_trades INTEGER,
                    total_pnl REAL
                )
            ''')
            
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (symbol, side, entry_price, exit_price, size, pnl, entry_time, exit_time, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('size'),
                    trade_data.get('pnl'),
                    trade_data.get('entry_time'),
                    trade_data.get('exit_time'),
                    trade_data.get('status', 'open')
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    
    def save_stats(self, capital: float, total_trades: int, total_pnl: float):
        """Save bot statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO bot_stats (timestamp, capital, total_trades, total_pnl)
                    VALUES (?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    capital,
                    total_trades,
                    total_pnl
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving stats: {e}")

# ============================================================================
# EXCHANGE HANDLER
# ============================================================================

class ExchangeHandler:
    """Simple exchange handler"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 10000
        })
        logger.info("Exchange handler initialized")
    
    def get_ohlcv(self, symbol: str, timeframe: str = Config.TIMEFRAME, limit: int = 50):
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
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_ticker(self, symbol: str):
        """Get current ticker"""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

# ============================================================================
# TRADING STRATEGY
# ============================================================================

class TradingStrategy:
    """Simple trading strategy"""
    
    def __init__(self):
        logger.info("Trading strategy initialized")
    
    def analyze(self, df: pd.DataFrame, symbol: str):
        """Analyze market data"""
        if df is None or len(df) < 20:
            return None
        
        try:
            close_prices = df['close'].values
            volumes = df['volume'].values
            
            # Calculate simple indicators
            sma_short = np.mean(close_prices[-5:])
            sma_long = np.mean(close_prices[-20:])
            current_price = close_prices[-1]
            
            # Volume analysis
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Generate signals
            if sma_short > sma_long and volume_ratio > 1.2:
                return {
                    'symbol': symbol,
                    'side': 'BUY',
                    'price': current_price,
                    'confidence': 0.7,
                    'reason': f'Bullish trend (Vol x{volume_ratio:.1f})'
                }
            elif sma_short < sma_long and volume_ratio > 1.2:
                return {
                    'symbol': symbol,
                    'side': 'SELL',
                    'price': current_price,
                    'confidence': 0.7,
                    'reason': f'Bearish trend (Vol x{volume_ratio:.1f})'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Simple risk manager"""
    
    def __init__(self):
        self.open_trades = []
        self.daily_pnl = 0
        self.daily_start = datetime.now().date()
        logger.info("Risk manager initialized")
    
    def can_trade(self, symbol: str):
        """Check if trading is allowed"""
        # Check daily loss
        if self.daily_pnl < -Config.INITIAL_CAPITAL * Config.MAX_DAILY_LOSS:
            return False, "Daily loss limit reached"
        
        # Check open trades count
        if len(self.open_trades) >= Config.MAX_OPEN_TRADES:
            return False, "Max open trades reached"
        
        # Check symbol cooldown
        for trade in self.open_trades:
            if trade['symbol'] == symbol:
                trade_time = trade.get('entry_time', datetime.min)
                if datetime.now() - trade_time < timedelta(seconds=Config.TRADE_COOLDOWN):
                    return False, f"Cooldown for {symbol}"
        
        return True, "OK"
    
    def calculate_position_size(self, capital: float, entry_price: float):
        """Calculate position size"""
        risk_amount = capital * Config.RISK_PER_TRADE
        price_risk = entry_price * 0.01  # Assume 1% risk
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        max_size = capital * Config.MAX_POSITION_SIZE / entry_price
        
        return min(position_size, max_size)

# ============================================================================
# HEALTH CHECK SERVER
# ============================================================================

async def health_check_server():
    """Simple health check server"""
    try:
        from aiohttp import web
        
        app = web.Application()
        
        async def health_handler(request):
            return web.Response(text='OK', status=200)
        
        async def status_handler(request):
            data = {
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'service': 'scalping-bot'
            }
            return web.json_response(data)
        
        app.router.add_get('/health', health_handler)
        app.router.add_get('/status', status_handler)
        app.router.add_get('/', health_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', Config.PORT)
        await site.start()
        
        logger.info(f"Health check server started on port {Config.PORT}")
        return runner
        
    except ImportError:
        logger.warning("aiohttp not available, health check disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to start health check: {e}")
        return None

# ============================================================================
# TELEGRAM NOTIFICATIONS (OPTIONAL)
# ============================================================================

class TelegramNotifier:
    """Simple Telegram notifications"""
    
    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = Config.TELEGRAM_ENABLED and self.token and self.chat_id
        
        if self.enabled:
            logger.info("Telegram notifier initialized")
        else:
            logger.info("Telegram notifier disabled")
    
    async def send_message(self, message: str):
        """Send Telegram message"""
        if not self.enabled:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.debug("Telegram message sent")
                    else:
                        logger.error(f"Telegram error: {await response.text()}")
                        
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

# ============================================================================
# MAIN BOT
# ============================================================================

class ScalpingBot:
    """Main scalping bot"""
    
    def __init__(self):
        self.config = Config()
        self.capital = Config.INITIAL_CAPITAL
        self.running = False
        self.start_time = datetime.now()
        
        # Initialize components
        self.exchange = ExchangeHandler()
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager()
        self.database = SimpleDatabase()
        
        # Telegram (optional)
        try:
            import aiohttp
            self.telegram = TelegramNotifier()
            self.has_aiohttp = True
        except ImportError:
            logger.warning("aiohttp not installed, Telegram disabled")
            self.telegram = None
            self.has_aiohttp = False
        
        # Statistics
        self.total_trades = 0
        self.total_pnl = 0
        self.cycle_count = 0
        
        logger.info("="*50)
        logger.info("🤖 SCALPING TRADING BOT")
        logger.info("="*50)
        logger.info(f"Capital: ${self.capital:.2f}")
        logger.info(f"Symbols: {', '.join(Config.SYMBOLS)}")
        logger.info(f"Check Interval: {Config.CHECK_INTERVAL}s")
        logger.info("="*50)
    
    async def start(self):
        """Start the bot"""
        self.running = True
        
        try:
            # Start health check server
            health_server = await health_check_server()
            
            logger.info("🚀 Bot started successfully")
            
            # Send startup notification
            if self.telegram and self.telegram.enabled:
                startup_msg = f"🚀 Bot started\nCapital: ${self.capital:.2f}\nSymbols: {', '.join(Config.SYMBOLS)}"
                await self.telegram.send_message(startup_msg)
            
            # Main loop
            while self.running:
                try:
                    self.cycle_count += 1
                    
                    # Log every 10 cycles
                    if self.cycle_count % 10 == 0:
                        logger.info(f"Cycle {self.cycle_count} - Capital: ${self.capital:.2f}")
                    
                    # Check markets
                    await self.check_markets()
                    
                    # Save stats every 20 cycles
                    if self.cycle_count % 20 == 0:
                        self.database.save_stats(
                            capital=self.capital,
                            total_trades=self.total_trades,
                            total_pnl=self.total_pnl
                        )
                    
                    # Sleep
                    await asyncio.sleep(Config.CHECK_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
        finally:
            await self.stop()
            if health_server:
                await health_server.cleanup()
    
    async def stop(self):
        """Stop the bot"""
        self.running = False
        
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("="*50)
        logger.info("🛑 Bot stopped")
        logger.info(f"Uptime: {hours}h {minutes}m {seconds}s")
        logger.info(f"Total trades: {self.total_trades}")
        logger.info(f"Final capital: ${self.capital:.2f}")
        logger.info(f"Total P&L: ${self.total_pnl:+.2f}")
        logger.info("="*50)
        
        # Send stop notification
        if self.telegram and self.telegram.enabled:
            stop_msg = f"🛑 Bot stopped\nUptime: {hours}h {minutes}m\nTrades: {self.total_trades}\nCapital: ${self.capital:.2f}\nP&L: ${self.total_pnl:+.2f}"
            await self.telegram.send_message(stop_msg)
    
    async def check_markets(self):
        """Check all markets"""
        for symbol in Config.SYMBOLS:
            try:
                # Get market data
                df = self.exchange.get_ohlcv(symbol)
                if df is None:
                    continue
                
                # Analyze
                signal = self.strategy.analyze(df, symbol)
                if signal is None:
                    continue
                
                # Check risk
                can_trade, reason = self.risk_manager.can_trade(symbol)
                if not can_trade:
                    logger.debug(f"Cannot trade {symbol}: {reason}")
                    continue
                
                # Calculate position
                position_size = self.risk_manager.calculate_position_size(
                    self.capital,
                    signal['price']
                )
                
                if position_size <= 0:
                    continue
                
                # Execute trade
                logger.info(f"📈 Signal: {signal['side']} {symbol} at ${signal['price']:.2f}")
                await self.execute_trade(signal, position_size)
                
                # Cooldown
                await asyncio.sleep(Config.TRADE_COOLDOWN)
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
    
    async def execute_trade(self, signal: Dict, position_size: float):
        """Execute a trade"""
        try:
            entry_price = signal['price']
            trade_value = entry_price * position_size
            
            # Simulate fees (0.1%)
            fees = trade_value * 0.001
            
            # Update capital
            self.capital -= trade_value + fees
            
            # Create trade record
            trade = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'entry_price': entry_price,
                'size': position_size,
                'entry_time': datetime.now(),
                'status': 'open'
            }
            
            # Save to database
            self.database.save_trade(trade)
            
            # Add to open trades
            self.risk_manager.open_trades.append(trade)
            
            # Send notification
            if self.telegram and self.telegram.enabled:
                msg = f"⚡ Trade: {signal['side']} {signal['symbol']}\nPrice: ${entry_price:.2f}\nSize: {position_size:.6f}\nCapital: ${self.capital:.2f}"
                await self.telegram.send_message(msg)
            
            logger.info(f"Trade executed: {signal['side']} {signal['symbol']} at ${entry_price:.2f}")
            
            # Monitor trade
            asyncio.create_task(self.monitor_trade(trade))
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def monitor_trade(self, trade: Dict):
        """Monitor and close trade"""
        try:
            entry_time = datetime.now()
            
            # Wait random time (1-5 minutes)
            wait_time = np.random.uniform(60, 300)
            await asyncio.sleep(wait_time)
            
            # Get exit price
            ticker = self.exchange.get_ticker(trade['symbol'])
            if ticker:
                exit_price = ticker['last']
            else:
                # Simulate price change (±0.3%)
                price_change = np.random.uniform(-0.003, 0.003)
                exit_price = trade['entry_price'] * (1 + price_change)
            
            # Calculate P&L
            if trade['side'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['size']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['size']
            
            # Deduct fees (0.1% each way)
            fees = trade['entry_price'] * trade['size'] * 0.001
            fees += exit_price * trade['size'] * 0.001
            pnl -= fees
            
            # Update capital
            self.capital += trade['size'] * exit_price - fees
            self.total_trades += 1
            self.total_pnl += pnl
            self.risk_manager.daily_pnl += pnl
            
            # Update trade
            trade.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'pnl': pnl,
                'status': 'closed'
            })
            
            # Remove from open trades
            self.risk_manager.open_trades = [
                t for t in self.risk_manager.open_trades 
                if t.get('symbol') != trade['symbol'] or t.get('entry_time') != trade['entry_time']
            ]
            
            # Save to database
            self.database.save_trade(trade)
            
            # Send notification
            if self.telegram and self.telegram.enabled:
                pnl_sign = '+' if pnl > 0 else ''
                msg = f"💰 Trade closed: {trade['side']} {trade['symbol']}\nP&L: {pnl_sign}${pnl:.2f} ({pnl/trade['entry_price']/trade['size']*100:+.2f}%)\nCapital: ${self.capital:.2f}"
                await self.telegram.send_message(msg)
            
            logger.info(f"Trade closed: {trade['symbol']} {trade['side']} P&L: ${pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"Error monitoring trade: {e}")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point"""
    print("\n" + "="*50)
    print("🤖 SCALPING TRADING BOT")
    print("="*50)
    print(f"Platform: {'Railway' if 'RAILWAY_ENVIRONMENT' in os.environ else 'Local'}")
    print(f"Capital: ${Config.INITIAL_CAPITAL}")
    print(f"Symbols: {', '.join(Config.SYMBOLS)}")
    print(f"Interval: {Config.CHECK_INTERVAL}s")
    print("="*50)
    print()
    
    # Check requirements
    try:
        import ccxt
        import pandas
        import numpy
        print("✅ Dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install ccxt pandas numpy")
        return
    
    # Create and run bot
    bot = ScalpingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # Run with asyncio
    asyncio.run(main())







