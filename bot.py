
import ccxt
import pandas as pd
import numpy as np
import requests
import asyncio
import json
import os
import joblib
import warnings
import time
import sys
from typing import Optional, List, Dict, Union  # <--- Add this line
from datetime import datetime, time as dtime, timedelta

import math
from datetime import datetime, time as dtime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import warnings
import aiohttp
from aiohttp import web

warnings.filterwarnings('ignore')

# ================= CONFIG =================
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", 
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT", 
    "LINK/USDT:USDT", "TRX/USDT:USDT", "POL/USDT:USDT", "LTC/USDT:USDT", 
    "BCH/USDT:USDT", "1000SHIB/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", 
    "SUI/USDT:USDT", "ICP/USDT:USDT", "RENDER/USDT:USDT", "STX/USDT:USDT"
]
  # REAL Binance symbols
TIMEFRAME = "5m"  # Options: 1m, 5m, 15m, 1h, 4h, 1d
CANDLES_TO_FETCH = 1000  # Get last 1000 candles

# Trading Parameters (Paper Trading)
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_POSITIONS = 3
STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit

# AI Model Thresholds
ML_PROB_THRESHOLD = 0.65
LSTM_PROB_THRESHOLD = 0.60
COMBINED_THRESHOLD = 0.70

# Model paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Telegram
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ================= TELEGRAM =================
def send_telegram(message: str) -> bool:
    """Send message to Telegram"""
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("Telegram credentials not set")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {
            "chat_id": TG_CHAT,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"Telegram sent: {message[:50]}...")
            return True
        else:
            logger.error(f"Telegram error: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Telegram failed: {e}")
        return False

# ================= REAL MARKET DATA FETCHER =================
class MarketDataFetcher:
    """Fetches REAL market data from Binance WITHOUT API keys"""
    
    def __init__(self):
        # Initialize CCXT WITHOUT API keys - using public endpoints only
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Respect rate limits
            'timeout': 30000,         # 30 second timeout
        })
        
        # Verify connection
        try:
            self.exchange.load_markets()
            logger.info(f"✓ Connected to Binance. Available markets: {len(self.exchange.markets)}")
            send_telegram("🤖 Connected to Binance - Free Real Data")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            send_telegram(f"❌ Binance connection failed: {e}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLES_TO_FETCH) -> pd.DataFrame:
        """Fetch REAL OHLCV data from Binance (NO API keys needed)"""
        try:
            logger.info(f"📊 Fetching REAL data for {symbol} ({timeframe})...")
            
            # Fetch data from Binance - this works WITHOUT API keys
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"✓ Fetched {len(df)} candles for {symbol} | Latest: ${df['close'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']  # Current price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def fetch_order_book(self, symbol: str, limit: int = 10):
        """Fetch order book data (bids/asks)"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

# ================= TECHNICAL INDICATORS =================
class TechnicalIndicators:
    """Calculate technical indicators from real market data"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple technical indicators"""
        if len(df) < 50:
            return df
        
        try:
            # Price data
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Moving Averages
            df['sma_20'] = pd.Series(close).rolling(20).mean()
            df['sma_50'] = pd.Series(close).rolling(50).mean()
            df['sma_200'] = pd.Series(close).rolling(200).mean()
            df['ema_12'] = pd.Series(close).ewm(span=12).mean()
            df['ema_26'] = pd.Series(close).ewm(span=26).mean()
            
            # MACD
            exp1 = pd.Series(close).ewm(span=12).mean()
            exp2 = pd.Series(close).ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = df['sma_20']
            rolling_std = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma_20 + (rolling_std * 2)
            df['bb_lower'] = sma_20 - (rolling_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = pd.Series(volume).rolling(20).mean()
            df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1)
            
            # Price change
            df['returns'] = pd.Series(close).pct_change()
            df['log_returns'] = np.log(pd.Series(close) / pd.Series(close).shift(1))
            
            # ATR (Average True Range)
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            true_range = np.maximum.reduce([high_low, high_close, low_close])
            df['atr'] = pd.Series(true_range).rolling(14).mean()
            
            # Stochastic
            lowest_low = pd.Series(low).rolling(14).min()
            highest_high = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, 0.001))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Clean NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            logger.debug(f"Calculated {len(df.columns)} indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

# ================= AI MODELS =================
class AIModels:
    """AI models for market prediction"""
    
    def __init__(self):
        self.ml_model = None
        self.lstm_model = None
        self.scaler = None
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Try to load ML model
            if os.path.exists(ML_MODEL_PATH):
                self.ml_model = joblib.load(ML_MODEL_PATH)
                logger.info("✓ Loaded ML model from disk")
            else:
                logger.info("Training ML model...")
                self.train_ml_model()
            
            # Try to load LSTM model
            if os.path.exists(LSTM_MODEL_PATH):
                self.lstm_model = load_model(LSTM_MODEL_PATH)
                logger.info("✓ Loaded LSTM model from disk")
            else:
                logger.info("Training LSTM model...")
                self.train_lstm_model()
            
            # Try to load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("✓ Loaded scaler from disk")
                
            send_telegram("✅ AI Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            send_telegram(f"❌ Model loading error: {str(e)[:100]}")
    
    def train_ml_model(self):
        """Train gradient boosting classifier"""
        # Create synthetic training data based on market patterns
        np.random.seed(42)
        n_samples = 10000
        
        # Features: simulate 20 features (technical indicators)
        X = np.random.randn(n_samples, 20)
        
        # Labels: create patterns that might indicate good trades
        # Pattern: positive momentum + low RSI + high volume
        y = (
            (X[:, 0] > 0.5) &  # Momentum positive
            (X[:, 1] < 0.3) &  # RSI low
            (X[:, 2] > 0.7)    # Volume high
        ).astype(int)
        
        # Add some noise
        y = y ^ (np.random.rand(n_samples) > 0.9)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ))
        ])
        
        self.ml_model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.ml_model, ML_MODEL_PATH)
        self.scaler = self.ml_model.named_steps['scaler']
        joblib.dump(self.scaler, SCALER_PATH)
        
        accuracy = self.ml_model.score(X_test, y_test)
        logger.info(f"ML Model trained. Accuracy: {accuracy:.2%}")
    
    def train_lstm_model(self):
        """Train LSTM model for sequence prediction"""
        # Create synthetic sequential data
        np.random.seed(42)
        n_sequences = 5000
        sequence_length = 30
        n_features = 10
        
        X = np.random.randn(n_sequences, sequence_length, n_features)
        
        # Create target: probability of price increase
        y = np.zeros(n_sequences)
        for i in range(n_sequences):
            # Simple pattern: if recent trend is positive
            trend = np.mean(X[i, -5:, 0]) - np.mean(X[i, -10:-5, 0])
            y[i] = 1 / (1 + np.exp(-trend * 5))
        
        # Add noise
        y = y + np.random.randn(n_sequences) * 0.1
        y = np.clip(y, 0, 1)
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        self.lstm_model.fit(
            X, y,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Save model
        self.lstm_model.save(LSTM_MODEL_PATH)
        logger.info("LSTM Model trained and saved")
    
    def predict_ml(self, features: np.ndarray) -> float:
        """Get ML model prediction probability"""
        if self.ml_model is None:
            return 0.5
        
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            prob = self.ml_model.predict_proba(features)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 0.5
    
    def predict_lstm(self, sequence: np.ndarray) -> float:
        """Get LSTM model prediction probability"""
        if self.lstm_model is None:
            return 0.5
        
        try:
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            prob = self.lstm_model.predict(sequence, verbose=0)[0][0]
            return float(prob)
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.5

# ================= TRADING BOT =================
class TradingBot:
    """Main trading bot using real market data and AI models"""
    
    def __init__(self):
        self.market_data = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.ai_models = AIModels()
        
        # Paper trading account
        self.balance = INITIAL_BALANCE
        self.positions = []  # List of open positions
        self.trade_history = []
        self.equity_curve = []
        
        logger.info("=" * 60)
        logger.info("🤖 AI TRADING BOT INITIALIZED")
        logger.info(f"📈 Symbols: {SYMBOLS}")
        logger.info(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
        logger.info(f"⏰ Timeframe: {TIMEFRAME}")
        logger.info("=" * 60)
        
        # Send startup message
        startup_msg = f"""🚀 AI Trading Bot Started
📊 Using REAL Binance Data (FREE)
💰 Paper Balance: ${INITIAL_BALANCE:,.2f}
📈 Trading: {', '.join(SYMBOLS)}
⏰ Timeframe: {TIMEFRAME}
🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        send_telegram(startup_msg)
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a symbol and return trading signals"""
        try:
            # 1. Fetch REAL market data
            df = self.market_data.fetch_ohlcv(symbol)
            if len(df) < 100:
                logger.warning(f"Not enough data for {symbol}")
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            # 2. Calculate indicators
            df = self.indicators.calculate_all_indicators(df)
            if len(df) < 50:
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            # Get latest data point
            latest = df.iloc[-1]
            
            # 3. Get AI predictions
            # Prepare features for ML model
            feature_cols = [
                'rsi', 'macd', 'macd_hist', 'bb_position',
                'volume_ratio', 'returns', 'atr', 'stoch_k'
            ]
            
            features = []
            for col in feature_cols:
                if col in df.columns:
                    features.append(latest[col])
            
            features = np.array(features).reshape(1, -1)
            
            # Get ML prediction
            ml_prob = self.ai_models.predict_ml(features)
            
            # Prepare sequence for LSTM
            sequence_cols = ['close', 'volume', 'rsi', 'macd', 'atr']
            sequence_data = []
            for col in sequence_cols:
                if col in df.columns:
                    sequence_data.append(df[col].values[-30:])  # Last 30 periods
            
            if len(sequence_data) >= 3:
                sequence = np.column_stack(sequence_data)
                lstm_prob = self.ai_models.predict_lstm(sequence)
            else:
                lstm_prob = 0.5
            
            # 4. Calculate combined confidence
            combined_prob = (ml_prob * 0.6 + lstm_prob * 0.4)
            
            # 5. Generate signal based on rules + AI
            signal = "HOLD"
            confidence = combined_prob
            
            # Buy conditions
            if (combined_prob > COMBINED_THRESHOLD and
                latest['rsi'] < 40 and  # Not overbought
                latest['close'] > latest['sma_20'] and  # Above short MA
                len(self.positions) < MAX_POSITIONS):
                
                signal = "BUY"
                
                # Calculate position size
                position_value = self.balance * RISK_PER_TRADE
                entry_price = latest['close']
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                
                # Create paper trade position
                position = {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(),
                    "size": position_value / entry_price,
                    "value": position_value,
                    "status": "OPEN"
                }
                
                self.positions.append(position)
                self.balance -= position_value
                
                # Log and notify
                logger.info(f"📈 BUY {symbol} @ ${entry_price:.2f}")
                trade_msg = f"""✅ PAPER TRADE ENTERED
Symbol: {symbol}
Entry: ${entry_price:.2f}
Stop Loss: ${stop_loss:.2f} ({STOP_LOSS_PCT*100:.1f}%)
Take Profit: ${take_profit:.2f} ({TAKE_PROFIT_PCT*100:.1f}%)
Size: {position['size']:.6f} {symbol.split('/')[0]}
AI Confidence: {combined_prob:.1%}
Balance: ${self.balance:,.2f}"""
                send_telegram(trade_msg)
            
            # Check existing positions
            self.check_positions(symbol, latest['close'])
            
            return {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "price": latest['close'],
                "rsi": latest['rsi'] if 'rsi' in latest else None,
                "ml_prob": ml_prob,
                "lstm_prob": lstm_prob
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return {"symbol": symbol, "signal": "ERROR", "confidence": 0}
    
    def check_positions(self, symbol: str, current_price: float):
        """Check and update existing positions"""
        positions_to_remove = []
        
        for position in self.positions:
            if position["symbol"] != symbol:
                continue
            
            # Check stop loss
            if current_price <= position["stop_loss"]:
                position["exit_price"] = position["stop_loss"]
                position["exit_time"] = datetime.now()
                position["status"] = "STOP_LOSS"
                position["pnl"] = (position["exit_price"] - position["entry_price"]) * position["size"]
                positions_to_remove.append(position)
                
                logger.info(f"❌ STOP LOSS {symbol} @ ${current_price:.2f}")
                loss_msg = f"""🛑 STOP LOSS HIT
Symbol: {symbol}
Exit: ${current_price:.2f}
Loss: ${position['pnl']:,.2f}
Duration: {(position['exit_time'] - position['entry_time']).total_seconds()/60:.1f} min
Balance: ${self.balance + position['value'] + position['pnl']:,.2f}"""
                send_telegram(loss_msg)
            
            # Check take profit
            elif current_price >= position["take_profit"]:
                position["exit_price"] = position["take_profit"]
                position["exit_time"] = datetime.now()
                position["status"] = "TAKE_PROFIT"
                position["pnl"] = (position["exit_price"] - position["entry_price"]) * position["size"]
                positions_to_remove.append(position)
                
                logger.info(f"✅ TAKE PROFIT {symbol} @ ${current_price:.2f}")
                profit_msg = f"""🎯 TAKE PROFIT HIT
Symbol: {symbol}
Exit: ${current_price:.2f}
Profit: ${position['pnl']:,.2f} (+{((position['exit_price']/position['entry_price'])-1)*100:.1f}%)
Duration: {(position['exit_time'] - position['entry_time']).total_seconds()/60:.1f} min
Balance: ${self.balance + position['value'] + position['pnl']:,.2f}"""
                send_telegram(profit_msg)
        
        # Remove closed positions and return funds
        for position in positions_to_remove:
            if position in self.positions:
                self.positions.remove(position)
                self.balance += position["value"] + position.get("pnl", 0)
                self.trade_history.append(position.copy())
    
    async def run(self):
        """Main bot loop"""
        logger.info("Starting trading bot...")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {iteration} - {current_time}")
                logger.info(f"Open Positions: {len(self.positions)}")
                logger.info(f"Balance: ${self.balance:,.2f}")
                
                # Analyze each symbol
                for symbol in SYMBOLS:
                    try:
                        analysis = self.analyze_symbol(symbol)
                        
                        if analysis["signal"] != "HOLD":
                            logger.info(f"{symbol}: {analysis['signal']} (Confidence: {analysis['confidence']:.1%})")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Record equity curve
                total_position_value = sum(p["value"] for p in self.positions)
                total_equity = self.balance + total_position_value
                self.equity_curve.append({
                    "timestamp": datetime.now(),
                    "equity": total_equity,
                    "balance": self.balance,
                    "positions": len(self.positions)
                })
                
                # Send hourly update
                if iteration % 12 == 0:  # Every hour (5min * 12)
                    self.send_status_update()
                
                # Wait for next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
    
    def send_status_update(self):
        """Send status update to Telegram"""
        total_position_value = sum(p["value"] for p in self.positions)
        total_equity = self.balance + total_position_value
        total_return = ((total_equity / INITIAL_BALANCE) - 1) * 100
        
        status_msg = f"""📊 BOT STATUS UPDATE
Total Equity: ${total_equity:,.2f}
Available Balance: ${self.balance:,.2f}
Open Positions: {len(self.positions)}
Total Return: {total_return:+.2f}%
Best Trade: {max([t.get('pnl', 0) for t in self.trade_history] + [0]):.2f}
Worst Trade: {min([t.get('pnl', 0) for t in self.trade_history] + [0]):.2f}
Win Rate: {self.calculate_win_rate():.1%}
Time: {datetime.now().strftime('%H:%M:%S')}"""
        
        send_telegram(status_msg)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        return winning_trades / len(self.trade_history)
    
    def shutdown(self):
        """Shutdown the bot"""
        total_position_value = sum(p["value"] for p in self.positions)
        total_equity = self.balance + total_position_value
        total_return = ((total_equity / INITIAL_BALANCE) - 1) * 100
        
        shutdown_msg = f"""🛑 BOT SHUTTING DOWN
Final Equity: ${total_equity:,.2f}
Total Return: {total_return:+.2f}%
Total Trades: {len(self.trade_history)}
Win Rate: {self.calculate_win_rate():.1%}
Run Time: {iteration * 5} minutes
Final Balance: ${self.balance:,.2f}"""
        
        send_telegram(shutdown_msg)
        logger.info("Bot shutdown complete")

# ================= HEALTH CHECK =================
async def health_check():
    """Health check endpoint for Railway"""
    app = web.Application()
    
    async def handle_health(request):
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "AI Trading Bot",
            "symbols": SYMBOLS,
            "timeframe": TIMEFRAME
        })
    
    app.router.add_get('/health', handle_health)
    app.router.add_get('/', handle_health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8080)))
    await site.start()
    
    logger.info(f"Health check server started on port {os.getenv('PORT', 8080)}")

# ================= MAIN =================
async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("🤖 AI TRADING BOT STARTING")
    logger.info("=" * 60)
    
    # Start health check
    health_task = asyncio.create_task(health_check())
    
    # Start trading bot
    bot = TradingBot()
    
    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        send_telegram(f"💥 Bot crashed: {str(e)[:100]}")
    finally:
        health_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
