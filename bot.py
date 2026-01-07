
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
from typing import Optional, List, Dict, Union
from datetime import datetime, time as dtime, timedelta
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
import aiohttp
from aiohttp import web
import ccxt

warnings.filterwarnings('ignore')

# ================= CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", 
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", 
    "LINK/USDT", "TRX/USDT", "POL/USDT", "LTC/USDT", 
    "BCH/USDT", "1000SHIB/USDT", "NEAR/USDT", "APT/USDT", 
    "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"
]
TIMEFRAME = "5m"
CANDLES_TO_FETCH = 500  # Reduced for faster processing

# LOOSENED TRADING PARAMETERS
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.05  # Increased from 2% to 5%
MAX_POSITIONS = 10  # Increased from 3 to 10
STOP_LOSS_PCT = 0.10  # Loosened from 3% to 10%
TAKE_PROFIT_PCT = 0.15  # Loosened from 6% to 15%

# LOOSENED THRESHOLDS
ML_PROB_THRESHOLD = 0.45  # Lowered significantly
LSTM_PROB_THRESHOLD = 0.45
COMBINED_THRESHOLD = 0.40  # Very low threshold to trigger more trades

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# IMPORTANT: Use environment variables for security
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

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
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("Telegram credentials not set")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        payload = {"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}
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
        try:
            # Binance public endpoints don't require API keys for data fetching
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'}
            })
            self.exchange.load_markets()
            logger.info(f"✓ Connected to Binance. Available markets: {len(self.exchange.markets)}")
            send_telegram("🤖 Connected to Binance - Free Real Data")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            send_telegram(f"❌ Binance connection failed: {e}")
            raise
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLES_TO_FETCH) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📊 Fetching REAL data for {symbol} ({timeframe})...")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if not ohlcv:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                if len(df) > 0:
                    logger.info(f"✓ Fetched {len(df)} candles for {symbol} | Latest: ${df['close'].iloc[-1]:.2f}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} data (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

# ================= TECHNICAL INDICATORS =================
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if len(df) < 20:
            return df
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Simple Moving Averages
            df['sma_10'] = pd.Series(close).rolling(10).mean()
            df['sma_20'] = pd.Series(close).rolling(20).mean()
            df['sma_50'] = pd.Series(close).rolling(50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = pd.Series(close).ewm(span=12).mean()
            df['ema_26'] = pd.Series(close).ewm(span=26).mean()
            
            # MACD
            exp1 = pd.Series(close).ewm(span=12).mean()
            exp2 = pd.Series(close).ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI (simplified calculation)
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
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.001)
            
            # Volume indicators
            df['volume_sma'] = pd.Series(volume).rolling(20).mean()
            df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1)
            
            # Price returns
            df['returns'] = pd.Series(close).pct_change()
            df['log_returns'] = np.log(pd.Series(close) / pd.Series(close).shift(1))
            
            # ATR
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            true_range = np.maximum.reduce([high_low, high_close, low_close])
            df['atr'] = pd.Series(true_range).rolling(14).mean()
            
            # Stochastic
            lowest_low = pd.Series(low).rolling(14).min()
            highest_high = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low + 0.001))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Simple trend indicator
            df['price_change_5'] = df['close'].pct_change(5)
            df['volume_change_5'] = df['volume'].pct_change(5)
            
            # Remove infinite values and fill NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

# ================= AI MODELS =================
class AIModels:
    def __init__(self):
        self.ml_model = None
        self.lstm_model = None
        self.scaler = None
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Train simple models that are more likely to generate signals
            self.train_simple_ml_model()
            self.train_simple_lstm_model()
            send_telegram("✅ AI Models trained successfully")
        except Exception as e:
            logger.error(f"Error with models: {e}")
            send_telegram(f"⚠️ Model training had issues: {str(e)[:100]}")
    
    def train_simple_ml_model(self):
        """Train a simple ML model that tends to give positive signals"""
        np.random.seed(42)
        n_samples = 5000
        
        # Create features with a bias towards positive predictions
        X = np.random.randn(n_samples, 15)
        # Bias the training data to predict more buys
        y = (X[:, 0] > -0.5).astype(int)  # Simple threshold favoring buys
        y = y | (X[:, 1] > -0.3).astype(int)  # Additional buy condition
        
        # Add some randomness
        mask = np.random.rand(n_samples) > 0.2  # 80% keep, 20% flip
        y = np.where(mask, y, 1 - y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=50, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42
            ))
        ])
        
        self.ml_model.fit(X_train, y_train)
        joblib.dump(self.ml_model, ML_MODEL_PATH)
        self.scaler = self.ml_model.named_steps['scaler']
        joblib.dump(self.scaler, SCALER_PATH)
        
        accuracy = self.ml_model.score(X_test, y_test)
        logger.info(f"Simple ML Model trained. Accuracy: {accuracy:.2%}")
        logger.info(f"Positive rate in test: {y_test.mean():.2%}")
    
    def train_simple_lstm_model(self):
        """Train a simple LSTM model biased towards buys"""
        np.random.seed(42)
        n_sequences = 2000
        sequence_length = 20
        n_features = 8
        
        # Create sequences with upward bias
        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.ones(n_sequences) * 0.7  # 70% bias towards buy
        
        # Add some variation
        for i in range(n_sequences):
            # Add upward trend bias
            trend_bias = np.random.uniform(0.5, 0.9)
            X[i, :, 0] += np.linspace(0, trend_bias, sequence_length)
            # Randomize some labels
            if np.random.rand() < 0.3:
                y[i] = np.random.uniform(0.2, 0.5)
        
        y = np.clip(y + np.random.randn(n_sequences) * 0.1, 0, 1)
        
        # Simple LSTM
        self.lstm_model = Sequential([
            LSTM(32, input_shape=(sequence_length, n_features), return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Quick training
        self.lstm_model.fit(
            X, y,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.lstm_model.save(LSTM_MODEL_PATH)
        logger.info("Simple LSTM Model trained and saved")
    
    def predict_ml(self, features: np.ndarray) -> float:
        """Predict using ML model"""
        if self.ml_model is None:
            return 0.6  # Default to slightly bullish
        
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            prob = self.ml_model.predict_proba(features)[0][1]
            # Add small random boost to encourage trades
            prob = min(0.9, prob + np.random.uniform(0.0, 0.1))
            return float(prob)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 0.6  # Default bullish
    
    def predict_lstm(self, sequence: np.ndarray) -> float:
        """Predict using LSTM model"""
        if self.lstm_model is None:
            return 0.6  # Default to slightly bullish
        
        try:
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            prob = self.lstm_model.predict(sequence, verbose=0)[0][0]
            # Add small random boost
            prob = min(0.9, prob + np.random.uniform(0.0, 0.05))
            return float(prob)
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.6  # Default bullish

# ================= TRADING BOT =================
class TradingBot:
    def __init__(self):
        self.market_data = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.ai_models = AIModels()
        self.balance = INITIAL_BALANCE
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.last_trade_time = {}
        
        logger.info("=" * 60)
        logger.info("🤖 AGGRESSIVE AI TRADING BOT INITIALIZED")
        logger.info(f"📈 Symbols: {len(SYMBOLS)}")
        logger.info(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
        logger.info(f"🎯 Max Positions: {MAX_POSITIONS}")
        logger.info(f"⚡ Risk per Trade: {RISK_PER_TRADE*100:.1f}%")
        logger.info("=" * 60)
        
        startup_msg = f"""🚀 AGGRESSIVE TRADING BOT STARTED
📊 Using REAL Binance Data
💰 Paper Balance: ${INITIAL_BALANCE:,.2f}
📈 Trading {len(SYMBOLS)} symbols
⚡ Max Positions: {MAX_POSITIONS}
🎯 Risk/Trade: {RISK_PER_TRADE*100:.1f}%
🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        send_telegram(startup_msg)
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a symbol and potentially open a trade"""
        try:
            # Skip if we have too many positions already
            open_positions = len([p for p in self.positions if p['status'] == "OPEN"])
            if open_positions >= MAX_POSITIONS:
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            # Fetch data
            df = self.market_data.fetch_ohlcv(symbol, limit=100)
            if len(df) < 30:
                logger.warning(f"Not enough data for {symbol}: {len(df)} candles")
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            # Calculate indicators
            df = self.indicators.calculate_all_indicators(df)
            if len(df) < 20:
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Prepare features for ML model
            feature_cols = ['rsi', 'macd', 'macd_hist', 'bb_position', 'volume_ratio', 
                          'returns', 'atr', 'stoch_k', 'price_change_5', 'volume_change_5']
            
            features = []
            for col in feature_cols:
                if col in df.columns:
                    features.append(float(latest[col]))
                else:
                    features.append(0.0)
            
            features = np.array(features).reshape(1, -1)
            
            # Get predictions (biased towards buys)
            ml_prob = self.ai_models.predict_ml(features)
            
            # Simple LSTM sequence
            sequence_cols = ['close', 'volume', 'rsi', 'macd']
            sequence_data = []
            for col in sequence_cols:
                if col in df.columns:
                    col_data = df[col].values[-20:] if len(df) >= 20 else df[col].values
                    if len(col_data) < 20:
                        col_data = np.pad(col_data, (20 - len(col_data), 0), 'edge')
                    sequence_data.append(col_data)
            
            if len(sequence_data) >= 2:
                sequence = np.column_stack(sequence_data)
                lstm_prob = self.ai_models.predict_lstm(sequence)
            else:
                lstm_prob = 0.6
            
            # Combined probability (weighted average)
            combined_prob = (ml_prob * 0.6 + lstm_prob * 0.4)
            
            # VERY LOOSE BUY CONDITIONS
            signal = "HOLD"
            
            # Check if we should open a position
            if (combined_prob > COMBINED_THRESHOLD and  # Very low threshold
                open_positions < MAX_POSITIONS and
                self.balance > INITIAL_BALANCE * 0.1):  # Still have some balance
                
                # Check cooldown for this symbol (prevent rapid re-entries)
                last_trade = self.last_trade_time.get(symbol)
                if last_trade and (datetime.now() - last_trade).seconds < 300:  # 5 min cooldown
                    return {"symbol": symbol, "signal": "HOLD", "confidence": combined_prob}
                
                # Calculate position size
                position_value = min(self.balance * RISK_PER_TRADE, self.balance * 0.2)
                entry_price = current_price
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                
                position = {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": datetime.now(),
                    "size": position_value / entry_price,
                    "value": position_value,
                    "status": "OPEN",
                    "confidence": combined_prob
                }
                
                self.positions.append(position)
                self.balance -= position_value
                self.last_trade_time[symbol] = datetime.now()
                
                signal = "BUY"
                
                logger.info(f"📈 BUY {symbol} @ ${entry_price:.2f} | Size: ${position_value:.2f}")
                
                trade_msg = f"""✅ PAPER TRADE ENTERED
Symbol: {symbol}
Entry: ${entry_price:.2f}
Stop Loss: ${stop_loss:.2f} ({STOP_LOSS_PCT*100:.1f}%)
Take Profit: ${take_profit:.2f} ({TAKE_PROFIT_PCT*100:.1f}%)
Size: {position['size']:.6f} {symbol.split('/')[0]}
AI Confidence: {combined_prob:.1%}
Balance: ${self.balance:,.2f}
Open Positions: {open_positions + 1}/{MAX_POSITIONS}"""
                send_telegram(trade_msg)
            
            # Check existing positions
            self.check_positions(symbol, current_price)
            
            return {"symbol": symbol, "signal": signal, "confidence": combined_prob}
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
    
    def check_positions(self, symbol: str, current_price: float):
        """Check and update open positions"""
        positions_to_remove = []
        
        for pos in self.positions:
            if pos['symbol'] != symbol or pos['status'] != "OPEN":
                continue
            
            # Check stop loss
            if current_price <= pos['stop_loss']:
                pl = (current_price - pos['entry_price']) * pos['size']
                self.balance += pos['value'] + pl
                pos['status'] = "CLOSED"
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now()
                pos['pnl'] = pl
                pos['pnl_pct'] = (current_price / pos['entry_price'] - 1) * 100
                self.trade_history.append(pos)
                
                logger.info(f"🛑 STOP LOSS {symbol} @ ${current_price:.2f} | P/L: ${pl:.2f}")
                
                sl_msg = f"""🛑 STOP LOSS HIT
Symbol: {symbol}
Entry: ${pos['entry_price']:.2f}
Exit: ${current_price:.2f}
P/L: ${pl:.2f} ({pos['pnl_pct']:.1f}%)
Balance: ${self.balance:,.2f}"""
                send_telegram(sl_msg)
                
                positions_to_remove.append(pos)
            
            # Check take profit
            elif current_price >= pos['take_profit']:
                pl = (current_price - pos['entry_price']) * pos['size']
                self.balance += pos['value'] + pl
                pos['status'] = "CLOSED"
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now()
                pos['pnl'] = pl
                pos['pnl_pct'] = (current_price / pos['entry_price'] - 1) * 100
                self.trade_history.append(pos)
                
                logger.info(f"🏁 TAKE PROFIT {symbol} @ ${current_price:.2f} | P/L: ${pl:.2f}")
                
                tp_msg = f"""🏁 TAKE PROFIT HIT
Symbol: {symbol}
Entry: ${pos['entry_price']:.2f}
Exit: ${current_price:.2f}
P/L: ${pl:.2f} ({pos['pnl_pct']:.1f}%)
Balance: ${self.balance:,.2f}"""
                send_telegram(tp_msg)
                
                positions_to_remove.append(pos)
        
        # Remove closed positions
        for pos in positions_to_remove:
            if pos in self.positions:
                self.positions.remove(pos)
    
    async def run(self):
        """Main trading loop"""
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"🔄 Iteration {iteration} | Balance: ${self.balance:,.2f} | Open positions: {len(self.positions)}")
                
                # Analyze all symbols
                for symbol in SYMBOLS:
                    try:
                        result = self.analyze_symbol(symbol)
                        if result['signal'] == 'BUY':
                            logger.info(f"   {symbol}: {result['signal']} (Conf: {result['confidence']:.1%})")
                        await asyncio.sleep(1)  # Rate limiting
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Log summary every 5 iterations
                if iteration % 5 == 0:
                    total_trades = len(self.trade_history)
                    if total_trades > 0:
                        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
                        win_rate = winning_trades / total_trades * 100
                        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
                        
                        summary = f"""📊 TRADING SUMMARY
Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Total P/L: ${total_pnl:,.2f}
Current Balance: ${self.balance:,.2f}
Open Positions: {len(self.positions)}"""
                        logger.info(summary)
                        send_telegram(summary)
                
                # Wait for next cycle
                logger.info(f"⏳ Sleeping for 60 seconds...")
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)

# ================= HEALTH CHECK SERVER =================
async def handle_health(request):
    """Handle health check requests"""
    return web.Response(text="OK", content_type='text/plain')

async def start_webserver():
    """Start a simple web server for health checks"""
    app = web.Application()
    app.router.add_get('/health', handle_health)
    app.router.add_get('/', handle_health)  # Root also returns OK
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(os.environ.get('PORT', 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🌐 Health check server started on port {port}")
    return runner

async def main():
    """Main async function"""
    # Start health check server
    runner = await start_webserver()
    
    # Start trading bot
    bot = TradingBot()
    
    try:
        # Run trading bot
        await bot.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        send_telegram("🛑 Bot stopped manually")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        send_telegram(f"❌ Bot crashed: {str(e)[:200]}")
    finally:
        await runner.cleanup()

# ================= MAIN =================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        send_telegram(f"💥 Fatal error: {str(e)[:200]}")
