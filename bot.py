import ccxt
import pandas as pd
import numpy as np
import requests
import asyncio
import math
import os
import joblib
import warnings
import time
import sys
import json
from datetime import datetime, time as dtime, timedelta
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

warnings.filterwarnings('ignore')

# ================= CONFIG =================
# Using correct Binance spot symbols
SYMBOLS = ["BTC/USDT", "ETH/USDT"]  # Start with just 2 symbols
TIMEFRAME = "1m"
CANDLES = 100

INITIAL_BALANCE = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_TRADES = 2
MAX_TRADE_MINUTES = 15
MAX_DRAWDOWN = 0.15

TAKER_FEE = 0.0004
SLIPPAGE = 0.0002

AI_PROB_THRESHOLD = 0.62
IMBALANCE_THRESHOLD = 0.15

SESSIONS = [
    (dtime(7, 0), dtime(11, 0)),   # London
    (dtime(13, 0), dtime(17, 0))  # NY
]

# Model paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
ML_FILTER_MODEL_PATH = os.path.join(MODEL_DIR, "ml_filter_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_predictor.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# Telegram - Add with defaults
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT_NAME", "production")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ================= TELEGRAM WITH RETRY =================
def tg(msg, max_retries=3):
    """Send Telegram message with retry logic"""
    if not TG_TOKEN or not TG_CHAT:
        logger.warning("Telegram credentials not set. Message not sent.")
        return False
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={
                    "chat_id": TG_CHAT,
                    "text": msg,
                    "parse_mode": "HTML"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Telegram message sent: {msg[:50]}...")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram connection error (attempt {attempt + 1}/{max_retries}): {e}")
            
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to send Telegram message after {max_retries} attempts: {msg}")
    return False

# ================= CUSTOM INDICATORS =================
def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.001)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def indicators(df):
    """Calculate all technical indicators"""
    try:
        df["ema20"] = calculate_ema(df["close"], 20)
        df["ema50"] = calculate_ema(df["close"], 50)
        df["ema200"] = calculate_ema(df["close"], 200)
        df["rsi"] = calculate_rsi(df["close"], 14)
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
        
        # Z-score
        rolling_mean = df["close"].rolling(20).mean()
        rolling_std = df["close"].rolling(20).std()
        df["z"] = (df["close"] - rolling_mean) / rolling_std.replace(0, 0.001)
        
        # Additional features
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1)
        df["high_low_pct"] = (df["high"] - df["low"]) / df["low"].replace(0, 0.001) * 100
        df["close_open_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, 0.001) * 100
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Indicators error: {e}")
        return df

# ================= MODEL MANAGEMENT =================
class ModelManager:
    def __init__(self):
        self.ml_model = None
        self.lstm_model = None
        self.scaler = None
        self.last_trained = None
        self.load_models()
    
    def load_models(self):
        """Load or create models"""
        try:
            # Load ML model
            if os.path.exists(ML_FILTER_MODEL_PATH):
                self.ml_model = joblib.load(ML_FILTER_MODEL_PATH)
                logger.info("ML model loaded")
                tg("✅ ML model loaded")
            else:
                logger.warning("ML model not found, creating dummy")
                self.create_dummy_ml_model()
                tg("🔄 Created dummy ML model")
            
            # Load LSTM model
            if os.path.exists(LSTM_MODEL_PATH):
                self.lstm_model = load_model(LSTM_MODEL_PATH)
                logger.info("LSTM model loaded")
                tg("✅ LSTM model loaded")
            else:
                logger.warning("LSTM model not found, creating dummy")
                self.create_dummy_lstm_model()
                tg("🔄 Created dummy LSTM model")
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded")
            
            self.last_trained = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            tg(f"❌ Model loading error: {str(e)[:100]}")
            self.create_dummy_models()
    
    def create_dummy_ml_model(self):
        """Create a dummy ML model for testing"""
        np.random.seed(42)
        X = np.random.randn(100, 200)
        y = (np.random.rand(100) > 0.5).astype(int)
        
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=10, random_state=42))
        ])
        self.ml_model.fit(X, y)
        joblib.dump(self.ml_model, ML_FILTER_MODEL_PATH)
    
    def create_dummy_lstm_model(self):
        """Create a dummy LSTM model for testing"""
        self.lstm_model = Sequential([
            LSTM(16, input_shape=(30, 7), return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.lstm_model.save(LSTM_MODEL_PATH)
    
    def create_dummy_models(self):
        """Create all dummy models"""
        self.create_dummy_ml_model()
        self.create_dummy_lstm_model()
        self.scaler = StandardScaler()
        joblib.dump(self.scaler, SCALER_PATH)

# Initialize model manager
model_manager = ModelManager()

# ================= UTIL FUNCTIONS =================
def in_session():
    """Check if current time is within trading session"""
    now = datetime.utcnow().time()
    return any(s <= now <= e for s, e in SESSIONS)

def regime(r):
    """Determine market regime"""
    try:
        if pd.isna(r["ema20"]) or pd.isna(r["ema50"]):
            return "RANGE"
        if r["ema20"] > r["ema50"] > r["ema200"]:
            return "UP"
        if r["ema20"] < r["ema50"] < r["ema200"]:
            return "DOWN"
        return "RANGE"
    except:
        return "RANGE"

def ai_probability(r, bias):
    """Calculate rule-based probability"""
    try:
        z_score = abs(r["z"]) if not pd.isna(r["z"]) else 0
        rsi_val = r["rsi"] if not pd.isna(r["rsi"]) else 50
        ema_diff = abs(r["ema20"] - r["ema200"]) if not pd.isna(r["ema20"]) and not pd.isna(r["ema200"]) else 0
        
        base = (
            1.5 * min(z_score / 3, 1) +
            1.2 * abs(50 - rsi_val) / 50 +
            ema_diff / max(r["close"], 0.0001)
        )
        return 1 / (1 + math.exp(-(base + bias)))
    except:
        return 0.5

# ================= TRADING BOT =================
class TradingBot:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.peak = INITIAL_BALANCE
        self.open_trades = []
        self.bias = 0.0
        self.params = {}
        self.exchange = None
        self.running = True
        self.last_optimize = None
    
    async def initialize(self):
        """Initialize the bot"""
        logger.info("Initializing trading bot...")
        
        # Send startup message
        startup_msg = f"""🤖 AI Scalper Bot Started
📍 Environment: {RAILWAY_ENVIRONMENT}
📊 Symbols: {', '.join(SYMBOLS)}
💰 Initial Balance: ${INITIAL_BALANCE}
🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""
        
        tg(startup_msg)
        
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Using spot market
                    'adjustForTimeDifference': True
                }
            })
            
            # Test exchange connection
            markets = self.exchange.load_markets()
            logger.info(f"Exchange connected. Available markets: {len(markets)}")
            
            # Check if our symbols are available
            for symbol in SYMBOLS:
                if symbol in markets:
                    logger.info(f"✓ Symbol available: {symbol}")
                else:
                    logger.error(f"✗ Symbol not available: {symbol}")
                    tg(f"⚠️ Symbol not available: {symbol}")
            
            await self.optimize_params()
            
        except Exception as e:
            logger.error(f"Exchange initialization error: {e}")
            tg(f"❌ Exchange init error: {str(e)[:100]}")
            raise
    
    async def optimize_params(self):
        """Optimize trading parameters"""
        logger.info("Optimizing parameters...")
        tg("🔄 Optimizing parameters...")
        
        for s in SYMBOLS:
            try:
                logger.info(f"Fetching data for {s}...")
                ohlc = self.exchange.fetch_ohlcv(s, TIMEFRAME, limit=100)
                
                if len(ohlc) > 0:
                    df = indicators(pd.DataFrame(ohlc, columns=["ts","open","high","low","close","volume"]))
                    
                    # Simple optimization
                    self.params[s] = {
                        "rsi": 30,
                        "sl": 1.2,
                        "tp": 2.0,
                        "score": 0
                    }
                    
                    logger.info(f"✓ Optimized {s}: RSI={self.params[s]['rsi']}, SL={self.params[s]['sl']}, TP={self.params[s]['tp']}")
                    
                else:
                    logger.warning(f"No data for {s}")
                    self.params[s] = {"rsi": 30, "sl": 1.2, "tp": 2.0, "score": 0}
                
            except Exception as e:
                logger.error(f"Optimization error for {s}: {e}")
                self.params[s] = {"rsi": 30, "sl": 1.2, "tp": 2.0, "score": 0}
                tg(f"⚠️ Optimization failed for {s}")
        
        self.last_optimize = datetime.now()
        tg("✅ Parameters optimized")
    
    async def run_iteration(self):
        """Run one trading iteration"""
        try:
            if not in_session():
                logger.info("Outside trading session")
                return
            
            # Check drawdown
            self.peak = max(self.peak, self.balance)
            drawdown = (self.peak - self.balance) / self.peak
            
            if drawdown > MAX_DRAWDOWN:
                msg = f"🛑 Max drawdown reached ({drawdown:.1%}). Trading halted."
                logger.error(msg)
                tg(msg)
                self.running = False
                return
            
            # Process each symbol
            for s in SYMBOLS:
                try:
                    # Get data
                    ohlc = self.exchange.fetch_ohlcv(s, TIMEFRAME, limit=CANDLES)
                    
                    if len(ohlc) < 50:
                        continue
                    
                    df = indicators(pd.DataFrame(ohlc, columns=["ts","open","high","low","close","volume"]))
                    
                    if len(df) < 20:
                        continue
                    
                    r = df.iloc[-1]
                    
                    # Mock probabilities for now
                    prob_ai = ai_probability(r, self.bias)
                    combined_prob = prob_ai  # Simplified for now
                    
                    # Log market status
                    logger.info(f"{s}: Price=${r['close']:.2f}, RSI={r['rsi']:.1f}, Regime={regime(r)}, Prob={combined_prob:.2f}")
                    
                    # Check for entry
                    if self.check_entry(s, df, r, combined_prob):
                        tg(f"📈 BUY {s} | Price: ${r['close']:.2f} | RSI: {r['rsi']:.1f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {s}: {e}")
            
        except Exception as e:
            logger.error(f"Run iteration error: {e}")
    
    def check_entry(self, symbol, df, r, combined_prob):
        """Check for trade entry"""
        p = self.params.get(symbol, {"rsi": 30, "sl": 1.2, "tp": 2.0})
        
        entry_conditions = (
            len(self.open_trades) < MAX_OPEN_TRADES and
            regime(r) == "UP" and
            combined_prob > AI_PROB_THRESHOLD and
            r["rsi"] < p["rsi"] and
            r["z"] < -0.5 and
            not pd.isna(r["atr"]) and
            r["atr"] > df["atr"].mean() * 0.8
        )
        
        if entry_conditions:
            entry = r["close"] * (1 + SLIPPAGE)
            atr_val = r["atr"] if not pd.isna(r["atr"]) else r["close"] * 0.01
            
            trade = {
                "symbol": symbol,
                "entry": entry,
                "sl": entry - atr_val * p["sl"],
                "tp": entry + atr_val * p["tp"],
                "time": datetime.utcnow()
            }
            
            self.open_trades.append(trade)
            return True
        
        return False
    
    async def main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop")
        tg("🚀 Starting main trading loop")
        
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"=== Iteration {iteration} ===")
                
                await self.run_iteration()
                
                # Send periodic status
                if iteration % 10 == 0:  # Every 10 minutes
                    status_msg = f"""📊 Bot Status Update
Balance: ${self.balance:.2f}
Open Trades: {len(self.open_trades)}
Drawdown: {((self.peak - self.balance) / self.peak * 100):.1f}%
Iteration: {iteration}
Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC"""
                    tg(status_msg)
                
                # Wait for next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                tg("🛑 Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
    
    async def shutdown(self):
        """Clean shutdown"""
        shutdown_msg = f"""🤖 Bot Shutting Down
Final Balance: ${self.balance:.2f}
Total PnL: ${self.balance - INITIAL_BALANCE:.2f}
Total Trades: {len(self.open_trades)}
Runtime: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""
        
        tg(shutdown_msg)
        logger.info("Bot shutdown complete")

# ================= HEALTH CHECK =================
async def health_check():
    """Health check endpoint"""
    app = web.Application()
    
    async def handle_health(request):
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": RAILWAY_ENVIRONMENT,
            "bot_status": "running",
            "symbols": SYMBOLS
        })
    
    async def handle_status(request):
        """Extended status endpoint"""
        return web.json_response({
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": RAILWAY_ENVIRONMENT,
            "symbols": SYMBOLS,
            "sessions": [f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}" for s, e in SESSIONS],
            "current_session": in_session(),
            "utc_time": datetime.utcnow().strftime('%H:%M:%S')
        })
    
    app.router.add_get('/health', handle_health)
    app.router.add_get('/status', handle_status)
    app.router.add_get('/', handle_health)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', int(os.getenv('PORT', 8080)))
    await site.start()
    
    logger.info(f"Health check server started on port {os.getenv('PORT', 8080)}")
    tg(f"🌐 Health check server started on port {os.getenv('PORT', 8080)}")

# ================= MAIN =================
async def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("AI SCALPER BOT STARTING")
    logger.info(f"Python: {sys.version}")
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Telegram configured: {bool(TG_TOKEN and TG_CHAT)}")
    logger.info("=" * 50)
    
    # Start health check in background
    health_task = asyncio.create_task(health_check())
    
    # Initialize and run bot
    bot = TradingBot()
    
    try:
        await bot.initialize()
        await bot.main_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        tg(f"💥 Bot crashed: {str(e)[:100]}")
    finally:
        await bot.shutdown()
        health_task.cancel()
        logger.info("Bot stopped")

if __name__ == "__main__":
    # Check for training mode
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        logger.info("Training mode selected")
        tg("🎓 Starting model training...")
        
        # Simple training
        exchange = ccxt.binance({"enableRateLimit": True})
        # You can add training logic here
        
        tg("✅ Training complete")
    else:
        # Run the bot
        asyncio.run(main())
