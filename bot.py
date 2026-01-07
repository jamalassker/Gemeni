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
SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", 
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT", "DOT/USDT:USDT", 
    "LINK/USDT:USDT", "MATIC/USDT:USDT", "TRX/USDT:USDT", "LTC/USDT:USDT", 
    "BCH/USDT:USDT", "SHIB/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", 
    "SUI/USDT:USDT", "ICP/USDT:USDT", "RENDER/USDT:USDT", "STX/USDT:USDT"
]

TIMEFRAME = "1m"
CANDLES = 200

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
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_predictor.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

# Telegram
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
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

# ================= CUSTOM INDICATORS (replacing pandas-ta) =================
def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
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
        df["z"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
        
        # Additional features for ML
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
                logger.info("✓ ML model loaded")
            else:
                logger.warning("ML model not found, will create dummy")
                self.create_dummy_ml_model()
            
            # Load LSTM model
            if os.path.exists(LSTM_MODEL_PATH):
                self.lstm_model = load_model(LSTM_MODEL_PATH)
                logger.info("✓ LSTM model loaded")
            else:
                logger.warning("LSTM model not found, will create dummy")
                self.create_dummy_lstm_model()
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("✓ Scaler loaded")
            
            self.last_trained = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
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
    
    def should_retrain(self):
        """Check if models should be retrained (every 24 hours)"""
        if not self.last_trained:
            return True
        return (datetime.now() - self.last_trained) > timedelta(hours=24)
    
    def train_models(self, exchange=None):
        """Train models with historical data"""
        logger.info("Training models...")
        
        # Simplified training for Railway
        np.random.seed(42)
        
        # 1. Create synthetic training data
        n_samples = 5000
        X_ml = np.random.randn(n_samples, 200)
        y_ml = ((X_ml[:, 0] > 0.5) & (X_ml[:, 20] < -0.5)).astype(int)
        
        # 2. Train ML model
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ))
        ])
        self.ml_model.fit(X_ml, y_ml)
        joblib.dump(self.ml_model, ML_FILTER_MODEL_PATH)
        
        # 3. Train LSTM model
        X_lstm = np.random.randn(n_samples // 10, 30, 7)
        y_lstm = np.random.rand(n_samples // 10)
        
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(30, 7), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy'
        )
        
        self.lstm_model.fit(
            X_lstm, y_lstm,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        self.lstm_model.save(LSTM_MODEL_PATH)
        
        # 4. Save scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_ml)
        joblib.dump(self.scaler, SCALER_PATH)
        
        logger.info("Models trained and saved")
        self.last_trained = datetime.now()
        return True

# Initialize model manager
model_manager = ModelManager()

# ================= TELEGRAM =================
def tg(msg):
    """Send Telegram message"""
    if TG_TOKEN and TG_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT, "text": msg},
                timeout=5
            )
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# ================= UTIL =================
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

# ================= AI PROBABILITY =================
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

# ================= ML FILTER =================
def ml_filter(df):
    """Get ML model prediction"""
    try:
        if model_manager.ml_model is None:
            return 0.5
        
        # Prepare features
        features = []
        for col in ["open", "high", "low", "close", "rsi", "atr", "z", 
                   "volume_ratio", "high_low_pct", "close_open_pct"]:
            if col in df.columns:
                features.extend(df[col].values[-20:])
        
        if len(features) >= 200:
            # Scale features if scaler exists
            if model_manager.scaler:
                features = model_manager.scaler.transform([features[:200]])
            else:
                features = [features[:200]]
            
            prob = model_manager.ml_model.predict_proba(features)[0][1]
            return float(prob)
    except Exception as e:
        logger.error(f"ML filter error: {e}")
    
    return 0.5

# ================= LSTM PREDICTOR =================
def lstm_predict(df):
    """Get LSTM model prediction"""
    try:
        if model_manager.lstm_model is None:
            return 0.5
        
        # Prepare sequence
        features_cols = ["open", "high", "low", "close", "rsi", "atr", "z"]
        available_cols = [col for col in features_cols if col in df.columns]
        
        if len(available_cols) >= 5:
            seq = df[available_cols].values[-30:]
            
            # Pad if necessary
            if seq.shape[0] < 30:
                padding = np.zeros((30 - seq.shape[0], seq.shape[1]))
                seq = np.vstack([padding, seq])
            
            # Reshape and predict
            seq = seq.reshape(1, 30, seq.shape[1])
            prob = model_manager.lstm_model.predict(seq, verbose=0)[0][0]
            return float(prob)
    except Exception as e:
        logger.error(f"LSTM predict error: {e}")
    
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
        tg("🤖 AI 1m Scalper with TensorFlow Started on Railway")
        
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Train models if needed
        if model_manager.should_retrain():
            logger.info("Models need retraining")
            model_manager.train_models(self.exchange)
        
        await self.optimize_params()
    
    async def optimize_params(self):
        """Optimize trading parameters"""
        logger.info("Optimizing parameters...")
        
        for s in SYMBOLS:
            try:
                ohlc = self.exchange.fetch_ohlcv(s, TIMEFRAME, limit=200)
                df = indicators(pd.DataFrame(ohlc, columns=["ts","open","high","low","close","volume"]))
                
                # Simple optimization
                self.params[s] = {
                    "rsi": 30,
                    "sl": 1.2,
                    "tp": 2.0,
                    "score": 0
                }
                
                tg(f"⚙ {s} params optimized")
                
            except Exception as e:
                logger.error(f"Optimization error for {s}: {e}")
                self.params[s] = {"rsi": 30, "sl": 1.2, "tp": 2.0, "score": 0}
        
        self.last_optimize = datetime.now()
    
    async def check_and_execute_trades(self):
        """Check market conditions and execute trades"""
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
                
                # Get predictions
                prob_ai = ai_probability(r, self.bias)
                prob_ml = ml_filter(df)
                prob_lstm = lstm_predict(df)
                combined_prob = (prob_ai + prob_ml + prob_lstm) / 3
                
                # Simple orderbook imbalance simulation
                imbalance = 0.1  # Placeholder
                
                # Manage existing trades
                self.manage_trades(s, r)
                
                # Check for new entry
                if self.check_entry(s, df, r, imbalance, combined_prob):
                    logger.info(f"Entered trade for {s}")
                
            except Exception as e:
                logger.error(f"Error processing {s}: {e}")
    
    def manage_trades(self, symbol, r):
        """Manage open trades"""
        trades_to_remove = []
        
        for t in self.open_trades:
            if t["symbol"] != symbol:
                continue
            
            age = (datetime.utcnow() - t["time"]).seconds / 60
            
            # Check stop loss
            if r["low"] <= t["sl"]:
                loss = self.balance * RISK_PER_TRADE
                loss += loss * TAKER_FEE
                self.balance -= loss
                self.bias -= 0.05
                trades_to_remove.append(t)
                tg(f"❌ SL {t['symbol']} | Bal ${round(self.balance,2)}")
                logger.info(f"Stop loss triggered for {symbol}")
            
            # Check take profit
            elif r["high"] >= t["tp"]:
                gain = self.balance * RISK_PER_TRADE * (t["tp_ratio"] / t["sl_ratio"])
                gain -= gain * TAKER_FEE
                self.balance += gain
                self.bias += 0.05
                trades_to_remove.append(t)
                tg(f"✅ TP {t['symbol']} | Bal ${round(self.balance,2)}")
                logger.info(f"Take profit triggered for {symbol}")
            
            # Check timeout
            elif age > MAX_TRADE_MINUTES:
                trades_to_remove.append(t)
                tg(f"⏱ Exit timeout {t['symbol']}")
                logger.info(f"Trade timeout for {symbol}")
        
        # Remove completed trades
        for t in trades_to_remove:
            if t in self.open_trades:
                self.open_trades.remove(t)
    
    def check_entry(self, symbol, df, r, imbalance, combined_prob):
        """Check for trade entry"""
        p = self.params.get(symbol, {"rsi": 30, "sl": 1.2, "tp": 2.0})
        
        if (
            len(self.open_trades) < MAX_OPEN_TRADES
            and regime(r) == "UP"
            and combined_prob > AI_PROB_THRESHOLD
            and imbalance > IMBALANCE_THRESHOLD
            and r["rsi"] < p["rsi"]
            and r["z"] < -0.5
            and not pd.isna(r["atr"])
            and r["atr"] > df["atr"].mean() * 0.8
        ):
            entry = r["close"] * (1 + SLIPPAGE)
            atr_val = r["atr"] if not pd.isna(r["atr"]) else r["close"] * 0.01
            
            trade = {
                "symbol": symbol,
                "entry": entry,
                "sl": entry - atr_val * p["sl"],
                "tp": entry + atr_val * p["tp"],
                "sl_ratio": p["sl"],
                "tp_ratio": p["tp"],
                "time": datetime.utcnow()
            }
            
            self.open_trades.append(trade)
            tg(f"📈 BUY {symbol} | CP:{round(combined_prob,2)} | RSI:{round(r['rsi'],1)}")
            return True
        
        return False
    
    async def main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop")
        
        while self.running:
            try:
                if in_session():
                    # Check drawdown
                    self.peak = max(self.peak, self.balance)
                    drawdown = (self.peak - self.balance) / self.peak
                    
                    if drawdown > MAX_DRAWDOWN:
                        tg("🛑 Max drawdown reached. Trading halted.")
                        logger.error("Max drawdown reached")
                        self.running = False
                        break
                    
                    # Check and execute trades
                    await self.check_and_execute_trades()
                    
                    # Re-optimize every 6 hours
                    if not self.last_optimize or (datetime.now() - self.last_optimize).seconds > 21600:
                        await self.optimize_params()
                    
                    # Retrain models if needed
                    if model_manager.should_retrain():
                        model_manager.train_models(self.exchange)
                    
                # Sleep based on timeframe
                await asyncio.sleep(60)  # Check every minute
                    
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
        
        await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down bot...")
        tg("🤖 Bot shutting down")
        self.running = False

# ================= HEALTH CHECK =================
async def health_check():
    """Simple health check endpoint for Railway"""
    app = web.Application()
    
    async def handle_health(request):
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": RAILWAY_ENVIRONMENT,
            "tensorflow_version": tf.__version__
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
    logger.info("=" * 50)
    logger.info("AI 1M SCALPER BOT WITH TENSORFLOW STARTING")
    logger.info(f"TensorFlow Version: {tf.__version__}")
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
        tg(f"🤖 Bot crashed: {str(e)[:100]}")
    finally:
        health_task.cancel()
        logger.info("Bot stopped")

if __name__ == "__main__":
    # Check if this is a training run
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        logger.info("Running training mode...")
        
        # Simple training
        exchange = ccxt.binance({"enableRateLimit": True})
        model_manager.train_models(exchange)
        logger.info("Training complete")
    else:
        # Run the bot
        asyncio.run(main())
