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
CANDLES_TO_FETCH = 1000

INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.02
MAX_POSITIONS = 3
STOP_LOSS_PCT = 0.03  # Loosened from 1.5% to 3%
TAKE_PROFIT_PCT = 0.06  # Loosened from 3% to 6%

ML_PROB_THRESHOLD = 0.55  # lowered for more trades
LSTM_PROB_THRESHOLD = 0.55
COMBINED_THRESHOLD = 0.55

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

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
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 30000})
        try:
            self.exchange.load_markets()
            logger.info(f"✓ Connected to Binance. Available markets: {len(self.exchange.markets)}")
            send_telegram("🤖 Connected to Binance - Free Real Data")
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
            send_telegram(f"❌ Binance connection failed: {e}")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = TIMEFRAME, limit: int = CANDLES_TO_FETCH) -> pd.DataFrame:
        try:
            logger.info(f"📊 Fetching REAL data for {symbol} ({timeframe})...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"✓ Fetched {len(df)} candles for {symbol} | Latest: ${df['close'].iloc[-1]:.2f}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def fetch_order_book(self, symbol: str, limit: int = 10):
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None

# ================= TECHNICAL INDICATORS =================
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 50:
            return df
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            df['sma_20'] = pd.Series(close).rolling(20).mean()
            df['sma_50'] = pd.Series(close).rolling(50).mean()
            df['sma_200'] = pd.Series(close).rolling(200).mean()
            df['ema_12'] = pd.Series(close).ewm(span=12).mean()
            df['ema_26'] = pd.Series(close).ewm(span=26).mean()
            
            exp1 = pd.Series(close).ewm(span=12).mean()
            exp2 = pd.Series(close).ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.001)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            sma_20 = df['sma_20']
            rolling_std = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma_20 + (rolling_std * 2)
            df['bb_lower'] = sma_20 - (rolling_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['volume_sma'] = pd.Series(volume).rolling(20).mean()
            df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1)
            
            df['returns'] = pd.Series(close).pct_change()
            df['log_returns'] = np.log(pd.Series(close) / pd.Series(close).shift(1))
            
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            true_range = np.maximum.reduce([high_low, high_close, low_close])
            df['atr'] = pd.Series(true_range).rolling(14).mean()
            
            lowest_low = pd.Series(low).rolling(14).min()
            highest_high = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, 0.001))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
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
        try:
            if os.path.exists(ML_MODEL_PATH):
                self.ml_model = joblib.load(ML_MODEL_PATH)
                logger.info("✓ Loaded ML model from disk")
            else:
                logger.info("Training ML model...")
                self.train_ml_model()
            
            if os.path.exists(LSTM_MODEL_PATH):
                self.lstm_model = load_model(LSTM_MODEL_PATH)
                logger.info("✓ Loaded LSTM model from disk")
            else:
                logger.info("Training LSTM model...")
                self.train_lstm_model()
            
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("✓ Loaded scaler from disk")
                
            send_telegram("✅ AI Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            send_telegram(f"❌ Model loading error: {str(e)[:100]}")
    
    def train_ml_model(self):
        np.random.seed(42)
        n_samples = 10000
        X = np.random.randn(n_samples, 20)
        y = ((X[:, 0] > 0.5) & (X[:, 1] < 0.3) & (X[:, 2] > 0.7)).astype(int)
        y = y ^ (np.random.rand(n_samples) > 0.9)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.ml_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42))
        ])
        self.ml_model.fit(X_train, y_train)
        joblib.dump(self.ml_model, ML_MODEL_PATH)
        self.scaler = self.ml_model.named_steps['scaler']
        joblib.dump(self.scaler, SCALER_PATH)
        accuracy = self.ml_model.score(X_test, y_test)
        logger.info(f"ML Model trained. Accuracy: {accuracy:.2%}")
    
    def train_lstm_model(self):
        np.random.seed(42)
        n_sequences = 5000
        sequence_length = 30
        n_features = 10
        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.zeros(n_sequences)
        for i in range(n_sequences):
            trend = np.mean(X[i, -5:, 0]) - np.mean(X[i, -10:-5, 0])
            y[i] = 1 / (1 + np.exp(-trend * 5))
        y = np.clip(y + np.random.randn(n_sequences)*0.1, 0, 1)
        self.lstm_model = Sequential([
            LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        self.lstm_model.save(LSTM_MODEL_PATH)
        logger.info("LSTM Model trained and saved")
    
    def predict_ml(self, features: np.ndarray) -> float:
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
    def __init__(self):
        self.market_data = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.ai_models = AIModels()
        self.balance = INITIAL_BALANCE
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        
        logger.info("=" * 60)
        logger.info("🤖 AI TRADING BOT INITIALIZED")
        logger.info(f"📈 Symbols: {SYMBOLS}")
        logger.info(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
        logger.info(f"⏰ Timeframe: {TIMEFRAME}")
        logger.info("=" * 60)
        
        startup_msg = f"""🚀 AI Trading Bot Started
📊 Using REAL Binance Data (FREE)
💰 Paper Balance: ${INITIAL_BALANCE:,.2f}
📈 Trading: {', '.join(SYMBOLS)}
⏰ Timeframe: {TIMEFRAME}
🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        send_telegram(startup_msg)
    
    def analyze_symbol(self, symbol: str) -> Dict:
        try:
            df = self.market_data.fetch_ohlcv(symbol)
            if len(df) < 100:
                logger.warning(f"Not enough data for {symbol}")
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            df = self.indicators.calculate_all_indicators(df)
            if len(df) < 50:
                return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
            
            latest = df.iloc[-1]
            
            feature_cols = ['rsi', 'macd', 'macd_hist', 'bb_position', 'volume_ratio', 'returns', 'atr', 'stoch_k']
            features = []
            for col in feature_cols:
                features.append(latest[col] if col in df.columns else 0.0)
            features = np.array(features).reshape(1, -1)
            ml_prob = self.ai_models.predict_ml(features)
            
            sequence_cols = ['close', 'volume', 'rsi', 'macd', 'atr']
            sequence_data = []
            for col in sequence_cols:
                if col in df.columns:
                    seq = df[col].values[-30:] if len(df[col]) >=30 else np.pad(df[col].values, (30-len(df[col]),0), 'edge')
                    sequence_data.append(seq)
            if len(sequence_data) >= 3:
                sequence = np.column_stack(sequence_data)
                lstm_prob = self.ai_models.predict_lstm(sequence)
            else:
                lstm_prob = 0.5
            
            combined_prob = (ml_prob * 0.6 + lstm_prob * 0.4)
            # loosened buy conditions
            signal = "HOLD"
            confidence = combined_prob
            rsi_limit = 50  # allow more trades
            
            if (combined_prob > COMBINED_THRESHOLD and
                latest['rsi'] < rsi_limit and
                len(self.positions) < MAX_POSITIONS):
                
                signal = "BUY"
                position_value = self.balance * RISK_PER_TRADE
                entry_price = latest['close']
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
                    "status": "OPEN"
                }
                
                self.positions.append(position)
                self.balance -= position_value
                
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
            
            self.check_positions(symbol, latest['close'])
            
            return {"symbol": symbol, "signal": signal, "confidence": combined_prob}
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "signal": "HOLD", "confidence": 0}
    
    def check_positions(self, symbol: str, current_price: float):
        for pos in self.positions[:]:
            if pos['symbol'] != symbol or pos['status'] != "OPEN":
                continue
            
            if current_price <= pos['stop_loss']:
                pl = (current_price - pos['entry_price']) * pos['size']
                self.balance += pos['value'] + pl
                pos['status'] = "CLOSED"
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now()
                self.trade_history.append(pos)
                logger.info(f"🛑 STOP LOSS HIT {symbol} @ ${current_price:.2f} | P/L: ${pl:.2f}")
                send_telegram(f"🛑 STOP LOSS HIT\nSymbol: {symbol}\nExit: ${current_price:.2f}\nP/L: ${pl:.2f}\nBalance: ${self.balance:,.2f}")
                self.positions.remove(pos)
            
            elif current_price >= pos['take_profit']:
                pl = (current_price - pos['entry_price']) * pos['size']
                self.balance += pos['value'] + pl
                pos['status'] = "CLOSED"
                pos['exit_price'] = current_price
                pos['exit_time'] = datetime.now()
                self.trade_history.append(pos)
                logger.info(f"🏁 TAKE PROFIT HIT {symbol} @ ${current_price:.2f} | P/L: ${pl:.2f}")
                send_telegram(f"🏁 TAKE PROFIT HIT\nSymbol: {symbol}\nExit: ${current_price:.2f}\nP/L: ${pl:.2f}\nBalance: ${self.balance:,.2f}")
                self.positions.remove(pos)
    
    async def run(self):
        while True:
            try:
                for symbol in SYMBOLS:
                    self.analyze_symbol(symbol)
                    await asyncio.sleep(0.5)  # prevent rate limits
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

# ================= MAIN =================
if __name__ == "__main__":
    bot = TradingBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        send_telegram("🛑 Bot stopped manually")
