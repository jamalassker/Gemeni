import pandas as pd
import numpy as np
import requests, asyncio, json, os, joblib, warnings, time, sys, logging, aiohttp
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt

warnings.filterwarnings('ignore')

# ================= CONFIG (SAME VARS) =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", "POL/USDT", "LTC/USDT", "BCH/USDT", "1000SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]
TIMEFRAME, CANDLES_TO_FETCH = "5m", 100
TAKER_FEE, INITIAL_BALANCE = 0.0010, 100
RISK_PER_TRADE, MAX_POSITIONS = 0.03, 3
STOP_LOSS_PCT, TAKE_PROFIT_PCT = 0.08, 0.12
MIN_TRADE_USDT, MIN_TRADE_CRYPTO = 10, 0.001
COMBINED_THRESHOLD = 0.55 # Increased for quality
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_model.pkl")
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "5665906172")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_telegram(message: str):
    if not TG_TOKEN or "YOUR" in TG_TOKEN: return
    try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

class MarketDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    def fetch_ohlcv(self, symbol, limit=100):
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df.set_index('ts')
        except: return pd.DataFrame()
    def fetch_current_price(self, symbol):
        try: return self.exchange.fetch_ticker(symbol)['last']
        except: return None

class TechnicalIndicators:
    @staticmethod
    def add_indicators(df):
        if len(df) < 50: return df
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / df['close'].diff().where(df['close'].diff() < 0, 0).abs().rolling(14).mean().replace(0, 0.001))))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bb_p'] = (df['close'] - (df['close'].rolling(20).mean() - 2*df['close'].rolling(20).std())) / (4*df['close'].rolling(20).std() + 0.001)
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['vol_r'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, 1)
        df['ret'] = df['close'].pct_change()
        df['stoch'] = 100 * (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 0.001)
        df['pc5'] = df['close'].pct_change(5)
        return df.ffill().bfill()

class AIModels:
    def __init__(self, fetcher):
        self.fetcher = fetcher
        self.model = None
        self.feature_cols = ['rsi', 'macd', 'bb_p', 'vol_r', 'ret', 'atr', 'stoch', 'pc5', 'sma_20', 'ema_12']
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(ML_MODEL_PATH):
            self.model = joblib.load(ML_MODEL_PATH)
            logger.info("✓ Model loaded from disk")
        else:
            self.train_on_real_data()

    def train_on_real_data(self):
        logger.info("🚀 Training AI on real historical data...")
        all_data = []
        for sym in SYMBOLS[:5]: # Use top 5 for fast training
            df = self.fetcher.fetch_ohlcv(sym, limit=500)
            if df.empty: continue
            df = TechnicalIndicators.add_indicators(df)
            # Label: 1 if price goes up 1.5% in next 10 candles
            df['target'] = (df['close'].shift(-10) > df['close'] * 1.015).astype(int)
            all_data.append(df.dropna())
        
        full_df = pd.concat(all_data)
        X = full_df[self.feature_cols]
        y = full_df['target']
        
        self.model = Pipeline([('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(n_estimators=100))])
        self.model.fit(X, y)
        joblib.dump(self.model, ML_MODEL_PATH)
        logger.info(f"✓ AI Trained. Samples: {len(full_df)}")

    def predict(self, df):
        try:
            feats = df[self.feature_cols].iloc[-1:].values
            return self.model.predict_proba(feats)[0][1]
        except: return 0.5

class TradingBot:
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.ai = AIModels(self.fetcher)
        self.balance, self.positions, self.trade_history = INITIAL_BALANCE, [], []
        self.total_fees = 0.0
        send_telegram(f"🤖 Live Bot Started | Bal: ${self.balance}")

    def analyze_symbol(self, symbol):
        if len([p for p in self.positions if p['status'] == 'OPEN']) >= MAX_POSITIONS: return
        df = self.fetcher.fetch_ohlcv(symbol)
        if df.empty: return
        df = TechnicalIndicators.add_indicators(df)
        prob = self.ai.predict(df)

        if prob > COMBINED_THRESHOLD:
            price = df['close'].iloc[-1]
            size = (self.balance * RISK_PER_TRADE) / price
            fee = (price * size) * TAKER_FEE
            total_cost = (price * size) + fee
            
            if total_cost < self.balance:
                pos = {'symbol': symbol, 'entry': price, 'size': size, 'fee': fee, 'sl': price*(1-STOP_LOSS_PCT), 'tp': price*(1+TAKE_PROFIT_PCT), 'status': 'OPEN', 'time': datetime.now()}
                self.balance -= total_cost
                self.total_fees += fee
                self.positions.append(pos)
                send_telegram(f"📈 BUY {symbol} @ {price:.2f}\nProb: {prob:.1%}")

    def check_exits(self):
        for p in [p for p in self.positions if p['status'] == 'OPEN']:
            curr = self.fetcher.fetch_current_price(p['symbol'])
            if not curr: continue
            reason = None
            if curr <= p['sl']: reason = "SL"
            elif curr >= p['tp']: reason = "TP"
            
            if reason:
                exit_val = curr * p['size']
                exit_fee = exit_val * TAKER_FEE
                net = exit_val - exit_fee
                self.balance += net
                self.total_fees += exit_fee
                p['status'] = 'CLOSED'
                p['pnl'] = net - (p['entry'] * p['size'] + p['fee'])
                send_telegram(f"📉 CLOSED {p['symbol']} via {reason}\nPNL: ${p['pnl']:.2f}")

    async def run(self):
        while True:
            for sym in SYMBOLS:
                self.analyze_symbol(sym)
                self.check_exits()
                await asyncio.sleep(1)
            logger.info(f"Bal: ${self.balance:.2f} | Fees: ${self.total_fees:.2f}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.run())
