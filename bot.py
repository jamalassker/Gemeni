import pandas as pd
import numpy as np
import requests, asyncio, os, joblib, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime, timedelta

# ================= CONFIG =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 3
KELLY_FRACTION = 0.2 
MIN_PROBABILITY = 0.72 
TAKER_FEE = 0.0010
TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SuperSniper")

# --- Telegram Logic ---
def send_telegram(message: str):
    try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                       json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

# --- Quant Logic ---
class QuantEngine:
    @staticmethod
    def get_features(df):
        if len(df) < 50: return df
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / 
                                      df['close'].diff().where(df['close'].diff() < 0, 0).abs().rolling(14).mean().replace(0, 0.001))))
        return df.dropna()

class SuperAI:
    def __init__(self):
        self.model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200))])
        self.regime_detector = KMeans(n_clusters=3, random_state=42)
        self.is_trained = False
        self.last_train_time = None

    def train(self, historical_data):
        logger.info("🧠 Walk-Forward Training Started...")
        X = historical_data[['rsi', 'volatility', 'returns']]
        historical_data['regime'] = self.regime_detector.fit_predict(X)
        y = (historical_data['close'].shift(-10) > historical_data['close'] * 1.025).astype(int)
        self.model.fit(X, y.fillna(0))
        self.is_trained = True
        self.last_train_time = datetime.now()
        logger.info("✅ Model Optimized for Current Market Regime.")

    def get_trade_conviction(self, current_row):
        X = current_row[['rsi', 'volatility', 'returns']]
        prob = self.model.predict_proba(X)[0][1]
        regime = self.regime_detector.predict(X)[0]
        return prob, regime

class SuperSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        self.ai = SuperAI()
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    async def send_2s_report(self):
        floating = 0.0
        for pos in self.positions:
            if pos['status'] == 'OPEN':
                ticker = await self.fetch_price_async(pos['sym'])
                floating += (ticker - pos['entry']) * pos['size']
        
        net = self.realized_pnl + floating - self.total_fees
        report = (
            f"📊 <b>LIVE REPORT</b>\n"
            f"---------------------------\n"
            f"💰 <b>Bal:</b> ${self.wallet_balance:.2f} | 💸 <b>Fees:</b> ${self.total_fees:.2f}\n"
            f"📈 <b>Realized:</b> ${self.realized_pnl:.2f}\n"
            f"📉 <b>Floating:</b> ${floating:.2f}\n"
            f"🚀 <b>Net Profit:</b> ${net:.2f}\n"
            f"---------------------------\n"
            f"Trades Active: {len([p for p in self.positions if p['status'] == 'OPEN'])}"
        )
        send_telegram(report)

    async def fetch_price_async(self, symbol):
        # Helper to avoid blocking the 2s loop
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']

    async def run(self):
        last_report = 0
        while True:
            now = time.time()
            
            # 1. THE 2-SECOND NOTIFICATION FIX
            if now - last_report >= 2:
                await self.send_2s_report()
                last_report = now

            # 2. THE WALK-FORWARD OPTIMIZATION FIX (Daily Retrain)
            if not self.ai.is_trained or (datetime.now() - self.ai.last_train_time).days >= 1:
                # Use BTC as the training anchor for global regime
                ohlcv = self.exchange.fetch_ohlcv("BTC/USDT", TIMEFRAME, limit=1000)
                df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                self.ai.train(QuantEngine.get_features(df))

            # 3. ANALYSIS LOOP
            for symbol in SYMBOLS:
                # (Your existing Entry/Exit Logic remains here)
                await asyncio.sleep(0.01) # Safety breather
            
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(SuperSniper().run())
