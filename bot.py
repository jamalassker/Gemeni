import pandas as import pandas as pd
import numpy as np
import requests, asyncio, os, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= MAX AGGRESSION CONFIG =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 5
MIN_PROBABILITY = 0.51  # Extreme low bar for entry
TAKER_FEE = 0.0010

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SuperSniper")

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

class TechnicalIndicators:
    @staticmethod
    def add_indicators(df):
        if len(df) < 30: return df # Lowered requirement
        df['sma_20'] = df['close'].rolling(20).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + gain/loss))
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        return df.dropna()

class SuperAI:
    def __init__(self):
        self.model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))])
        self.is_trained = False

    def train(self, data):
        # Simplified features to ensure training succeeds quickly
        X = data[['rsi', 'volatility', 'returns']]
        y = (data['close'].shift(-1) > data['close']).astype(int) # Predict ANY upward move
        self.model.fit(X, y.fillna(0))
        self.is_trained = True
        logger.info("✅ AI Retrained: Prediction targeted at immediate upward movement.")

    def predict(self, row):
        try:
            X = row[['rsi', 'volatility', 'returns']]
            return self.model.predict_proba(X)[0][1]
        except: return 0.5

class SuperSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.ai = SuperAI()
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    async def report_loop(self):
        while True:
            active = [p for p in self.positions if p['status'] == 'OPEN']
            if active:
                floating = 0.0
                for p in active:
                    try:
                        ticker = self.exchange.fetch_ticker(p['sym'])
                        floating += (ticker['last'] - p['entry']) * p['size']
                    except: continue
                net = self.realized_pnl + floating - self.total_fees
                msg = f"📊 <b>LIVE:</b> Bal ${self.wallet_balance:.1f} | Float ${floating:.2f} | Net ${net:.2f}"
                send_telegram(msg)
            await asyncio.sleep(2)

    async def trading_loop(self):
        # Initial Fast Train
        hist = self.exchange.fetch_ohlcv("BTC/USDT", TIMEFRAME, limit=200)
        df_hist = TechnicalIndicators.add_indicators(pd.DataFrame(hist, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
        self.ai.train(df_hist)

        while True:
            for symbol in SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=40)
                    df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                    df = TechnicalIndicators.add_indicators(df)
                    
                    if df.empty:
                        logger.warning(f"Skipping {symbol}: Not enough data")
                        continue
                    
                    prob = self.ai.predict(df.iloc[-1:])
                    logger.info(f"Scanning {symbol}: Prob {prob:.2f}") # This helps you see it working in Railway logs

                    if prob >= MIN_PROBABILITY and len([x for x in self.positions if x['status']=='OPEN']) < MAX_POSITIONS:
                        price = df['close'].iloc[-1]
                        trade_val = self.wallet_balance * 0.15 # 15% trade size
                        fee = trade_val * TAKER_FEE
                        self.wallet_balance -= (trade_val + fee)
                        self.total_fees += fee
                        self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                        send_telegram(f"🚀 <b>BUY:</b> {symbol} (Prob: {prob:.2f})")
                except Exception as e:
                    logger.error(f"Error {symbol}: {e}")
                await asyncio.sleep(1) # Slow down slightly to avoid Binance ban

    async def run(self):
        await asyncio.gather(self.trading_loop(), self.report_loop())

if __name__ == "__main__":
    asyncio.run(SuperSniper().run())


