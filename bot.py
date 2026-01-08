import pandas as pd
import numpy as np
import requests, asyncio, os, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= AGGRESSIVE CONFIG + EXIT RULES =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 5
MIN_PROBABILITY = 0.51 
TAKER_FEE = 0.0010

# --- NEW EXIT SETTINGS ---
TARGET_NET_PROFIT = 20.0  # Sell everything and lock in $20 profit
MAX_NET_LOSS = -50.0      # Emergency exit if total loss hits -$50

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
        if len(df) < 30: return df
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
        X = data[['rsi', 'volatility', 'returns']]
        y = (data['close'].shift(-1) > data['close']).astype(int)
        self.model.fit(X, y.fillna(0))
        self.is_trained = True

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

    async def report_and_exit_check(self):
        """Monitors Profit and triggers Auto-Exit"""
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
                
                # 2-Second Report
                msg = f"📊 <b>LIVE:</b> Bal ${self.wallet_balance:.1f} | Net <b>${net:.2f}</b>"
                send_telegram(msg)

                # --- AUTO-EXIT LOGIC ---
                if net >= TARGET_NET_PROFIT:
                    await self.close_all_positions(active, floating, "TARGET PROFIT REACHED 💰")
                elif net <= MAX_NET_LOSS:
                    await self.close_all_positions(active, floating, "STOP LOSS TRIGGERED ⚠️")
            
            await asyncio.sleep(2)

    async def close_all_positions(self, active_list, current_floating, reason):
        """Sells everything and resets for the next round"""
        exit_fees = (sum(p['size'] * p['entry'] for p in active_list)) * TAKER_FEE
        self.realized_pnl += (current_floating - exit_fees)
        self.wallet_balance += sum(p['size'] * p['entry'] for p in active_list) + current_floating
        self.total_fees += exit_fees
        
        for p in self.positions: p['status'] = 'CLOSED'
        
        send_telegram(f"🏁 <b>{reason}</b>\nFinal Net: ${self.realized_pnl - self.total_fees:.2f}\nAll positions cleared.")
        self.positions = [] # Clear history to stop report spam

    async def trading_loop(self):
        # Initial Fast Train
        hist = self.exchange.fetch_ohlcv("BTC/USDT", TIMEFRAME, limit=200)
        df_hist = TechnicalIndicators.add_indicators(pd.DataFrame(hist, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
        self.ai.train(df_hist)

        while True:
            for symbol in SYMBOLS:
                if len([x for x in self.positions if x['status']=='OPEN']) >= MAX_POSITIONS:
                    break
                
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=40)
                    df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                    df = TechnicalIndicators.add_indicators(df)
                    if df.empty: continue
                    
                    prob = self.ai.predict(df.iloc[-1:])
                    if prob >= MIN_PROBABILITY:
                        price = df['close'].iloc[-1]
                        trade_val = self.wallet_balance * 0.15
                        self.wallet_balance -= (trade_val + (trade_val * TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                        send_telegram(f"🚀 <b>BUY:</b> {symbol}")
                except: continue
                await asyncio.sleep(1)
            await asyncio.sleep(5)

    async def run(self):
        await asyncio.gather(self.trading_loop(), self.report_and_exit_check())

if __name__ == "__main__":
    asyncio.run(SuperSniper().run())
