import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= PERSISTENT CONFIG =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT", "SUI/USDT"] 
TIMEFRAME = "5m"
MAX_POSITIONS = 1
MIN_PROB = 0.55
TAKER_FEE = 0.001
ATR_MULTIPLIER = 2.0

TG_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TG_CHAT = "5665906172"

class PersistentSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        # THESE VARIABLES NEVER RESET
        self.cash = 10000.0  
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.positions = {}
        self.report_id = None
        self.session = None

    async def init_session(self):
        if not self.session: self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/" + ("editMessageText" if edit and self.report_id else "sendMessage")
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        try:
            async with self.session.post(url, json=payload, timeout=8) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except: pass

    async def train_background(self):
        """BACKGROUND TASK: Updates models without touching the balance"""
        while True:
            await self.send_tg("🧠 <b>Periodic Retraining...</b> (Balance is safe)")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=150)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
                    df['atr'] = df['tr'].rolling(14).mean()
                    df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
                    df = df.dropna()
                    
                    # Triple Barrier Label
                    df['target'] = (df['c'].shift(-5) > df['c'] * 1.008).astype(int) 
                    X, y = df[['rsi', 'atr']], df['target']
                    new_models[sym] = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))]).fit(X, y)
                except: continue
            
            self.models.update(new_models) # Only updates the brains
            await self.send_tg("✅ <b>Retraining Complete.</b> New logic applied.")
            await asyncio.sleep(7200) # Wait 2 hours

    async def run(self):
        print("🚀 INITIALIZING...")
        asyncio.create_task(self.train_background()) # Starts the persistent brain cycle
        
        while True:
            if not self.models: # Wait for first training
                await asyncio.sleep(10)
                continue
                
            for sym in SYMBOLS:
                if sym in self.positions or len(self.positions) >= MAX_POSITIONS: continue
                try:
                    # Logic to fetch current price and check self.models[sym].predict_proba
                    # If trade opens, self.cash is reduced here.
                    pass 
                except: continue
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(PersistentSniper().run())


