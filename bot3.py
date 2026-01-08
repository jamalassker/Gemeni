
import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
import ccxt.async_support as ccxt 

# ================= CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",]
    

TIMEFRAME = "1m"
MAX_POSITIONS = 1
MIN_PROB = 0.51        # Loosened further to force activity
TAKER_FEE = 0.001
MIN_WIN_PCT = 0.002    # Targeting 0.2% gains for fast turnover

TG_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
TG_CHAT = "8287625785"

class Bot3Scalper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 10.0      
        self.realized_pnl = 0.0
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
            async with self.session.post(url, json=payload, timeout=10) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]
        except Exception as e: print(f"TG Error: {e}")

    async def get_features(self, sym, limit=100):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
            df['body_size'] = (df['c'] - df['o']).abs() / (df['h'] - df['l']).replace(0, 0.001)
            df['ret_lag'] = df['c'].pct_change()
            return df.dropna()
        except: return pd.DataFrame()

    async def train_background(self):
        while True:
            await self.send_tg("⚙️ <b>Phase 1: Brain Training Started</b>")
            new_models = {}
            for i, sym in enumerate(SYMBOLS):
                df = await self.get_features(sym, limit=200)
                if df.empty: continue
                
                df['target'] = (df['c'].shift(-3) > df['c'] * (1 + MIN_WIN_PCT)).astype(int)
                X = df[['rsi', 'body_size', 'ret_lag']]
                y = df['target']
                
                if len(y.unique()) > 1: # Ensure data has both wins and losses
                    model = GradientBoostingClassifier(n_estimators=20)
                    model.fit(X, y)
                    new_models[sym] = {'model': model, 'scaler': RobustScaler().fit(X)}
                
                if i % 5 == 0: # Update every 5 symbols
                    print(f"Trained {sym}...")
            
            self.models = new_models
            await self.send_tg(f"✅ <b>Phase 2: Scanning {len(self.models)} Markets</b>")
            await asyncio.sleep(3600)

    async def run_scalper(self):
        await self.send_tg("🚀 <b>BOT3.1 ONLINE</b>\nInitializing Engine...")
        asyncio.create_task(self.train_background())
        
        while True:
            if not self.models:
                await asyncio.sleep(5) # Still training
                continue

            # 1. RISK & DASHBOARD
            float_pnl = 0.0
            if self.positions:
                for sym, p in list(self.positions.items()):
                    ticker = await self.exchange.fetch_ticker(sym)
                    curr_p = ticker['last']
                    float_pnl = (curr_p - p['entry']) * p['size']
                    
                    elapsed = (time.time() - p['time']) / 60
                    if curr_p >= p['tp'] or curr_p <= p['sl'] or elapsed > 5:
                        reason = "🎯 TP" if curr_p >= p['tp'] else "🛡️ SL" if curr_p <= p['sl'] else "⏰ Time"
                        val = p['size'] * curr_p
                        self.cash += (val * (1 - TAKER_FEE))
                        self.realized_pnl += (val - (p['size'] * p['entry']) - (val * TAKER_FEE))
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym} ({reason}):</b> Net {self.realized_pnl:+.4f}")
            
            # 2. ENTRY SEARCH
            elif not self.positions:
                best_sym, best_prob = None, 0
                for sym in list(self.models.keys()):
                    df = await self.get_features(sym, limit=20)
                    if df.empty: continue
                    feat = df[['rsi', 'body_size', 'ret_lag']].iloc[-1:]
                    prob = self.models[sym]['model'].predict_proba(feat)[0][1]
                    
                    if prob > best_prob:
                        best_prob, best_sym = prob, sym
                
                if best_sym and best_prob > MIN_PROB:
                    price = (await self.exchange.fetch_ticker(best_sym))['last']
                    size = (self.cash * (1 - TAKER_FEE)) / price
                    self.positions[best_sym] = {
                        'entry': price, 'size': size, 'time': time.time(),
                        'tp': price * 1.004, 'sl': price * 0.996
                    }
                    self.cash = 0 # All in
                    await self.send_tg(f"⚡ <b>SCALP BUY:</b> {best_sym}\nProb: {best_prob:.2f}")

            # Live Dashboard Update
            net_val = (self.cash if not self.positions else 0) + (self.positions[list(self.positions.keys())[0]]['size'] * (await self.exchange.fetch_ticker(list(self.positions.keys())[0]))['last'] if self.positions else 0)
            status = f"💰 <b>LIVE DASHBOARD</b>\nCash: ${self.cash + (float_pnl if self.positions else 0):.2f}\nNet PnL: {self.realized_pnl:+.4f}\nStatus: {'TRADING' if self.positions else 'SCANNING'}"
            await self.send_tg(status, edit=True)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(Bot3Scalper().run_scalper())
