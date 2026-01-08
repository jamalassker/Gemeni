import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= PERSISTENT PRO-QUANT CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT",
    "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT",
    "FET/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "TIA/USDT", "SEI/USDT", "OP/USDT", "ARB/USDT", "INJ/USDT",
    "LDO/USDT", "JUP/USDT", "PYTH/USDT", "ORDI/USDT", "RUNE/USDT", "KAS/USDT", "AAVE/USDT", "MKR/USDT", "PENDLE/USDT", "EVE/USDT",
    "ENA/USDT", "W/USDT", "TAO/USDT", "FIL/USDT", "ETC/USDT", "IMX/USDT", "HBAR/USDT", "VET/USDT", "GRT/USDT", "THETA/USDT",
    "ALGO/USDT", "FTM/USDT", "EGLD/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "FLOW/USDT", "CHZ/USDT", "NEO/USDT", "EOS/USDT"
]

TIMEFRAME = "5m"
MAX_POSITIONS = 10     # Cap at 10 simultaneous trades
MIN_PROB = 0.55        # Balanced confidence threshold
TAKER_FEE = 0.001
ATR_MULTIPLIER = 2.0   # Take Profit = 2x ATR

TG_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TG_CHAT = "5665906172"

class PersistentProSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        # STATE VARIABLES - Persistent throughout the session
        self.cash = 10000.0  
        self.realized_pnl = 0.0
        self.total_fees = 0.0
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

    async def get_processed_data(self, sym, limit=150):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['tr'] = np.maximum(df['h'] - df['l'], np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        return df.dropna()

    async def train_background(self):
        """BACKGROUND TASK: Updates brains every 2h without touching balance or trades"""
        while True:
            await self.send_tg("🧠 <b>Periodic Retraining...</b> (Balance is safe)")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_processed_data(sym)
                    # Triple Barrier Label (Targeting 0.8% move)
                    df['target'] = (df['c'].shift(-5) > df['c'] * 1.008).astype(int) 
                    X, y = df[['rsi', 'atr']], df['target']
                    new_models[sym] = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))]).fit(X, y)
                except: continue
            
            self.models.update(new_models) 
            await self.send_tg("✅ <b>Retraining Complete.</b> New logic applied.")
            await asyncio.sleep(7200) # Wait 2 hours

    async def risk_manager(self):
        """BACKGROUND TASK: Monitors exits and updates Dashboard"""
        while True:
            if not self.positions:
                net = self.realized_pnl - self.total_fees
                await self.send_tg(f"💎 <b>PRO DASHBOARD</b>\nCash: {self.cash:.2f}\nNet: {net:+.2f}\n\n<i>Waiting for signals...</i>", edit=True)
                await asyncio.sleep(5)
                continue
                
            float_pnl = 0.0
            pos_details = ""
            for sym, p in list(self.positions.items()):
                try:
                    ticker = await self.exchange.fetch_ticker(sym)
                    price = ticker['last']
                    cur_pnl = (price - p['entry']) * p['size']
                    float_pnl += cur_pnl
                    pos_details += f"▫️ {sym}: {cur_pnl:+.2f}\n"

                    # Exit Logic
                    if price >= p['tp'] or price <= p['sl']:
                        reason = "✅ TP" if price >= p['tp'] else "❌ SL"
                        val = p['size'] * price
                        self.cash += (val * (1 - TAKER_FEE))
                        self.realized_pnl += (val - (p['size'] * p['entry']))
                        self.total_fees += (val * TAKER_FEE)
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {reason}")
                except: continue

            net = self.realized_pnl + float_pnl - self.total_fees
            report = f"💎 <b>PRO DASHBOARD</b>\nCash: {self.cash:.2f}\nNet: {net:+.2f}\n\n{pos_details}"
            await self.send_tg(report, edit=True)
            await asyncio.sleep(2)

    async def trade_loop(self):
        print("🚀 INITIALIZING ENGINE...")
        asyncio.create_task(self.train_background())
        asyncio.create_task(self.risk_manager())
        
        while True:
            if not self.models:
                await asyncio.sleep(10)
                continue
                
            for sym in SYMBOLS:
                if sym in self.positions or len(self.positions) >= MAX_POSITIONS: continue
                try:
                    df = await self.get_processed_data(sym, limit=30)
                    curr = df.iloc[-1]
                    prob = self.models[sym].predict_proba(df[['rsi', 'atr']].iloc[-1:])[0][1]
                    
                    if prob >= MIN_PROB:
                        entry = curr['c']
                        tp = entry + (curr['atr'] * ATR_MULTIPLIER)
                        sl = entry - (curr['atr'] * 1.5)
                        
                        trade_val = self.cash * 0.10 # Stake 10% of cash per trade
                        size = trade_val / entry
                        self.cash -= (trade_val * (1 + TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        
                        self.positions[sym] = {'entry': entry, 'size': size, 'tp': tp, 'sl': sl}
                        await self.send_tg(f"🚀 <b>BUY:</b> {sym} (Prob: {prob:.2f})")
                except: continue
                await asyncio.sleep(0.1)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(PersistentProSniper().trade_loop())



