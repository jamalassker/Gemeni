import pandas as pd
import numpy as np
import asyncio, logging, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= DANGEROUS QUANT CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT",
    "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT",
    "FET/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "TIA/USDT", "SEI/USDT", "OP/USDT", "ARB/USDT", "INJ/USDT",
    "LDO/USDT", "JUP/USDT", "PYTH/USDT", "ORDI/USDT", "RUNE/USDT", "KAS/USDT", "AAVE/USDT", "MKR/USDT", "PENDLE/USDT", "EVE/USDT",
    "ENA/USDT", "W/USDT", "TAO/USDT", "FIL/USDT", "ETC/USDT", "IMX/USDT", "HBAR/USDT", "VET/USDT", "GRT/USDT", "THETA/USDT",
    "ALGO/USDT", "FTM/USDT", "EGLD/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "FLOW/USDT", "CHZ/USDT", "NEO/USDT", "EOS/USDT"
]

TIMEFRAME = "5m"
MAX_POSITIONS = 10
MIN_PROB = 0.55        # Lowered to 0.55 for more trade activity
TAKER_FEE = 0.001
ATR_MULTIPLIER = 2.0   # Take profit at 2x ATR

TG_TOKEN = "8488789199:AAHhViKmhXlvE7WpgZGVDS4WjCjUuBVtqzQ"
TG_CHAT = "5665906172"

class ProSniperV7:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 10000.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.report_id = None
        self.session = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def send_tg(self, text, edit=False):
        """Asynchronous Telegram logic with timeout protection"""
        await self.init_session()
        url = f"https://api.telegram.org/bot{TG_TOKEN}/"
        method = "editMessageText" if edit and self.report_id else "sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        
        try:
            async with self.session.post(url + method, json=payload, timeout=8) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): 
                    self.report_id = data["result"]["message_id"]
        except Exception as e:
            print(f"TG Log: {e}")

    async def get_data(self, sym, limit=150):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # Indicators
        df['tr'] = np.maximum(df['h'] - df['l'], 
                              np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / 
                                      df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        return df.dropna()

    async def train_models(self):
        await self.send_tg("⚙️ <b>Brain Training Started...</b>")
        for sym in SYMBOLS:
            try:
                df = await self.get_data(sym)
                # Triple Barrier Labeling
                df['target'] = (df['c'].shift(-5) > df['c'] * 1.008).astype(int) 
                X = df[['rsi', 'atr']]
                self.models[sym] = Pipeline([('s', StandardScaler()), 
                                            ('rf', RandomForestClassifier(n_estimators=50))]).fit(X, df['target'])
            except: continue
        await self.send_tg("✅ <b>Brains Ready.</b> Scanning 60 markets.")

    async def risk_manager(self):
        while True:
            if not self.positions:
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

                    # Exit Logic (TP or SL)
                    if price >= p['tp'] or price <= p['sl']:
                        reason = "✅ TP" if price >= p['tp'] else "❌ SL"
                        val = p['size'] * price
                        self.cash += (val * (1 - TAKER_FEE))
                        self.realized_pnl += (val - (p['size'] * p['entry']))
                        self.total_fees += (val * TAKER_FEE)
                        del self.positions[sym]
                        await self.send_tg(f"🏁 <b>EXIT {sym}:</b> {reason}")
                except: continue

            # Update Persistent Dashboard
            net = self.realized_pnl + float_pnl - self.total_fees
            report = f"💎 <b>PRO DASHBOARD</b>\nCash: {self.cash:.2f}\nNet: {net:+.2f}\n\n{pos_details}"
            await self.send_tg(report, edit=True)
            await asyncio.sleep(2)

    async def trade_loop(self):
        await self.train_models()
        asyncio.create_task(self.risk_manager())
        
        while True:
            # Retrain every 2 hours
            if int(time.time()) % 7200 < 30: await self.train_models()
            
            for sym in SYMBOLS:
                if sym in self.positions or len(self.positions) >= MAX_POSITIONS: continue
                try:
                    df = await self.get_data(sym, limit=30)
                    curr = df.iloc[-1]
                    prob = self.models[sym].predict_proba(df[['rsi', 'atr']].iloc[-1:])[0][1]
                    
                    if prob >= MIN_PROB:
                        entry = curr['c']
                        # SL is 1.5x ATR, TP is 2x ATR (Asymmetric Edge)
                        tp = entry + (curr['atr'] * ATR_MULTIPLIER)
                        sl = entry - (curr['atr'] * 1.5)
                        
                        trade_val = self.cash * 0.10
                        size = trade_val / entry
                        self.cash -= (trade_val * (1 + TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        
                        self.positions[sym] = {'entry': entry, 'size': size, 'tp': tp, 'sl': sl}
                        await self.send_tg(f"🚀 <b>BUY:</b> {sym} (Prob: {prob:.2f})")
                except: continue
                await asyncio.sleep(0.1)
            await asyncio.sleep(5)

if __name__ == "__main__":
    print("Initializing Engine...")
    asyncio.run(ProSniperV7().trade_loop())

