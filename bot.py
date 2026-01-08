import pandas as pd
import numpy as np
import asyncio, logging, time, aiohttp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt.async_support as ccxt 

# ================= PRO-QUANT CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT",
    "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT",
    "FET/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "TIA/USDT", "SEI/USDT", "OP/USDT", "ARB/USDT", "INJ/USDT",
    "LDO/USDT", "JUP/USDT", "PYTH/USDT", "ORDI/USDT", "RUNE/USDT", "KAS/USDT", "AAVE/USDT", "MKR/USDT", "PENDLE/USDT", "EVE/USDT",
    "ENA/USDT", "W/USDT", "TAO/USDT", "FIL/USDT", "ETC/USDT", "IMX/USDT", "HBAR/USDT", "VET/USDT", "GRT/USDT", "THETA/USDT",
    "ALGO/USDT", "FTM/USDT", "EGLD/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "FLOW/USDT", "CHZ/USDT", "NEO/USDT", "EOS/USDT"
]

TIMEFRAME = "5m"
MAX_POSITIONS = 1
MIN_PROB = 0.58       # Higher confidence threshold
STOP_LOSS_PCT = 0.015 # Hard -1.5% per trade
TAKER_FEE = 0.001

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

class ProSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.wallet = 10000.0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.peak_net = 0.0
        self.report_id = None

    async def send_tg(self, text, edit=False):
        url = f"https://api.telegram.org/bot{TG_TOKEN}/"
        method = "editMessageText" if edit and self.report_id else "sendMessage"
        payload = {"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}
        if edit: payload["message_id"] = self.report_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url + method, json=payload) as resp:
                data = await resp.json()
                if not edit and data.get("ok"): self.report_id = data["result"]["message_id"]

    async def get_processed_data(self, sym, limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / 
                                      df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        df['vol'] = df['c'].pct_change().rolling(10).std()
        df['ret'] = df['c'].pct_change()
        return df.dropna()

    async def train_models(self):
        for sym in SYMBOLS:
            try:
                df = await self.get_processed_data(sym)
                X = df[['rsi', 'vol', 'ret']]
                y = (df['c'].shift(-1) > df['c']).astype(int)
                self.models[sym] = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))]).fit(X, y)
            except: continue
        await self.send_tg("⚙️ <b>Models Retrained</b>: Per-asset intelligence active.")

    async def run(self):
        await self.train_models()
        asyncio.create_task(self.dashboard_loop())
        
        while True:
            # Retrain every 2 hours
            if int(time.time()) % 7200 < 10: await self.train_models()
            
            for sym in SYMBOLS:
                if len(self.positions) >= MAX_POSITIONS or sym in self.positions: continue
                try:
                    df = await self.get_processed_data(sym, limit=20)
                    prob = self.models[sym].predict_proba(df[['rsi', 'vol', 'ret']].iloc[-1:])[0][1]
                    
                    if prob >= MIN_PROB:
                        price = df['c'].iloc[-1]
                        size = (self.wallet * 0.1) / price
                        self.wallet -= (size * price * (1 + TAKER_FEE))
                        self.total_fees += (size * price * TAKER_FEE)
                        self.positions[sym] = {'entry': price, 'size': size, 'stop': price * (1 - STOP_LOSS_PCT)}
                        await self.send_tg(f"🚀 <b>PRO BUY:</b> {sym} @ {price:.4f}")
                except: continue
                await asyncio.sleep(0.05)
            await asyncio.sleep(1)

    async def dashboard_loop(self):
        while True:
            if self.positions:
                float_pnl = 0.0
                pos_list = ""
                for sym, p in list(self.positions.items()):
                    ticker = await self.exchange.fetch_ticker(sym)
                    cur_pnl = (ticker['last'] - p['entry']) * p['size']
                    float_pnl += cur_pnl
                    pos_list += f"▫️ {sym}: {cur_pnl:+.2f}\n"
                    
                    # Individual Stop Loss Check
                    if ticker['last'] <= p['stop']:
                        exit_val = p['size'] * ticker['last']
                        self.wallet += (exit_val * (1 - TAKER_FEE))
                        self.total_fees += (exit_val * TAKER_FEE)
                        self.realized_pnl += (cur_pnl - (exit_val * TAKER_FEE))
                        del self.positions[sym]
                        await self.send_tg(f"🛡️ <b>STOP HIT:</b> {sym}")

                net = self.realized_pnl + float_pnl - self.total_fees
                report = f"💎 <b>PRO DASHBOARD</b>\nBal: {self.wallet:.2f}\nNet: {net:.2f}\n\n{pos_list}"
                await self.send_tg(report, edit=True)
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(ProSniper().run())



