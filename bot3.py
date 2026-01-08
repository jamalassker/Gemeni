import pandas as pd
import numpy as np
import asyncio, time, aiohttp
from sklearn.ensemble import GradientBoostingClassifier # Switched to GBM for better small-move detection
from sklearn.preprocessing import RobustScaler # Better for handling crypto volatility
import ccxt.async_support as ccxt 

# ================= BOT3: MICRO-SCALPER CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", 
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT",
    "SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "PEPE/USDT",
    "FET/USDT", "RENDER/USDT", "WIF/USDT", "TIA/USDT", "ARB/USDT"
]

TIMEFRAME = "1m"       # Ultra-fast scalping
MAX_POSITIONS = 1      # Focus 100% of $10 on one trade
MIN_PROB = 0.52        # Looser entry for higher activity
TAKER_FEE = 0.001      # 0.1% Binance fee
MIN_WIN_PCT = 0.003    # Aiming for 0.3% per scalp (Fast growth)

TG_TOKEN = "8287625785:AAH5CzpIgBiDYWO3WGikKYSaTwgz0rgc2y0"
TG_CHAT = "8287625785"

class Bot3Scalper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.models = {}
        self.positions = {}
        self.cash = 10.0      # Starting with $10
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

    async def get_features(self, sym, limit=100):
        ohlcv = await self.exchange.fetch_ohlcv(sym, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        
        # ML Features: Momentum + Volatility + Price Action
        df['rsi'] = 100 - (100 / (1 + (df['c'].diff().where(df['c'].diff() > 0, 0).rolling(14).mean() / df['c'].diff().where(df['c'].diff() < 0, 0).abs().rolling(14).mean().replace(0,0.001))))
        df['body_size'] = (df['c'] - df['o']).abs() / (df['h'] - df['l'])
        df['ret_lag'] = df['c'].pct_change()
        df['vol_change'] = df['v'].pct_change()
        
        return df.dropna()

    async def train_background(self):
        while True:
            await self.send_tg("🤖 <b>Bot3: Retraining Scalp Logic...</b>")
            new_models = {}
            for sym in SYMBOLS:
                try:
                    df = await self.get_features(sym, limit=300)
                    # Target: 0.3% gain in the next 3 candles
                    df['target'] = (df['c'].shift(-3) > df['c'] * (1 + MIN_WIN_PCT)).astype(int)
                    X = df[['rsi', 'body_size', 'ret_lag', 'vol_change']]
                    y = df['target']
                    
                    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1)
                    model.fit(X, y)
                    new_models[sym] = {'model': model, 'scaler': RobustScaler().fit(X)}
                except: continue
            
            self.models = new_models
            await self.send_tg("🚀 <b>Scalp Ready.</b> Searching for high-prob setups...")
            await asyncio.sleep(3600) # Retrain every hour for 1m timeframe

    async def run_scalper(self):
        asyncio.create_task(self.train_background())
        
        while True:
            if not self.models:
                await asyncio.sleep(5)
                continue

            # 1. RISK & DASHBOARD MANAGEMENT
            if self.positions:
                for sym, p in list(self.positions.items()):
                    ticker = await self.exchange.fetch_ticker(sym)
                    curr_p = ticker['last']
                    float_pnl = (curr_p - p['entry']) * p['size']
                    
                    # Exit Conditions: TP (0.5%), SL (-0.3%), or Time (3 mins)
                    elapsed = (time.time() - p['time']) / 60
                    if curr_p >= p['tp'] or curr_p <= p['sl'] or elapsed > 3:
                        val = p['size'] * curr_p
                        fee = val * TAKER_FEE
                        self.cash += (val - fee)
                        self.realized_pnl += (val - (p['size'] * p['entry']) - fee)
                        del self.positions[sym]
                        await self.send_tg(f"💰 <b>Scalp Closed:</b> {sym} | Net: {self.realized_pnl:+.4f}")
                
                # Update Dashboard
                net_val = self.cash + (float_pnl if self.positions else 0)
                dashboard = f"📈 <b>BOT3 LIVE</b>\nBalance: ${net_val:.4f}\nTotal PnL: {self.realized_pnl:+.4f}\nActive: {list(self.positions.keys())[0] if self.positions else 'Scanning...'}"
                await self.send_tg(dashboard, edit=True)
            
            # 2. ENTRY SEARCH (Only if no position open)
            elif len(self.positions) < MAX_POSITIONS:
                best_sym, best_prob = None, 0
                for sym in SYMBOLS:
                    try:
                        df = await self.get_features(sym, limit=20)
                        feat = df[['rsi', 'body_size', 'ret_lag', 'vol_change']].iloc[-1:]
                        prob = self.models[sym]['model'].predict_proba(feat)[0][1]
                        
                        if prob > best_prob:
                            best_prob = prob
                            best_sym = sym
                    except: continue
                
                if best_sym and best_prob > MIN_PROB:
                    price = (await self.exchange.fetch_ticker(best_sym))['last']
                    entry_cost = self.cash
                    # Order execution
                    fee = entry_cost * TAKER_FEE
                    self.cash -= entry_cost
                    size = (entry_cost - fee) / price
                    
                    self.positions[best_sym] = {
                        'entry': price, 'size': size, 'time': time.time(),
                        'tp': price * 1.005, 'sl': price * 0.997
                    }
                    await self.send_tg(f"🎯 <b>Scalp Entry:</b> {best_sym} (Prob: {best_prob:.2f})")

            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(Bot3Scalper().run_scalper())
