import pandas as pd
import numpy as np
import requests, asyncio, os, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= ELITE RAPID-FIRE CONFIG =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT",
    "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT",
    "FET/USDT", "PEPE/USDT", "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "TIA/USDT", "SEI/USDT", "OP/USDT", "ARB/USDT", "INJ/USDT",
    "LDO/USDT", "JUP/USDT", "PYTH/USDT", "ORDI/USDT", "RUNE/USDT", "KAS/USDT", "AAVE/USDT", "MKR/USDT", "PENDLE/USDT", "EVE/USDT",
    "ENA/USDT", "W/USDT", "TAO/USDT", "FIL/USDT", "ETC/USDT", "IMX/USDT", "HBAR/USDT", "VET/USDT", "GRT/USDT", "THETA/USDT",
    "ALGO/USDT", "FTM/USDT", "EGLD/USDT", "SAND/USDT", "MANA/USDT", "AXS/USDT", "FLOW/USDT", "CHZ/USDT", "NEO/USDT", "EOS/USDT"
]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 8
MIN_PROBABILITY = 0.51      # Ultra-Aggressive
TAKER_FEE = 0.0010

# --- ELITE CONSTANTS (Loosened for Speed) ---
TRAIL_START_PROFIT = 15.0   # Lowered target to start trailing sooner
TRAIL_DROP_PCT = 0.20       # Give it more room to breathe
Z_SCORE_LIMIT = 3.5         # Allow buying even on big pumps

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RapidElite")

def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

class TechnicalIndicators:
    @staticmethod
    def add_indicators(df):
        if len(df) < 20: return df
        df['sma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std().replace(0, 0.001)
        df['z_score'] = (df['close'] - df['sma_20']) / df['std_20']
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(10).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(10).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + gain/loss))
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        return df.dropna()

class SuperSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50))])
        self.is_trained = False
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0
        self.peak_net_profit = 0.0

    def get_obi(self, symbol):
        """Elite Move #1: Reduced filter to open more trades"""
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=5)
            bid_vol = sum([q for p, q in ob['bids']])
            ask_vol = sum([q for p, q in ob['asks']])
            return (bid_vol - ask_vol) / (bid_vol + ask_vol + 0.001)
        except: return 0.1 # Default to positive if error

    async def report_and_trail_exit(self):
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
                self.peak_net_profit = max(self.peak_net_profit, net)
                send_telegram(f"💎 <b>RAPID ELITE:</b> Net ${net:.2f} | Peak ${self.peak_net_profit:.2f}")

                if self.peak_net_profit >= TRAIL_START_PROFIT:
                    if net <= (self.peak_net_profit * (1 - TRAIL_DROP_PCT)):
                        await self.close_all(active, floating, "TRAIL EXIT 🎯")
            await asyncio.sleep(2)

    async def close_all(self, active_list, current_floating, reason):
        exit_fees = (sum(p['size'] * p['entry'] for p in active_list)) * TAKER_FEE
        self.realized_pnl += (current_floating - exit_fees)
        self.total_fees += exit_fees
        for p in self.positions: p['status'] = 'CLOSED'
        send_telegram(f"🏁 <b>{reason}</b> Net: ${self.realized_pnl - self.total_fees:.2f}")
        self.positions = []
        self.peak_net_profit = 0.0

    async def trading_loop(self):
        # Initial Fast Train on small data for speed
        hist = self.exchange.fetch_ohlcv("BTC/USDT", "5m", limit=100)
        df_hist = TechnicalIndicators.add_indicators(pd.DataFrame(hist, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
        self.model.fit(df_hist[['rsi', 'volatility', 'returns']], (df_hist['close'].shift(-1) > df_hist['close']).astype(int).fillna(0))

        while True:
            for symbol in SYMBOLS:
                if len([x for x in self.positions if x['status']=='OPEN']) >= MAX_POSITIONS: break
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=30)
                    df = TechnicalIndicators.add_indicators(pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
                    row = df.iloc[-1:]
                    
                    if row['z_score'].values[0] > Z_SCORE_LIMIT: continue 
                    prob = self.model.predict_proba(row[['rsi', 'volatility', 'returns']])[0][1]
                    obi = self.get_obi(symbol)

                    # TRADING TRIGGER
                    if prob >= MIN_PROBABILITY and obi > -0.1: 
                        price = row['close'].values[0]
                        trade_val = self.wallet_balance * 0.10
                        self.wallet_balance -= (trade_val * (1+TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                        send_telegram(f"⚡ <b>RAPID BUY:</b> {symbol} (P: {prob:.2f})")
                except: continue
                await asyncio.sleep(0.1) # Aggressive scan speed
            await asyncio.sleep(2)

    async def run(self):
        await asyncio.gather(self.trading_loop(), self.report_and_trail_exit())

if __name__ == "__main__":
    asyncio.run(SuperSniper().run())

