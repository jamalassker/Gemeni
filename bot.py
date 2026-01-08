import pandas as pd
import numpy as np
import requests, asyncio, os, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= ELITE CONFIG =================
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
MAX_POSITIONS = 10
MIN_PROBABILITY = 0.51
TAKER_FEE = 0.0010
TRAIL_START_PROFIT = 15.0
TRAIL_DROP_PCT = 0.15

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("PersistentElite")

# --- Persistent Telegram Helper ---
class TelegramManager:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.report_msg_id = None

    def send_or_edit_report(self, text):
        url_send = f"https://api.telegram.org/bot{self.token}/sendMessage"
        url_edit = f"https://api.telegram.org/bot{self.token}/editMessageText"
        
        try:
            if self.report_msg_id is None:
                r = requests.post(url_send, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}).json()
                if r.get("ok"): self.report_msg_id = r["result"]["message_id"]
            else:
                requests.post(url_edit, json={"chat_id": self.chat_id, "message_id": self.report_msg_id, "text": text, "parse_mode": "HTML"}, timeout=2)
        except Exception as e: 
            logger.error(f"TG Error: {e}")
            self.report_msg_id = None # Reset on error to try a fresh message

    def notify(self, text):
        """Critical Alerts (New Notifications)"""
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"})

tg = TelegramManager(TG_TOKEN, TG_CHAT)

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
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0
        self.peak_net_profit = 0.0

    async def report_and_trail(self):
        while True:
            active = [p for p in self.positions if p['status'] == 'OPEN']
            if active:
                floating = 0.0
                details = ""
                for p in active:
                    try:
                        ticker = self.exchange.fetch_ticker(p['sym'])
                        p_pnl = (ticker['last'] - p['entry']) * p['size']
                        floating += p_pnl
                        # Numbers only display for individual symbols
                        details += f"▫️ {p['sym']}: {p_pnl:+.2f}\n"
                    except: continue
                
                net = self.realized_pnl + floating - self.total_fees
                self.peak_net_profit = max(self.peak_net_profit, net)
                
                dashboard = (f"💎 <b>ELITE DASHBOARD</b>\n"
                             f"💰 Bal: {self.wallet_balance:.2f}\n"
                             f"📈 Net: {net:.2f}\n"
                             f"🚀 Peak: {self.peak_net_profit:.2f}\n"
                             f"-------------------\n"
                             f"{details}")
                tg.send_or_edit_report(dashboard)

                if self.peak_net_profit >= TRAIL_START_PROFIT:
                    if net <= (self.peak_net_profit * (1 - TRAIL_DROP_PCT)):
                        await self.close_all(active, floating, "TRAILING PROFIT LOCK 🏁")
            else:
                # Idle Dashboard
                tg.send_or_edit_report("💎 <b>ELITE DASHBOARD</b>\nStatus: Scanning 60 symbols...")
            
            await asyncio.sleep(2)

    async def close_all(self, active_list, current_floating, reason):
        exit_fees = (sum(p['size'] * p['entry'] for p in active_list)) * TAKER_FEE
        self.realized_pnl += (current_floating - exit_fees)
        self.wallet_balance += (sum(p['size'] * p['entry'] for p in active_list) + current_floating)
        self.total_fees += exit_fees
        for p in self.positions: p['status'] = 'CLOSED'
        tg.notify(f"🏁 <b>{reason}</b> Net: ${self.realized_pnl - self.total_fees:.2f}")
        self.positions = []
        self.peak_net_profit = 0.0
        tg.report_msg_id = None # Clear old report to start fresh next time

    async def trading_loop(self):
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
                    prob = self.model.predict_proba(row[['rsi', 'volatility', 'returns']])[0][1]

                    if prob >= MIN_PROBABILITY:
                        price = row['close'].values[0]
                        trade_val = self.wallet_balance * 0.10 # Auto-Compounds on growth
                        self.wallet_balance -= (trade_val * (1+TAKER_FEE))
                        self.total_fees += (trade_val * TAKER_FEE)
                        self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                        tg.notify(f"🚀 <b>BUY:</b> {symbol}")
                except: continue
                await asyncio.sleep(0.1)
            await asyncio.sleep(2)

    async def run(self):
        await asyncio.gather(self.trading_loop(), self.report_and_trail())

if __name__ == "__main__":
    asyncio.run(SuperSniper().run())


