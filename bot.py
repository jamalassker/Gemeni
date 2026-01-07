import pandas as pd
import numpy as np
import requests, asyncio, os, joblib, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= CONFIG (PRESERVED) =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "1000SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 3
KELLY_FRACTION = 0.2 
MIN_PROBABILITY = 0.72 
TAKER_FEE, STOP_LOSS_PCT, TAKE_PROFIT_PCT = 0.0010, 0.08, 0.12

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SuperSniper")

# --- Helper Functions ---
def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

# --- Indicators & AI Classes (Logic Preserved) ---
class TechnicalIndicators:
    @staticmethod
    def add_indicators(df):
        if len(df) < 50: return df
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
        df['rsi'] = 100 - (100 / (1 + gain/loss))
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['bb_p'] = (df['close'] - (df['sma_20'] - 2*df['close'].rolling(20).std())) / (4*df['close'].rolling(20).std() + 0.001)
        df['stoch'] = 100 * (df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 0.001)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        return df.dropna()

class SuperAI:
    def __init__(self):
        self.model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200))])
        self.regime_detector = KMeans(n_clusters=3, random_state=42)
        self.is_trained = False
        self.last_train_day = None

    def train(self, data):
        X = data[['rsi', 'volatility', 'returns', 'macd', 'stoch']]
        data['regime'] = self.regime_detector.fit_predict(X)
        y = (data['close'].shift(-10) > data['close'] * 1.025).astype(int)
        self.model.fit(X, y.fillna(0))
        self.is_trained = True
        self.last_train_day = datetime.now().day
        logger.info("🔥 AI Recalibrated.")

    def predict(self, row):
        X = row[['rsi', 'volatility', 'returns', 'macd', 'stoch']]
        prob = self.model.predict_proba(X)[0][1]
        regime = self.regime_detector.predict(X)[0]
        return prob, regime

class SuperSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        self.ai = SuperAI()
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    async def report_loop(self):
        """2-Second High Frequency Notification - ONLY ACTIVE DURING TRADES"""
        while True:
            # Check if we have any active positions
            active_trades = [p for p in self.positions if p['status'] == 'OPEN']
            
            if active_trades:
                floating = 0.0
                for p in active_trades:
                    curr = self.exchange.fetch_ticker(p['sym'])['last']
                    floating += (curr - p['entry']) * p['size']
                
                net = self.realized_pnl + floating - self.total_fees
                msg = (f"📊 <b>ACTIVE TRADE REPORT</b>\n--------------------\n"
                       f"💰 <b>Wallet:</b> ${self.wallet_balance:.2f}\n"
                       f"📈 <b>Realized:</b> ${self.realized_pnl:.2f}\n"
                       f"📉 <b>Floating:</b> ${floating:.2f}\n"
                       f"💸 <b>Fees:</b> ${self.total_fees:.2f}\n"
                       f"🚀 <b>Net PnL:</b> ${net:.2f}\n--------------------\n"
                       f"Open Positions: {len(active_trades)}")
                send_telegram(msg)
            
            await asyncio.sleep(2)

    async def trading_loop(self):
        while True:
            if not self.ai.is_trained or datetime.now().day != self.ai.last_train_day:
                hist = self.exchange.fetch_ohlcv("BTC/USDT", TIMEFRAME, limit=500)
                df_hist = TechnicalIndicators.add_indicators(pd.DataFrame(hist, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
                self.ai.train(df_hist)

            for symbol in SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                    df = TechnicalIndicators.add_indicators(df)
                    row = df.iloc[-1:]
                    
                    prob, regime = self.ai.predict(row)
                    
                    if prob > MIN_PROBABILITY and regime != 0 and len([x for x in self.positions if x['status']=='OPEN']) < MAX_POSITIONS:
                        price = df['close'].iloc[-1]
                        k_perc = (prob * 2.0 - (1 - prob)) / 2.0
                        k_size = max(0, k_perc * KELLY_FRACTION)
                        
                        if k_size > 0:
                            trade_val = self.wallet_balance * k_size
                            fee = trade_val * TAKER_FEE
                            self.wallet_balance -= (trade_val + fee)
                            self.total_fees += fee
                            self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                            send_telegram(f"⚡ <b>TRADE OPENED:</b> {symbol}\nSize: {k_size:.1%}")
                except: continue
            await asyncio.sleep(10)

    async def run(self):
        await asyncio.gather(self.trading_loop(), self.report_loop())

if __name__ == "__main__":
    bot = SuperSniper()
    asyncio.run(bot.run())
