import pandas as pd
import numpy as np
import requests, asyncio, os, joblib, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= LOOSENED CONFIG =================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "1000SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 5          # Increased to allow more trades
KELLY_FRACTION = 0.4       # Risk more per trade
MIN_PROBABILITY = 0.52     # LOOSENED: Triggers on moderate signals
TAKER_FEE = 0.0010         # Binance Spot Fee (0.1%)

TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SuperSniper")

# --- Telegram Helper ---
def send_telegram(message: str):
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

# --- Indicator Engine ---
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
        self.is_trained = False
        self.last_train_day = None

    def train(self, data):
        # Using a broader feature set for prediction
        X = data[['rsi', 'volatility', 'returns', 'macd', 'stoch']]
        y = (data['close'].shift(-5) > data['close'] * 1.005).astype(int) # Predict 0.5% gain in 5 candles
        self.model.fit(X, y.fillna(0))
        self.is_trained = True
        self.last_train_day = datetime.now().day
        logger.info("✅ AI Training Complete (Aggressive Settings)")

    def predict(self, row):
        X = row[['rsi', 'volatility', 'returns', 'macd', 'stoch']]
        prob = self.model.predict_proba(X)[0][1]
        return prob

class SuperSniper:
    def __init__(self):
        self.exchange = ccxt.binance({'options': {'defaultType': 'spot'}})
        self.ai = SuperAI()
        self.wallet_balance = INITIAL_BALANCE
        self.positions = []
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    async def report_loop(self):
        """2-Second Report: Only sends if positions are open"""
        while True:
            active_trades = [p for p in self.positions if p['status'] == 'OPEN']
            if active_trades:
                floating = 0.0
                for p in active_trades:
                    try:
                        ticker = self.exchange.fetch_ticker(p['sym'])
                        floating += (ticker['last'] - p['entry']) * p['size']
                    except: continue
                
                net = self.realized_pnl + floating - self.total_fees
                msg = (f"📊 <b>LIVE TRADE REPORT</b>\n"
                       f"--------------------\n"
                       f"💰 <b>Wallet:</b> ${self.wallet_balance:.2f}\n"
                       f"📈 <b>Realized:</b> ${self.realized_pnl:.2f}\n"
                       f"📉 <b>Floating:</b> ${floating:.2f}\n"
                       f"💸 <b>Fees:</b> ${self.total_fees:.2f}\n"
                       f"🚀 <b>Net Profit:</b> ${net:.2f}\n"
                       f"--------------------\n"
                       f"Active Positions: {len(active_trades)}")
                send_telegram(msg)
            await asyncio.sleep(2)

    async def trading_loop(self):
        while True:
            # Daily AI Re-optimization
            if not self.ai.is_trained or datetime.now().day != self.ai.last_train_day:
                hist = self.exchange.fetch_ohlcv("BTC/USDT", TIMEFRAME, limit=1000)
                df_hist = TechnicalIndicators.add_indicators(pd.DataFrame(hist, columns=['t','o','h','l','c','v']).rename(columns={'c':'close','h':'high','l':'low'}))
                self.ai.train(df_hist)

            for symbol in SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                    df = TechnicalIndicators.add_indicators(df)
                    row = df.iloc[-1:]
                    
                    prob = self.ai.predict(row)
                    
                    # LOOSENED ENTRY: Only check Probability and Position Count
                    if prob > MIN_PROBABILITY and len([x for x in self.positions if x['status']=='OPEN']) < MAX_POSITIONS:
                        price = df['close'].iloc[-1]
                        trade_val = self.wallet_balance * 0.05 # Risk 5% of balance per trade
                        fee = trade_val * TAKER_FEE
                        
                        self.wallet_balance -= (trade_val + fee)
                        self.total_fees += fee
                        self.positions.append({'sym': symbol, 'entry': price, 'size': trade_val/price, 'status': 'OPEN'})
                        
                        send_telegram(f"🚀 <b>TRADE OPENED:</b> {symbol}\nProb: {prob:.1%}\nPrice: {price}")
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
            
            await asyncio.sleep(5) # Scan symbols every 5 seconds

    async def run(self):
        logger.info("🤖 Sniper Bot Active - Aggressive Mode")
        await asyncio.gather(self.trading_loop(), self.report_loop())

if __name__ == "__main__":
    bot = SuperSniper()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot Stopped by User")

