import pandas as pd
import numpy as np
import requests, asyncio, os, joblib, logging, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import ccxt
from datetime import datetime

# ================= CONFIG (Enhanced) =================
# Added your top 20 symbols as requested
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", 
           "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "LINK/USDT", "TRX/USDT", 
           "POL/USDT", "LTC/USDT", "BCH/USDT", "SHIB/USDT", "NEAR/USDT", 
           "APT/USDT", "SUI/USDT", "ICP/USDT", "RENDER/USDT", "STX/USDT"]

TIMEFRAME = "5m"
INITIAL_BALANCE = 10000.0
MAX_POSITIONS = 3
KELLY_FRACTION = 0.2 
MIN_PROBABILITY = 0.72 
TAKER_FEE = 0.0010
TG_TOKEN = "8560134874:AAHF4efOAdsg2Y01eBHF-2DzEUNf9WAdniA"
TG_CHAT = "5665906172"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("SuperSniper")

def send_telegram(message: str):
    try: requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                       json={"chat_id": TG_CHAT, "text": message, "parse_mode": "HTML"}, timeout=5)
    except: pass

class QuantEngine:
    @staticmethod
    def get_features(df):
        if len(df) < 50: return df
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / 
                                      df['close'].diff().where(df['close'].diff() < 0, 0).abs().rolling(14).mean().replace(0, 0.001))))
        return df.dropna()

class SuperAI:
    def __init__(self):
        self.model = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200))])
        self.regime_detector = KMeans(n_clusters=3, random_state=42)
        self.is_trained = False

    def train(self, historical_data):
        X = historical_data[['rsi', 'volatility', 'returns']]
        historical_data['regime'] = self.regime_detector.fit_predict(X)
        y = (historical_data['close'].shift(-10) > historical_data['close'] * 1.025).astype(int)
        self.model.fit(X, y.fillna(0))
        self.is_trained = True

    def get_trade_conviction(self, current_row):
        X = current_row[['rsi', 'volatility', 'returns']]
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

    def calculate_kelly_size(self, prob):
        p, b = prob, 2.0 
        q = 1 - p
        k_percent = (p * b - q) / b
        return max(0, k_percent * KELLY_FRACTION)

    def get_floating_pnl(self):
        floating = 0.0
        for pos in self.positions:
            if pos['status'] == 'OPEN':
                curr_price = self.exchange.fetch_ticker(pos['sym'])['last']
                floating += (curr_price - pos['entry']) * pos['size']
        return floating

    async def send_2s_report(self):
        """The 'Older Bot' style high-frequency report"""
        floating_pnl = self.get_floating_pnl()
        net_profit = self.realized_pnl + floating_pnl - self.total_fees
        
        report = (
            f"📊 <b>LIVE REPORT</b>\n"
            f"---------------------------\n"
            f"💰 <b>Balance:</b> ${self.wallet_balance:.2f}\n"
            f"📈 <b>Realized:</b> ${self.realized_pnl:.2f}\n"
            f"📉 <b>Floating:</b> ${floating_pnl:.2f}\n"
            f"💸 <b>Fees:</b> ${self.total_fees:.2f}\n"
            f"🚀 <b>Net Profit:</b> ${net_profit:.2f}\n"
            f"---------------------------\n"
            f"Active Trades: {len([p for p in self.positions if p['status'] == 'OPEN'])}"
        )
        print(report.replace("<b>","").replace("</b>","")) # Print to console
        send_telegram(report)

    async def run(self):
        last_report_time = 0
        while True:
            # 1. High Frequency Reporting (Every 2 Seconds)
            current_time = time.time()
            if current_time - last_report_time >= 2:
                await self.send_2s_report()
                last_report_time = current_time

            # 2. Main Trading Logic
            for symbol in SYMBOLS:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                    df = pd.DataFrame(ohlcv, columns=['t','open','high','low','close','v'])
                    df = QuantEngine.get_features(df)
                    
                    if not self.ai.is_trained: self.ai.train(df)
                    
                    prob, regime = self.ai.get_trade_conviction(df.iloc[-1:])
                    
                    # Entry Logic
                    if prob > MIN_PROBABILITY and regime != 0 and len(self.positions) < MAX_POSITIONS:
                        price = df['close'].iloc[-1]
                        k_size = self.calculate_kelly_size(prob)
                        trade_amount = self.wallet_balance * k_size
                        
                        fee = trade_amount * TAKER_FEE
                        self.total_fees += fee
                        self.wallet_balance -= (trade_amount + fee)
                        
                        self.positions.append({
                            'sym': symbol, 'entry': price, 'size': trade_amount/price, 
                            'status':'OPEN', 'tp': price * 1.05, 'sl': price * 0.97
                        })
                        send_telegram(f"💎 <b>KELLY BUY:</b> {symbol}\nProb: {prob:.1%}")
                except Exception as e:
                    continue
            
            await asyncio.sleep(0.1) # Fast loop for responsiveness

if __name__ == "__main__":
    bot = SuperSniper()
    asyncio.run(bot.run())
