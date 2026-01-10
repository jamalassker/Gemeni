import os
import time
import json
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import ccxt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ============================================================================
# ULTIMATE CONFIGURATION - RELAXED FOR MORE TRADES
# ============================================================================

class UltimateConfig:
    EXCHANGE = 'binance'
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT',
        'LINK/USDT', 'MATIC/USDT', 'NEAR/USDT', 'LTC/USDT', 'BCH/USDT',
        'SHIB/USDT', 'UNI/USDT', 'STX/USDT', 'FIL/USDT', 'ARB/USDT'
    ]
    
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.05      # Increased risk to 5% to grow balance faster
    MAX_OPEN_TRADES = 8        # Increased to 8 to capture more opportunities
    
    # Scalping Parameters
    BASE_TAKE_PROFIT = 0.0075   # 0.75% (Slightly lower for faster exits)
    BASE_STOP_LOSS = 0.0050     # 0.50%
    BREAK_EVEN_TRIGGER = 0.0025 # Move to BE earlier at 0.25%
    
    # Relaxed Entry Rules
    DL_MIN_CONFIDENCE = 0.60    # Lowered to trigger more trades
    TAKER_FEE = 0.0010
    CHECK_INTERVAL = 5         # Scans every 5 seconds instead of 10
    
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================

def send_telegram_msg(message: str):
    if not UltimateConfig.TELEGRAM_TOKEN or not UltimateConfig.TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{UltimateConfig.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": UltimateConfig.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_trades = {}
        
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        pnl_pct = trade['pnl_pct']
        
        if pnl_pct >= UltimateConfig.BREAK_EVEN_TRIGGER:
            if not trade.get('is_break_even', False):
                trade['stop_loss'] = -0.0001 # Risk-free
                trade['is_break_even'] = True
                send_telegram_msg(f"🛡️ <b>SAFE: {trade['symbol']}</b> (BE Set)")

        if pnl_pct >= trade['take_profit']: return f"TP: {pnl_pct:.2%}"
        if pnl_pct <= -trade['stop_loss']: return f"SL: {pnl_pct:.2%}"
        return None

    def add_trade(self, trade_record: Dict):
        trade_record['is_break_even'] = False
        self.open_trades[trade_record['trade_id']] = trade_record
        self.current_capital -= trade_record['size']
        
        msg = (f"💰 <b>MONEY IN: {trade_record['symbol']}</b>\n"
               f"Price: ${trade_record['entry_price']:.4f}\n"
               f"Risk: {UltimateConfig.RISK_PER_TRADE:.0%}")
        send_telegram_msg(msg)

    def update_trades(self, market_prices: Dict):
        trades_to_close = []
        for tid, trade in self.open_trades.items():
            if trade['symbol'] not in market_prices: continue
            curr_price = market_prices[trade['symbol']]
            raw_pnl = (curr_price - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'BUY' else (trade['entry_price'] - curr_price) / trade['entry_price']
            trade['pnl_pct'] = raw_pnl - (UltimateConfig.TAKER_FEE * 2)
            exit_reason = self.check_exit_conditions(trade)
            if exit_reason: trades_to_close.append((tid, exit_reason))
        
        for tid, reason in trades_to_close:
            self.close_trade(tid, reason)

    def close_trade(self, trade_id: str, reason: str):
        trade = self.open_trades.pop(trade_id)
        final_pnl_usd = trade['pnl_pct'] * trade['size']
        self.current_capital += trade['size'] + final_pnl_usd
        
        icon = "💵" if final_pnl_usd > 0 else "📉"
        msg = (f"{icon} <b>RESULT: {trade['symbol']}</b>\n"
               f"P&L: <b>${final_pnl_usd:+.2f}</b>\n"
               f"Balance: ${self.current_capital:.2f}")
        send_telegram_msg(msg)

# ============================================================================
# PREDICTOR (RELAXED RULES)
# ============================================================================

class DeepLearningPredictor:
    def create_features(self, df):
        df = df.copy()
        df['rsi'] = self.calc_rsi(df['close'], 14)
        df['ema_fast'] = df['close'].ewm(span=8).mean()
        df['ema_slow'] = df['close'].ewm(span=21).mean()
        return df.fillna(0)

    def calc_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def predict(self, symbol, df):
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # RELAXED RULES: EMA Cross + Wider RSI Range
        if last['ema_fast'] > last['ema_slow'] and last['rsi'] < 70:
            return {'direction': 'BUY', 'confidence': 0.75}
        elif last['ema_fast'] < last['ema_slow'] and last['rsi'] > 30:
            return {'direction': 'SELL', 'confidence': 0.75}
        return {'direction': 'HOLD', 'confidence': 0}

# ============================================================================
# ENGINE
# ============================================================================

class DeepLearningScalpingBot:
    def __init__(self):
        self.predictor = DeepLearningPredictor()
        self.portfolio = PortfolioManager(UltimateConfig.INITIAL_CAPITAL)
        self.exchange = ccxt.binance()
        logging.basicConfig(level=logging.INFO)

    def run(self):
        send_telegram_msg("🔥 <b>Aggressive Mode Active</b>\nSearching for profits...")
        while True:
            market_prices = {}
            for symbol in UltimateConfig.SYMBOLS:
                try:
                    bars = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=30)
                    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    df = self.predictor.create_features(df)
                    price = df['close'].iloc[-1]
                    market_prices[symbol] = price
                    
                    if len(self.portfolio.open_trades) < UltimateConfig.MAX_OPEN_TRADES:
                        if symbol not in [t['symbol'] for t in self.portfolio.open_trades.values()]:
                            signal = self.predictor.predict(symbol, df)
                            if signal['direction'] != 'HOLD':
                                self.portfolio.add_trade({
                                    'trade_id': f"{symbol}_{int(time.time())}",
                                    'symbol': symbol,
                                    'direction': signal['direction'],
                                    'entry_price': price,
                                    'size': self.portfolio.current_capital * UltimateConfig.RISK_PER_TRADE,
                                    'take_profit': UltimateConfig.BASE_TAKE_PROFIT,
                                    'stop_loss': UltimateConfig.BASE_STOP_LOSS,
                                    'pnl_pct': 0
                                })
                    time.sleep(0.05)
                except: continue
            
            self.portfolio.update_trades(market_prices)
            time.sleep(UltimateConfig.CHECK_INTERVAL)

if __name__ == "__main__":
    DeepLearningScalpingBot().run()


