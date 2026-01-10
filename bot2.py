import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import ccxt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ============================================================================
# ULTIMATE CONFIGURATION - OPTIMIZED FOR ACTIVITY & ACCURACY
# ============================================================================

class UltimateConfig:
    EXCHANGE = 'binance'
    
    # ADDED TOP 20 CRYPTO SYMBOLS AS REQUESTED
    SYMBOLS = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
        'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT',
        'LINK/USDT', 'MATIC/USDT', 'NEAR/USDT', 'LTC/USDT', 'BCH/USDT',
        'SHIB/USDT', 'UNI/USDT', 'STX/USDT', 'FIL/USDT', 'ARB/USDT'
    ]
    
    TIMEFRAMES = ['1m', '5m']
    PRIMARY_TIMEFRAME = '1m'
    
    # Capital Management
    INITIAL_CAPITAL = 100.0
    RISK_PER_TRADE = 0.02      # 2% risk
    MAX_OPEN_TRADES = 5        # Concentrated quality
    
    # Scalping Parameters - ADJUSTED TO PREVENT INSTANT LOSS
    BASE_TAKE_PROFIT = 0.0080   # 0.8% (Higher to cover fees)
    BASE_STOP_LOSS = 0.0045     # 0.45% (Give trade room to breathe)
    BREAK_EVEN_TRIGGER = 0.0030 # Move SL to entry at 0.3% profit
    TRAILING_STOP = 0.0020      
    
    # Deep Learning Parameters - LOOSENED FOR MORE FREQUENT TRADES
    DL_LOOKBACK = 100           
    DL_TRAIN_INTERVAL = 50      
    DL_MIN_CONFIDENCE = 0.65    # Reduced from 0.75 to increase trade frequency
    
    # Fees
    TAKER_FEE = 0.0010
    
    CHECK_INTERVAL = 10         # Seconds between market scans
    TELEGRAM_ENABLED = True

# ============================================================================
# IMPROVED PORTFOLIO MANAGER (FEATURING BREAK-EVEN LOGIC)
# ============================================================================

class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_trades = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        
    def check_exit_conditions(self, trade: Dict) -> Optional[str]:
        pnl_pct = trade['pnl_pct']
        
        # 1. BREAK-EVEN LOGIC (Protects from turning profit into loss)
        if pnl_pct >= UltimateConfig.BREAK_EVEN_TRIGGER:
            if not trade.get('is_break_even', False):
                trade['stop_loss'] = -0.0002 # Set SL slightly above entry to cover dust
                trade['is_break_even'] = True
                logging.info(f"PROTECTION: Moved {trade['symbol']} to Break-Even.")

        # 2. TAKE PROFIT
        if pnl_pct >= trade['take_profit']:
            return f"TP Hit: {pnl_pct:.2%}"
        
        # 3. STOP LOSS
        if pnl_pct <= -trade['stop_loss']:
            return f"SL Hit: {pnl_pct:.2%}"
            
        return None

    def add_trade(self, trade_record: Dict):
        trade_record['is_break_even'] = False # Initialize break-even tracker
        self.open_trades[trade_record['trade_id']] = trade_record
        self.current_capital -= trade_record['size']
        logging.info(f"OPENED: {trade_record['symbol']} at {trade_record['entry_price']}")

    def update_trades(self, market_prices: Dict):
        trades_to_close = []
        for tid, trade in self.open_trades.items():
            if trade['symbol'] not in market_prices: continue
            
            curr_price = market_prices[trade['symbol']]
            trade['current_price'] = curr_price
            
            # P&L Calculation including fees
            raw_pnl = (curr_price - trade['entry_price']) / trade['entry_price'] if trade['direction'] == 'BUY' else (trade['entry_price'] - curr_price) / trade['entry_price']
            trade['pnl_pct'] = raw_pnl - (UltimateConfig.TAKER_FEE * 2)
            
            exit_reason = self.check_exit_conditions(trade)
            if exit_reason:
                trades_to_close.append((tid, exit_reason))
        
        for tid, reason in trades_to_close:
            self.close_trade(tid, reason)

    def close_trade(self, trade_id: str, reason: str):
        trade = self.open_trades.pop(trade_id)
        final_pnl_usd = trade['pnl_pct'] * trade['size']
        self.current_capital += trade['size'] + final_pnl_usd
        self.daily_pnl += final_pnl_usd
        self.trade_history.append(trade)
        logging.info(f"CLOSED: {trade['symbol']} | P&L: ${final_pnl_usd:.2f} | Reason: {reason}")

# ============================================================================
# DATA & PREDICTION (SIMPLIFIED FOR SPEED)
# ============================================================================

class DeepLearningPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def create_features(self, df):
        df = df.copy()
        df['rsi'] = self.calc_rsi(df['close'], 14)
        df['ema_fast'] = df['close'].ewm(span=8).mean()
        df['ema_slow'] = df['close'].ewm(span=21).mean()
        df['vol_change'] = df['volume'].pct_change()
        return df.fillna(0)

    def calc_rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def predict(self, symbol, df):
        # Fallback to technical logic if model isn't trained yet
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish logic: EMA Cross + RSI not overbought
        if last['ema_fast'] > last['ema_slow'] and prev['ema_fast'] <= prev['ema_slow'] and last['rsi'] < 65:
            return {'direction': 'BUY', 'confidence': 0.70}
        # Bearish logic: EMA Cross + RSI not oversold
        elif last['ema_fast'] < last['ema_slow'] and prev['ema_fast'] >= prev['ema_slow'] and last['rsi'] > 35:
            return {'direction': 'SELL', 'confidence': 0.70}
            
        return {'direction': 'HOLD', 'confidence': 0}

# ============================================================================
# MAIN BOT ENGINE
# ============================================================================

class DeepLearningScalpingBot:
    def __init__(self):
        self.predictor = DeepLearningPredictor()
        self.portfolio = PortfolioManager(UltimateConfig.INITIAL_CAPITAL)
        self.exchange = ccxt.binance()
        logging.basicConfig(level=logging.INFO)

    def fetch_data(self, symbol):
        try:
            bars = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=50)
            df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            return self.predictor.create_features(df)
        except:
            return None

    def run(self):
        print("🚀 Scalping Bot Started - High Activity Mode")
        while True:
            market_prices = {}
            
            for symbol in UltimateConfig.SYMBOLS:
                df = self.fetch_data(symbol)
                if df is None: continue
                
                price = df['close'].iloc[-1]
                market_prices[symbol] = price
                
                # Check for signals if we have room in portfolio
                if len(self.portfolio.open_trades) < UltimateConfig.MAX_OPEN_TRADES:
                    if symbol not in [t['symbol'] for t in self.portfolio.open_trades.values()]:
                        signal = self.predictor.predict(symbol, df)
                        
                        if signal['direction'] != 'HOLD' and signal['confidence'] >= UltimateConfig.DL_MIN_CONFIDENCE:
                            trade_size = self.portfolio.current_capital * UltimateConfig.RISK_PER_TRADE
                            trade_rec = {
                                'trade_id': f"{symbol}_{int(time.time())}",
                                'symbol': symbol,
                                'direction': signal['direction'],
                                'entry_price': price,
                                'size': trade_size,
                                'take_profit': UltimateConfig.BASE_TAKE_PROFIT,
                                'stop_loss': UltimateConfig.BASE_STOP_LOSS,
                                'pnl_pct': 0
                            }
                            self.portfolio.add_trade(trade_rec)
                
                time.sleep(0.1) # Prevent rate limits
            
            self.portfolio.update_trades(market_prices)
            time.sleep(UltimateConfig.CHECK_INTERVAL)

if __name__ == "__main__":
    bot = DeepLearningScalpingBot()
    bot.run()
()      # optimal price based on direction and order
