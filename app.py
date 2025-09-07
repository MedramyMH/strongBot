import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
import time
import logging
import json
import csv
import pickle
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constants - Focus on shorter timeframes
TIMEFRAME_CONFIG = {
    '1m': {'interval': '1m', 'period': '1d', 'contract': '1min'},
    '5m': {'interval': '5m', 'period': '5d', 'contract': '5min'},
    '15m': {'interval': '15m', 'period': '15d', 'contract': '15min'},
}

# Reduced set of assets for faster processing - focus on highly liquid instruments
TOP_ASSETS = {
    'forex': [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'
    ],
    'crypto': [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD'
    ],
    'stocks': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'
    ],
    'indices': [
        '^GSPC', '^DJI', '^IXIC', 'SPY', 'QQQ'
    ]
}

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str
    timeframe: str
    contract_period: str
    entry_low: float
    entry_high: float
    target: float
    stop_loss: float
    confidence: float
    reasoning: str
    timestamp: datetime
    remaining_seconds: int
    risk_reward_ratio: float
    market_type: str
    outcome: str = None
    outcome_price: float = None
    outcome_timestamp: datetime = None

class MLTradeOptimizer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)  # Simplified for speed
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data_path = 'data/training_data.csv'
        self.model_path = 'models/trade_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded existing ML model and scaler")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def save_model(self):
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Saved ML model and scaler")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def prepare_features(self, indicators):
        features = []
        
        # Essential features only for speed
        features.append(indicators.get('price', 0))
        features.append(indicators.get('sma_10', 0))
        features.append(indicators.get('ema_12', 0))
        features.append(indicators.get('rsi', 50))
        features.append(indicators.get('macd', 0))
        features.append(indicators.get('atr', 0))
        features.append(indicators.get('volume_ratio', 1))
        features.append(indicators.get('bb_position', 0.5))
        
        return np.array(features).reshape(1, -1)
    
    def predict_success_probability(self, indicators, signal_type):
        if not self.is_trained:
            return 0.5
            
        try:
            features = self.prepare_features(indicators)
            features_scaled = self.scaler.transform(features)
            
            proba = self.model.predict_proba(features_scaled)[0]
            
            if signal_type == "BUY":
                return proba[1]
            else:
                return proba[0]
                
        except Exception as e:
            logger.error(f"Error predicting success probability: {e}")
            return 0.5
    
    def train_model(self, data):
        if len(data) < 30:
            logger.warning("Insufficient data for training")
            return False
        
        try:
            X = []
            y = []
            
            for trade in data:
                if trade['outcome'] is not None:
                    features = self.prepare_features(trade['indicators']).flatten()
                    X.append(features)
                    
                    success = 1 if trade['outcome'] == 'WIN' else 0
                    y.append(success)
            
            if len(X) < 15:
                logger.warning("Not enough labeled examples for training")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            
            self.is_trained = True
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

class TradeDataManager:
    def __init__(self):
        self.data_file = 'data/trade_history.csv'
        self.initialize_data_file()
    
    def initialize_data_file(self):
        if not os.path.exists(self.data_file):
            os.makedirs('data', exist_ok=True)
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'signal_type', 'timeframe', 'market_type',
                    'entry_low', 'entry_high', 'target', 'stop_loss', 'confidence',
                    'reasoning', 'risk_reward_ratio', 'outcome', 'outcome_price',
                    'outcome_timestamp', 'indicators'
                ])
    
    def save_trade(self, signal, indicators):
        try:
            with open(self.data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.signal_type,
                    signal.timeframe,
                    signal.market_type,
                    signal.entry_low,
                    signal.entry_high,
                    signal.target,
                    signal.stop_loss,
                    signal.confidence,
                    signal.reasoning,
                    signal.risk_reward_ratio,
                    signal.outcome or '',
                    signal.outcome_price or '',
                    signal.outcome_timestamp.isoformat() if signal.outcome_timestamp else '',
                    json.dumps(indicators)
                ])
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
    
    def load_trade_history(self, limit=1000):
        trades = []
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
                for _, row in df.iterrows():
                    trade = {
                        'timestamp': pd.to_datetime(row['timestamp']),
                        'symbol': row['symbol'],
                        'signal_type': row['signal_type'],
                        'timeframe': row['timeframe'],
                        'market_type': row['market_type'],
                        'entry_low': float(row['entry_low']),
                        'entry_high': float(row['entry_high']),
                        'target': float(row['target']),
                        'stop_loss': float(row['stop_loss']),
                        'confidence': float(row['confidence']),
                        'reasoning': row['reasoning'],
                        'risk_reward_ratio': float(row['risk_reward_ratio']),
                        'outcome': row['outcome'] if pd.notna(row['outcome']) else None,
                        'outcome_price': float(row['outcome_price']) if pd.notna(row['outcome_price']) else None,
                        'outcome_timestamp': pd.to_datetime(row['outcome_timestamp']) if pd.notna(row['outcome_timestamp']) else None,
                        'indicators': json.loads(row['indicators']) if pd.notna(row['indicators']) else {}
                    }
                    trades.append(trade)
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
        
        return trades[-limit:]

class OptimizedSignalGenerator:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 10  # Reduced for faster trading
        self.data_manager = TradeDataManager()
        self.ml_optimizer = MLTradeOptimizer()
        self.selected_assets = set()
    
    def set_selected_assets(self, assets):
        self.selected_assets = set(assets)
        logger.info(f"Set selected assets: {self.selected_assets}")
    
    def generate_signal(self, symbol: str, timeframe: str, market_type: str) -> Optional[TradingSignal]:
        try:
            # Skip if not in selected assets
            if self.selected_assets and symbol not in self.selected_assets:
                return None
                
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            current_time = time.time()
            if cache_key in self.cache and current_time - self.cache[cache_key]['timestamp'] < self.cache_timeout:
                return self.cache[cache_key]['signal']
            
            # Fetch data - faster for 1m timeframe
            config = TIMEFRAME_CONFIG[timeframe]
            ticker = yf.Ticker(symbol)
            
            # For 1m timeframe, use shorter period to speed up data retrieval
            if timeframe == '1m':
                df = ticker.history(period='1d', interval='1m')
            else:
                df = ticker.history(period=config['period'], interval=config['interval'])
            
            if df.empty or len(df) < 20:  # Reduced minimum for faster signals
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate optimized indicators for fast trading
            indicators = self._calculate_fast_indicators(df)
            if not indicators:
                logger.warning(f"No indicators calculated for {symbol}")
                return None
                
            # Generate signal optimized for fast trading
            signal_type, confidence, reasoning = self._generate_fast_signal(indicators, market_type, df, timeframe)
            
            # Skip low confidence signals
            if confidence < 60:  # Lower threshold for more signals
                return None
                
            # Use ML to predict success probability
            ml_confidence = self.ml_optimizer.predict_success_probability(indicators, signal_type)
            # Higher weight to ML for fast trading
            final_confidence = (confidence * 0.6) + (ml_confidence * 100 * 0.4)
            
            logger.info(f"{symbol}: {signal_type} signal with {final_confidence:.1f}% confidence - {reasoning}")
            
            # Calculate targets for fast trading - tighter stops and targets
            price = indicators['price']
            atr = indicators.get('atr', price * 0.005)  # Smaller ATR for fast trades
            
            if signal_type == "BUY":
                stop_loss = price - (atr * 1.5)  # Tighter stop loss
                target = price + (atr * 2.0)     # Quicker target
                risk_reward = 1.33
            elif signal_type == "SELL":
                stop_loss = price + (atr * 1.5)
                target = price - (atr * 2.0)
                risk_reward = 1.33
            else:
                return None
            
            # Narrower entry zone for fast execution
            entry_band = atr * 0.2
            entry_low = price - entry_band
            entry_high = price + entry_band
            
            # Calculate remaining time - shorter for fast trades
            remaining = self._calculate_remaining_time(timeframe)
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                timeframe=timeframe,
                contract_period=config['contract'],
                entry_low=entry_low,
                entry_high=entry_high,
                target=target,
                stop_loss=stop_loss,
                confidence=final_confidence,
                reasoning=reasoning + f" | ML: {ml_confidence:.2f}",
                timestamp=datetime.now(),
                remaining_seconds=remaining,
                risk_reward_ratio=risk_reward,
                market_type=market_type
            )
            
            # Save the trade to CSV
            self.data_manager.save_trade(signal, indicators)
            
            # Cache the signal with shorter timeout
            self.cache[cache_key] = {
                'signal': signal,
                'timestamp': current_time
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _calculate_fast_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate fast indicators optimized for 1m trading"""
        try:
            close = df['Close'].astype(float)
            high = df['High'].astype(float)
            low = df['Low'].astype(float)
            volume = df['Volume'].astype(float)
            
            indicators = {}
            indicators['price'] = float(close.iloc[-1])
            
            # Faster moving averages for quick signals
            indicators['sma_10'] = float(close.rolling(10).mean().iloc[-1])
            indicators['ema_12'] = float(close.ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(close.ewm(span=26).mean().iloc[-1])
            
            # Faster MACD
            exp12 = close.ewm(span=12).mean()
            exp26 = close.ewm(span=26).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9).mean()
            
            indicators['macd'] = float(macd.iloc[-1])
            indicators['macd_signal'] = float(signal.iloc[-1])
            
            # Faster RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(10).mean()  # Shorter period
            avg_loss = loss.rolling(10).mean()  # Shorter period
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
            # Faster Bollinger Bands
            sma10 = close.rolling(10).mean()  # Shorter period
            std10 = close.rolling(10).std()   # Shorter period
            bb_upper = sma10 + (2 * std10)
            bb_lower = sma10 - (2 * std10)
            indicators['bb_upper'] = float(bb_upper.iloc[-1])
            indicators['bb_lower'] = float(bb_lower.iloc[-1])
            
            # Bollinger Band position
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if bb_range > 0:
                indicators['bb_position'] = (indicators['price'] - bb_lower.iloc[-1]) / bb_range
            else:
                indicators['bb_position'] = 0.5
            
            # Faster ATR
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(10).mean()  # Shorter period
            indicators['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else indicators['price'] * 0.005
            
            # Volume ratio with shorter period
            if len(volume) >= 5:
                vol_avg = volume.rolling(5).mean()  # Shorter period
                indicators['volume_ratio'] = float(volume.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0
            else:
                indicators['volume_ratio'] = 1.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Fast indicator calculation error: {e}")
            return {}
    def _generate_fast_signal(self, ind: Dict, market_type: str, df: pd.DataFrame, timeframe: str) -> Tuple[str, float, str]:
        """Generate signals optimized for fast trading with better validation"""
        
        try:
            # Validate indicators
            if not all(key in ind for key in ['price', 'sma_10', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'bb_position', 'volume_ratio']):
                return "WAIT", 50, "Incomplete indicator data"
            
            price = ind['price']
            sma_10 = ind['sma_10']
            ema_12 = ind['ema_12']
            ema_26 = ind['ema_26']
            rsi = ind['rsi']
            macd = ind['macd']
            macd_signal = ind['macd_signal']
            bb_position = ind['bb_position']
            volume_ratio = ind['volume_ratio']
            
            # Validate values
            if any(np.isnan([price, sma_10, ema_12, ema_26, rsi, macd, macd_signal, bb_position, volume_ratio])):
                return "WAIT", 50, "Invalid indicator values"
            
            # Fast trading strategies
            signals = []
            confidences = []
            reasons = []
            
            # 1. Fast trend following with momentum confirmation
            if ema_12 > ema_26 and price > sma_10 and macd > macd_signal:
                signals.append("BUY")
                confidences.append(70)
                reasons.append("Fast uptrend with momentum")
            elif ema_12 < ema_26 and price < sma_10 and macd < macd_signal:
                signals.append("SELL")
                confidences.append(70)
                reasons.append("Fast downtrend with momentum")
            
            # 2. Fast RSI with volume confirmation
            if rsi < 35 and volume_ratio > 1.5:
                signals.append("BUY")
                confidences.append(75)
                reasons.append(f"RSI oversold at {rsi:.0f} with volume")
            elif rsi > 65 and volume_ratio > 1.5:
                signals.append("SELL")
                confidences.append(75)
                reasons.append(f"RSI overbought at {rsi:.0f} with volume")
            
            # 3. Fast Bollinger Bands with RSI confirmation
            if bb_position < 0.2 and rsi < 40:
                signals.append("BUY")
                confidences.append(72)
                reasons.append("Near lower BB with RSI confirmation")
            elif bb_position > 0.8 and rsi > 60:
                signals.append("SELL")
                confidences.append(72)
                reasons.append("Near upper BB with RSI confirmation")
            
            # 4. MACD crossover with price confirmation
            if macd > macd_signal and price > ema_12:
                signals.append("BUY")
                confidences.append(68)
                reasons.append("MACD bullish with price confirmation")
            elif macd < macd_signal and price < ema_12:
                signals.append("SELL")
                confidences.append(68)
                reasons.append("MACD bearish with price confirmation")
            
            # If no signals, return WAIT
            if not signals:
                return "WAIT", 50, "No clear fast signal with current confirmations"
            
            # Find the strongest signal
            max_confidence = max(confidences)
            best_index = confidences.index(max_confidence)
            best_signal = signals[best_index]
            
            # Count confirmations
            same_direction = sum(1 for s in signals if s == best_signal)
            
            # Adjust confidence based on confirmations
            if same_direction > 1:
                confidence_boost = min(15, (same_direction - 1) * 5)
                max_confidence = min(95, max_confidence + confidence_boost)
            
            # Compile reasoning
            main_reason = reasons[best_index]
            if same_direction > 1:
                main_reason += f" + {same_direction-1} confirmations"
            
            return best_signal, max_confidence, main_reason
            
        except Exception as e:
            logger.error(f"Fast signal generation error: {e}")
            return "WAIT", 50, f"Error in fast signal generation: {str(e)}"
    
    def _calculate_remaining_time(self, timeframe: str) -> int:
        now = datetime.now()
        
        try:
            if timeframe == '1m':
                next_close = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            elif timeframe == '5m':
                minutes = (now.minute // 5 + 1) * 5
                if minutes >= 60:
                    next_close = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_close = now.replace(minute=minutes, second=0, microsecond=0)
            elif timeframe == '15m':
                minutes = (now.minute // 15 + 1) * 15
                if minutes >= 60:
                    next_close = now.replace(hour=now.hour+1, minute=0, second=0, microsecond=0)
                else:
                    next_close = now.replace(minute=minutes, second=0, microsecond=0)
            else:
                next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                
            return max(1, int((next_close - now).total_seconds()))
        except Exception:
            return 30  # Default to 30 seconds for fast trading
    
    def evaluate_outcomes(self):
        """Evaluate outcomes of recent trades for fast trading"""
        try:
            # Load recent trade history
            trades = self.data_manager.load_trade_history(limit=200)
            
            # Check for trades that need outcome evaluation
            for trade in trades:
                if trade['outcome'] is None:
                    # For fast trading, check outcomes more quickly
                    symbol = trade['symbol']
                    ticker = yf.Ticker(symbol)
                    current_data = ticker.history(period='1d', interval='1m')
                    
                    if not current_data.empty:
                        current_price = float(current_data['Close'].iloc[-1])
                        
                        if trade['signal_type'] == 'BUY':
                            if current_price >= trade['target']:
                                trade['outcome'] = 'WIN'
                                trade['outcome_price'] = current_price
                            elif current_price <= trade['stop_loss']:
                                trade['outcome'] = 'LOSS'
                                trade['outcome_price'] = current_price
                        else:  # SELL
                            if current_price <= trade['target']:
                                trade['outcome'] = 'WIN'
                                trade['outcome_price'] = current_price
                            elif current_price >= trade['stop_loss']:
                                trade['outcome'] = 'LOSS'
                                trade['outcome_price'] = current_price
                        
                        # Update outcome timestamp
                        trade['outcome_timestamp'] = datetime.now()
            
            # Train ML model with updated data
            self.ml_optimizer.train_model(trades)
            
        except Exception as e:
            logger.error(f"Error evaluating outcomes: {e}")

# Flask App
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

signal_generator = OptimizedSignalGenerator()
current_signals = {}
update_lock = threading.Lock()

def update_signals():
    """Background task to update signals - faster for 1m trading"""
    while True:
        try:
            logger.info("Updating fast trading signals...")
            new_signals = {}
            generated_count = 0
            
            # Get the current selected assets
            selected_assets = signal_generator.selected_assets
            
            # Process only selected assets if any are selected
            assets_to_process = {}
            if selected_assets:
                logger.info(f"Processing only selected assets: {selected_assets}")
                # Find which market each selected asset belongs to
                for market_type, symbols in TOP_ASSETS.items():
                    for symbol in symbols:
                        if symbol in selected_assets:
                            if market_type not in assets_to_process:
                                assets_to_process[market_type] = []
                            assets_to_process[market_type].append(symbol)
            else:
                # If no assets are selected, process all assets
                logger.info("No assets selected, processing all assets")
                assets_to_process = TOP_ASSETS
            
            # Process the selected assets
            for market_type, symbols in assets_to_process.items():
                for symbol in symbols:
                    # Focus on 1m timeframe for fast trading
                    for timeframe in ['1m']:
                        try:
                            signal = signal_generator.generate_signal(symbol, timeframe, market_type)
                            if signal and signal.confidence >= 60:
                                key = f"{symbol}_{timeframe}"
                                new_signals[key] = signal
                                generated_count += 1
                                if generated_count <= 5:
                                    logger.info(f"✓ {signal.signal_type} {symbol} {timeframe} ({signal.confidence:.0f}%)")
                        except Exception as e:
                            logger.error(f"Error processing {symbol} {timeframe}: {e}")
                            continue
                            
            with update_lock:
                current_signals.clear()
                current_signals.update(new_signals)
                logger.info(f"✓ Generated {len(new_signals)} fast trading signals")
                
        except Exception as e:
            logger.error(f"Update signals error: {e}")
            
        time.sleep(5)  # Faster updates for 1m trading

def evaluate_trade_outcomes():
    """Background task to evaluate trade outcomes and train ML model"""
    while True:
        try:
            logger.info("Evaluating fast trade outcomes and training ML model...")
            
            # Evaluate outcomes
            signal_generator.evaluate_outcomes()
            
            # Load recent trade history for training
            trades = signal_generator.data_manager.load_trade_history(limit=200)
            
            # Count how many trades have outcomes
            trades_with_outcomes = [t for t in trades if t['outcome'] is not None]
            logger.info(f"Found {len(trades_with_outcomes)} trades with outcomes")
            
            # Train ML model if we have enough data
            if len(trades_with_outcomes) >= 20:
                logger.info("Training ML model with recent trade data...")
                success = signal_generator.ml_optimizer.train_model(trades_with_outcomes)
                if success:
                    logger.info("ML model training completed successfully")
                else:
                    logger.warning("ML model training failed or skipped due to insufficient data")
            else:
                logger.info(f"Need {20 - len(trades_with_outcomes)} more trades with outcomes before ML training")
                
        except Exception as e:
            logger.error(f"Error evaluating trade outcomes: {e}")
        
        # Check more frequently initially, then less frequently
        if len(signal_generator.data_manager.load_trade_history(limit=200)) < 50:
            time.sleep(60)  # Check every minute if we have few trades
        else:
            time.sleep(300)  # Check every 5 minutes if we have enough trades

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signals')
def get_signals():
    timeframe = request.args.get('timeframe', '1m')  # Default to 1m for fast trading
    min_confidence = float(request.args.get('min_confidence', 60))  # Lower threshold
    signal_filter = request.args.get('filter', 'all')
    market_filter = request.args.get('market', 'all')
    
    with update_lock:
        filtered_signals = []
        
        for key, signal in current_signals.items():
            if (signal.timeframe == timeframe and 
                signal.confidence >= min_confidence and
                (signal_filter == 'all' or signal.signal_type.lower() == signal_filter.lower()) and
                (market_filter == 'all' or signal.market_type == market_filter)):
                
                filtered_signals.append({
                    'signal': signal.signal_type,
                    'asset': format_symbol(signal.symbol),
                    'market': signal.market_type.title(),
                    'timeframe': signal.timeframe,
                    'contract_period': signal.contract_period,
                    'entry_zone': f"{signal.entry_low:.4f} - {signal.entry_high:.4f}",
                    'target': f"{signal.target:.4f}",
                    'stop_loss': f"{signal.stop_loss:.4f}",
                    'confidence': f"{signal.confidence:.0f}%",
                    'reasoning': signal.reasoning,
                    'remaining_time': f"{signal.remaining_seconds}s",
                    'risk_reward': f"{signal.risk_reward_ratio:.1f}:1"
                })
    
    # Sort by confidence (highest first)
    filtered_signals.sort(key=lambda x: float(x['confidence'].rstrip('%')), reverse=True)
    
    logger.info(f"Returning {len(filtered_signals)} signals for {timeframe} timeframe")
    return jsonify(filtered_signals)

@app.route('/api/select-assets', methods=['POST'])
def select_assets():
    try:
        data = request.get_json()
        assets = data.get('assets', [])
        signal_generator.set_selected_assets(assets)
        return jsonify({'status': 'success', 'message': f'Selected {len(assets)} assets'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/asset-list')
def get_asset_list():
    return jsonify(TOP_ASSETS)

@app.route('/api/trade-history')
def get_trade_history():
    limit = int(request.args.get('limit', 50))
    trades = signal_generator.data_manager.load_trade_history(limit=limit)
    return jsonify(trades)

@app.route('/api/ml-status')
def get_ml_status():
    return jsonify({
        'is_trained': signal_generator.ml_optimizer.is_trained,
        'training_data_path': signal_generator.ml_optimizer.training_data_path,
        'model_path': signal_generator.ml_optimizer.model_path
    })

def format_symbol(symbol: str) -> str:
    if symbol.endswith("=X"):
        base = symbol.replace("=X", "")
        if "USD" in base:
            return f"{base.replace('USD', '')}/USD"
        return base
    elif symbol.endswith("-USD"):
        return symbol.replace("-USD", "/USD")
    return symbol

@app.route('/api/status')
def get_status():
    with update_lock:
        signal_breakdown = {}
        for signal in current_signals.values():
            market = signal.market_type
            if market not in signal_breakdown:
                signal_breakdown[market] = {'BUY': 0, 'SELL': 0, 'WAIT': 0, 'total': 0}
            signal_breakdown[market][signal.signal_type] += 1
            signal_breakdown[market]['total'] += 1
            
        return jsonify({
            'status': 'online',
            'total_signals': len(current_signals),
            'markets': signal_breakdown,
            'last_update': datetime.now().isoformat()
        })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    # Start background signal updater
    update_thread = threading.Thread(target=update_signals, daemon=True)
    update_thread.start()
    
    # Start background outcome evaluator
    outcome_thread = threading.Thread(target=evaluate_trade_outcomes, daemon=True)
    outcome_thread.start()
    
    print("🚀 Starting FAST TRADING Bot (1-Minute Focus)...")
    print("📈 Generating rapid signals for fast execution...")
    print("⚡ Optimized for 1-minute timeframe trading...")
    print("🤖 Machine learning adapting to fast market conditions...")
    
    time.sleep(2)
    
    app.run(debug=False, host='0.0.0.0', port=5000)