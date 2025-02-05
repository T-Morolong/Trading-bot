import os
import sys
import time
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

# ---------------- MetaTrader5 Imports ----------------
try:
    import MetaTrader5 as mt5
except ModuleNotFoundError:
    print("Error: The 'MetaTrader5' module is not installed.")
    print("Please install it by running: pip install MetaTrader5")
    sys.exit(1)

# ---------------- PyQt5 Imports ----------------
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QLabel, QGridLayout, QMessageBox, QDialog
)
from PyQt5.QtCore import QTimer

# ===================== 1. Enhanced Error Handling & Resilience =====================

def check_mt5_connection(max_retries=3):
    """Try to initialize MT5 connection with automatic retries."""
    for attempt in range(max_retries):
        if mt5.initialize():
            print("MT5 connection established.")
            return True
        print(f"Connection attempt {attempt+1} failed")
        time.sleep(2)
    raise ConnectionError("Failed to connect to MT5 after multiple attempts")

def mt5_operation(func):
    """Decorator to ensure MT5 is connected before an operation."""
    def wrapper(*args, **kwargs):
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None or not terminal_info.connected:
                check_mt5_connection()
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Operation failed: {str(e)}")
            mt5.shutdown()
            check_mt5_connection()
            return func(*args, **kwargs)  # Retry after reconnection
    return wrapper

# ===================== Data Caching for Performance =====================

class DataCache:
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl

    def get(self, symbol, timeframe):
        key = f"{symbol}_{timeframe}"
        if key in self.cache and time.time() - self.cache[key]['timestamp'] < self.ttl:
            return self.cache[key]['data']
        return None

    def set(self, symbol, timeframe, data):
        key = f"{symbol}_{timeframe}"
        self.cache[key] = {
            'timestamp': time.time(),
            'data': data
        }

data_cache = DataCache(ttl=60)

# ===================== UTILITY FUNCTIONS =====================

def is_forex_symbol(sym_name):
    """Return True if sym_name is a 6-letter Forex pair."""
    if sym_name.endswith(".r"):
        sym_name = sym_name[:-2]
    return len(sym_name) == 6 and sym_name.isalpha()

@mt5_operation
def get_historical_data(symbol, timeframe=mt5.TIMEFRAME_M30, hours=4):
    """
    Retrieve historical data for the given symbol for the past 'hours' hours.
    Data is returned as a list of bars.
    Uses caching to improve performance.
    """
    cached = data_cache.get(symbol, timeframe)
    if cached:
        return cached

    utc_from = datetime.now(timezone.utc) - timedelta(hours=hours)
    utc_to = datetime.now(timezone.utc)
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None:
        return []
    bars = list(rates)
    data_cache.set(symbol, timeframe, bars)
    return bars

def get_multi_timeframe_data(symbol):
    """Return historical data for multiple timeframes."""
    return {
        'M15': get_historical_data(symbol, mt5.TIMEFRAME_M15, 24),
        'H1': get_historical_data(symbol, mt5.TIMEFRAME_H1, 48),
        'H4': get_historical_data(symbol, mt5.TIMEFRAME_H4, 96)
    }

def calculate_rsi(closes, period=14):
    """Calculate a simple RSI indicator for a list of closing prices."""
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    avg_gain = statistics.mean(gains[-period:])
    avg_loss = statistics.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(bars, period=14):
    """Calculate the Average True Range (ATR) from historical bars."""
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        high = bars[i]['high']
        low = bars[i]['low']
        prev_close = bars[i - 1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr = statistics.mean(trs[-period:])
    return atr

def calculate_technical_indicators(bars):
    """Calculate technical indicators such as SMA20, RSI and ATR."""
    closes = [bar['close'] for bar in bars]
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    rsi = calculate_rsi(closes)
    atr = calculate_atr(bars)
    return {
        'sma20': sma20,
        'rsi': rsi,
        'atr': atr
    }

def analyze_history(symbol):
    """
    Download 4 hours of historical data (30-min bars) and compute the average spread
    and standard deviation (volatility) of the spread.
    Returns a dict with keys: avg_spread, spread_volatility, count.
    """
    bars = get_historical_data(symbol)
    if not bars:
        return {"avg_spread": None, "spread_volatility": None, "count": 0}
    # Directly index the 'spread' field to avoid issues with numpy.void objects.
    spreads = [float(bar['spread']) for bar in bars if bar['spread'] is not None]
    if spreads:
        avg_spread = statistics.mean(spreads)
        spread_volatility = statistics.stdev(spreads) if len(spreads) > 1 else 0
    else:
        avg_spread, spread_volatility = None, None
    return {"avg_spread": avg_spread, "spread_volatility": spread_volatility, "count": len(bars)}

# ===================== Risk Management Functions =====================

def calculate_position_size(symbol, recommended_sl_price, risk_percent=2):
    """
    Calculate dynamic position size based on account balance and risk.
    recommended_sl_price is the stop loss price recommended.
    """
    acc = mt5.account_info()
    if not acc:
        return 0.01  # default minimal lot size

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return 0.01
    price = (tick.ask + tick.bid) / 2
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return 0.01
    point_value = symbol_info.point

    risk_amount = acc.balance * (risk_percent / 100)
    sl_points = abs(price - recommended_sl_price) / point_value
    if sl_points == 0:
        return 0.01
    position_size = risk_amount / (sl_points * point_value)
    return round(position_size, 2)

def dynamic_sl_adjustment(symbol, base_sl):
    """
    Adjust stop loss dynamically based on volatility measured via ATR (from 1-hour bars).
    """
    bars = get_historical_data(symbol, mt5.TIMEFRAME_H1, 24)
    atr = calculate_atr(bars)
    if atr is None:
        return base_sl
    return base_sl * (1 + atr * 0.5)

# ===================== 2. Enhanced Plugin System Features =====================

class TradingStrategyPlugin:
    """
    Base class for trading strategy plugins.
    Each plugin must implement scan_chart(market_data), get_settings, and update_settings.
    """
    def __init__(self):
        self.name = self.__class__.__name__

    def scan_chart(self, market_data):
        """
        Analyze market_data (a dict of live symbol data) along with historical context,
        and return advice as a dict.
        """
        raise NotImplementedError("scan_chart must be implemented by the plugin.")

    def get_settings(self):
        """Return current plugin settings as a dict."""
        return {}

    def update_settings(self, settings):
        """Update plugin settings."""
        pass

class PluginManager:
    """
    Loads plugins from a designated folder, validates them, and manages their execution.
    Now with configuration and enable/disable features.
    """
    def __init__(self, plugin_folder="plugins"):
        self.plugin_folder = plugin_folder
        self.plugin_config = self.load_plugin_config()
        self.enabled_plugins = {}  # plugin name => bool
        self.plugins = []
        self.load_plugins()
    
    def load_plugin_config(self):
        config_path = os.path.join(self.plugin_folder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return {}

    def validate_plugin(self, plugin):
        required_methods = ['scan_chart', 'get_settings', 'update_settings']
        return all(hasattr(plugin, method) for method in required_methods)

    def load_plugins(self):
        if not os.path.exists(self.plugin_folder):
            print(f"Plugin folder '{self.plugin_folder}' not found. Creating it.")
            os.makedirs(self.plugin_folder)
        for filename in os.listdir(self.plugin_folder):
            if filename.endswith(".py") and filename != "__init__.py":
                filepath = os.path.join(self.plugin_folder, filename)
                plugin = self.load_plugin_from_file(filepath)
                if plugin and self.validate_plugin(plugin):
                    # Check config to see if plugin is enabled (default True)
                    enabled = self.plugin_config.get(plugin.name, {}).get("enabled", True)
                    self.enabled_plugins[plugin.name] = enabled
                    if enabled:
                        self.plugins.append(plugin)
                        print(f"Loaded plugin: {plugin.name}")
                    else:
                        print(f"Plugin {plugin.name} is disabled via configuration.")
    
    def load_plugin_from_file(self, filepath):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = __import__('importlib.util').util.spec_from_file_location(module_name, filepath)
        if spec and spec.loader:
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if (isinstance(attribute, type) and 
                    issubclass(attribute, TradingStrategyPlugin) and 
                    attribute is not TradingStrategyPlugin):
                    return attribute()
        return None

    def run_plugin(self, plugin, market_data):
        start_time = time.time()
        try:
            advice = plugin.scan_chart(market_data)
            elapsed = time.time() - start_time
            advice['plugin'] = plugin.name
            advice['scan_time'] = elapsed
            print(f"[{plugin.name}] scan_chart completed in {elapsed:.3f} seconds.")
            return advice
        except Exception as e:
            print(f"Error in plugin {plugin.name}: {e}")
            return {"plugin": plugin.name, "scan_time": None, "message": f"Error: {e}", "advice": {}}

    def run_all_plugins(self, market_data):
        advices = []
        # Run plugins in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.run_plugin, plugin, market_data) for plugin in self.plugins]
            for future in futures:
                advices.append(future.result())
        return advices

# ===================== Plugin Settings GUI =====================

class PluginSettingsDialog(QDialog):
    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"{self.plugin.name} Settings")
        layout = QVBoxLayout()
        # Dynamically add controls based on plugin.get_settings()
        settings = self.plugin.get_settings()
        for key, value in settings.items():
            label = QLabel(f"{key}: {value}")
            layout.addWidget(label)
        self.setLayout(layout)

# ===================== 3. Comprehensive Strategy Plugin with Enhanced Historical Analysis =====================

class ComprehensiveStrategyScanner(TradingStrategyPlugin):
    """
    This plugin scans the market for recommended symbols, downloads historical data,
    and chooses a trading strategy for each selected symbol. It computes recommended
    take profit (TP) and stop loss (SL) values (in ZAR) dynamically based on the account balance.
    """
    def __init__(self):
        super().__init__()
        self.strategists = {
            "AggressiveBuy": {
                "risk_multiplier": 2.0,
                "recommended_position": "BUY",
                "description": "Low spread & low volatility; aggressive long position."
            },
            "ConservativeBuy": {
                "risk_multiplier": 1.0,
                "recommended_position": "BUY",
                "description": "Moderate conditions; cautious long position."
            },
            "AggressiveSell": {
                "risk_multiplier": 2.0,
                "recommended_position": "SELL",
                "description": "High volatility with widening spreads; aggressive short position."
            },
            "ConservativeSell": {
                "risk_multiplier": 1.0,
                "recommended_position": "SELL",
                "description": "Stable conditions with moderate spreads; conservative short position."
            },
        }
        self.settings = {"sample_setting": True}
    
    def get_settings(self):
        return self.settings

    def update_settings(self, settings):
        self.settings.update(settings)

    def choose_strategy(self, hist_stats):
        """Choose a strategy based on average spread and volatility."""
        avg = hist_stats.get("avg_spread")
        vol = hist_stats.get("spread_volatility")
        if avg is None or vol is None:
            return self.strategists["ConservativeBuy"]
        if avg < 3 and vol < 1:
            return self.strategists["AggressiveBuy"]
        elif avg < 5:
            return self.strategists["ConservativeBuy"]
        elif avg >= 5 and vol >= 2:
            return self.strategists["AggressiveSell"]
        else:
            return self.strategists["ConservativeSell"]

    def scan_chart(self, market_data):
        """
        For each symbol in market_data, download historical data and compute stats.
        Then choose the two symbols with the lowest average spread and assign a strategy.
        Also, determine recommended TP/SL based on account balance using a dynamic formula.
        """
        symbol_analysis = {}
        for symbol in market_data.keys():
            stats = analyze_history(symbol)
            if stats["count"] > 0 and stats["avg_spread"] is not None:
                symbol_analysis[symbol] = stats

        if not symbol_analysis:
            return {"symbols": [], "advice": {}, "message": "Insufficient historical data."}

        sorted_symbols = sorted(symbol_analysis.items(), key=lambda x: x[1]["avg_spread"])
        chosen = sorted_symbols[:2]

        acc_info = mt5.account_info()
        balance = acc_info.balance if acc_info is not None else 0

        advice_dict = {}
        for symbol, stats in chosen:
            strategy = self.choose_strategy(stats)
            # Dynamically calculate TP/SL based on account balance and market conditions.
            tp_sl = get_tp_sl_recommendation(symbol, balance)
            adjusted_sl = dynamic_sl_adjustment(symbol, tp_sl["sl"])
            advice_dict[symbol] = {
                "recommended_action": strategy["recommended_position"],
                "risk_multiplier": strategy["risk_multiplier"],
                "strategy_description": strategy["description"],
                "recommended_tp": tp_sl["tp"],
                "recommended_sl": adjusted_sl,
                "historical_stats": stats
            }
        message = "Strategy advice based on 4-hour historical analysis (30-min bars) and account balance."
        return {"symbols": [s for s, _ in chosen], "advice": advice_dict, "message": message}

# ===================== Dynamic TP/SL Recommendation =====================

def get_tp_sl_recommendation(symbol, balance, risk_percent=0.01, risk_reward_ratio=1.5):
    """
    Dynamically calculate recommended take profit (TP) and stop loss (SL) values (in ZAR)
    based on the account balance and market volatility.
    
    - risk_percent: The fraction of the account balance to risk on the trade (default 1%).
    - risk_reward_ratio: The ratio of TP to SL (default 1.5, meaning TP is 1.5 times SL).
    
    The formula used is:
       risk_amount = balance * risk_percent (minimum ZAR 10)
       recommended SL = risk_amount * (1 + avg_spread/50)
       recommended TP = recommended SL * risk_reward_ratio
    
    This method applies to all forex pairs.
    """
    risk_amount = balance * risk_percent
    if risk_amount < 10:
        risk_amount = 10
    hist = analyze_history(symbol)
    avg_spread = hist.get("avg_spread")
    if avg_spread is None or avg_spread == 0:
        avg_spread = 1
    recommended_sl = risk_amount * (1 + avg_spread / 50)
    recommended_tp = recommended_sl * risk_reward_ratio
    return {"tp": recommended_tp, "sl": recommended_sl}

# ===================== MetaTrader5 Data Functions =====================

@mt5_operation
def initialize_mt5_connection():
    if not mt5.initialize():
        print("Failed to initialize MetaTrader5. Exiting.")
        sys.exit(1)
    print("MetaTrader5 initialized successfully.")

@mt5_operation
def shutdown_mt5_connection():
    mt5.shutdown()
    print("MetaTrader5 connection closed.")

@mt5_operation
def get_live_market_data():
    """
    Retrieves live market data for all tradable Forex symbols.
    Returns a dict: {symbol: {"bid": float, "ask": float}, ...}
    """
    market_data = {}
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        print("No symbols retrieved from MT5.")
        return market_data
    for sym in all_symbols:
        if sym.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED and sym.visible:
            if is_forex_symbol(sym.name):
                tick = mt5.symbol_info_tick(sym.name)
                if tick:
                    market_data[sym.name] = {"bid": tick.bid, "ask": tick.ask}
    return market_data

# ===================== 5. GUI Enhancements =====================

class StrategyAdviceWidget(QWidget):
    """
    A widget to display individual strategy advice with visual elements.
    """
    def __init__(self, advice):
        super().__init__()
        self.init_ui(advice)
        
    def init_ui(self, advice):
        layout = QGridLayout()
        self.symbol_label = QLabel(advice.get('symbol', 'N/A'))
        self.action_label = QLabel(advice.get('recommended_action', 'N/A'))
        # Color coding: green for BUY, red for SELL.
        if advice.get('recommended_action') == "BUY":
            self.action_label.setStyleSheet("color: white; background-color: green")
        else:
            self.action_label.setStyleSheet("color: white; background-color: red")
        self.sparkline = QLabel("Sparkline: (not implemented)")
        layout.addWidget(self.symbol_label, 0, 0)
        layout.addWidget(self.action_label, 0, 1)
        layout.addWidget(self.sparkline, 1, 0, 1, 2)
        self.setLayout(layout)
        
    def draw_sparkline(self, prices):
        # Implement a simple ASCII sparkline or integrate a plotting library.
        pass

class AdviceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Advice Plugin")
        self.resize(800, 700)
        self.init_ui()
        self.plugin_manager = PluginManager(plugin_folder="plugins")
        # If no external plugins are found, add the inline comprehensive plugin.
        if not self.plugin_manager.plugins:
            self.plugin_manager.plugins.append(ComprehensiveStrategyScanner())
        self.refresh_advice()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        self.status_label = QLabel("Press 'Refresh Advice' to scan.")
        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.advice_text)
        self.refresh_button = QPushButton("Refresh Advice")
        self.refresh_button.clicked.connect(self.refresh_advice)
        layout.addWidget(self.refresh_button)
        central_widget.setLayout(layout)

    def refresh_advice(self):
        self.status_label.setText("Scanning market data...")
        QApplication.processEvents()  # Force UI update
        market_data = get_live_market_data()
        if not market_data:
            QMessageBox.warning(self, "No Data", "No market data available from MT5.")
            return
        advices = self.plugin_manager.run_all_plugins(market_data)
        report_lines = []
        for advice in advices:
            report_lines.append(f"Plugin: {advice.get('plugin')}")
            scan_time = advice.get('scan_time')
            if scan_time is not None:
                report_lines.append(f"Scan Time: {scan_time:.3f} seconds")
            else:
                report_lines.append("Scan Time: N/A")
            report_lines.append(f"Message: {advice.get('message')}")
            recommended = advice.get("advice", {})
            for symbol, rec in recommended.items():
                stats = rec.get("historical_stats", {})
                report_lines.append(f"Symbol: {symbol}")
                report_lines.append(f"  Recommended Action: {rec.get('recommended_action')}")
                report_lines.append(f"  Risk Multiplier: {rec.get('risk_multiplier')}")
                report_lines.append(f"  Strategy: {rec.get('strategy_description')}")
                report_lines.append(f"  Recommended TP (ZAR): {rec.get('recommended_tp')}")
                report_lines.append(f"  Recommended SL (ZAR): {rec.get('recommended_sl')}")
                if stats.get("avg_spread") is not None and stats.get("spread_volatility") is not None:
                    report_lines.append(f"  Historical Avg Spread: {stats.get('avg_spread'):.2f} | Volatility: {stats.get('spread_volatility'):.2f}")
                else:
                    report_lines.append("  Historical data not available.")
            report_lines.append(f"Chosen Symbols: {advice.get('symbols')}")
            report_lines.append("-" * 60)
        self.advice_text.setPlainText("\n".join(report_lines))
        self.status_label.setText("Advice refreshed.")

# ===================== 7. Enhanced Backtesting Integration =====================

class Backtester:
    """
    A simple backtester that runs a strategy on historical data.
    """
    def __init__(self, strategy):
        self.strategy = strategy
        self.historical_data = {}

    def load_data(self, symbol, timeframe, days):
        # Load historical data for a number of days.
        self.historical_data[symbol] = get_historical_data(symbol, timeframe, hours=days*24)

    def run_test(self, initial_balance=10000):
        results = {
            'balance': initial_balance,
            'trades': [],
            'max_drawdown': 0
        }
        # A very basic simulation loop over the historical data.
        for symbol, bars in self.historical_data.items():
            for bar in bars:
                advice = self.strategy.scan_chart({symbol: {"bid": bar['close'], "ask": bar['close']}})
                results['trades'].append({
                    'symbol': symbol,
                    'bar_time': bar['time'],
                    'advice': advice
                })
        return results

# ===================== MAIN APPLICATION CODE =====================

def main():
    initialize_mt5_connection()
    app = QApplication(sys.argv)
    window = AdviceWindow()
    window.show()
    app.exec_()
    shutdown_mt5_connection()

if __name__ == "__main__":
    main()
