import MetaTrader5 as mt5
import time
import random
from datetime import datetime
import os

# ===================== CONFIGURATION =====================

# List of symbols to consider.
SYMBOLS = [
    "EURUSD.r",
    "AUDUSD.r",
    "AUDCAD.r",
    "AUDCHF.r",
    "EURZAR.r",
    "XAGUSD.r"  # Silver
]

LOT_SIZE = 0.01            # Fixed lot size for all trades.
DEVIATION = 20             # Maximum allowed price deviation.
MAGIC_NUMBER = 234000      # Identifier for our trades.
COMMENT = "Python scalping bot order"

# Profit & Loss settings (in ZAR)
MIN_TRADE_PROFIT = 5       # Minimum profit target per trade (scalp) in ZAR
MAX_TRADE_PROFIT = 20      # Maximum profit target per trade (scalp) in ZAR
GLOBAL_WIN_TARGET = 50     # Close trade if profit reaches +50 ZAR.
GLOBAL_LOSS_LIMIT = -100   # Close trade if loss reaches -100 ZAR.
# For 0.01 lots on most pairs, ~1 pip (0.0001) ≈ 2 ZAR.
ZAR_PER_PIP = 2.0          # Conversion factor

# Logging folder (make sure it exists or will be created)
LOG_FOLDER = r"C:\Users\Thape.THAPELOMOROLONG\Downloads\Text file for bot"

# Polling interval for trade monitoring (in seconds)
MIN_POLL_INTERVAL = 60
MAX_POLL_INTERVAL = 120

# ===================== UTILITY FUNCTIONS =====================

def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    print("MT5 Initialized successfully.")

def shutdown_mt5():
    mt5.shutdown()
    print("MT5 connection closed.")

def check_and_select_symbol(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol '{symbol}' not found.")
        return None
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol '{symbol}'.")
            return None
    return symbol_info

def get_spread(symbol):
    """Return the spread (ask - bid) for the given symbol."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return tick.ask - tick.bid

def silver_trade_confidence():
    """
    Dummy function to simulate checking if silver (XAGUSD.r)
    has at least an 85% chance of making a profit within 1 hour.
    Replace this with your actual prediction logic.
    """
    confidence = random.uniform(0, 1)
    print(f"Silver trade confidence: {confidence:.2f}")
    return confidence >= 0.85

def score_symbol(symbol):
    """
    For demonstration, score the symbol based on its spread.
    (Lower spread is considered better.)
    """
    info = check_and_select_symbol(symbol)
    if info is None:
        return float('inf')
    spread = get_spread(symbol)
    if spread is None:
        return float('inf')
    return spread

def decide_trade_direction(symbol):
    """
    For demonstration, randomly decide BUY or SELL.
    You can incorporate technical indicators here.
    """
    direction = random.choice(["BUY", "SELL"])
    print(f"Decided trade direction for {symbol}: {direction}")
    return direction

def calculate_price_offset(zar_amount):
    """
    Convert a ZAR risk/profit target into a price offset.
    For 0.01 lots: 1 pip (0.0001) ≈ ZAR_PER_PIP, so:
        price offset = (zar_amount / ZAR_PER_PIP) * 0.0001.
    """
    return (zar_amount / ZAR_PER_PIP) * 0.0001

def get_trade_log_filename(trade_count):
    """Return a filename for the current trade log based on trade count."""
    os.makedirs(LOG_FOLDER, exist_ok=True)
    filename = os.path.join(LOG_FOLDER, f"Trade{trade_count}.txt")
    return filename

def append_to_master_log(text):
    """Append a line to the master log file."""
    master_log = os.path.join(LOG_FOLDER, "Master_Trade_Log.txt")
    with open(master_log, "a") as f:
        f.write(text + "\n")

# ===================== TRADE FUNCTIONS =====================

def open_trade(symbol, direction, profit_target_zar, loss_limit_zar):
    """
    Open a trade on a given symbol with a specified direction.
    profit_target_zar: desired profit target (for TP) in ZAR.
    loss_limit_zar: maximum loss (for SL) in ZAR (should be a negative value).
    Enforces that abs(loss_limit_zar) does not exceed 4× profit_target_zar.
    """
    # Enforce stop loss constraint: SL <= 4 * TP
    max_allowed_loss = 4 * profit_target_zar
    if abs(loss_limit_zar) > max_allowed_loss:
        print(f"Adjusting loss limit from {loss_limit_zar} ZAR to {-max_allowed_loss} ZAR to meet the 4x constraint.")
        loss_limit_zar = -max_allowed_loss

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for {symbol}")
        return None, None

    is_buy = (direction == "BUY")
    entry_price = tick.ask if is_buy else tick.bid

    # Calculate TP and SL price offsets.
    tp_offset = calculate_price_offset(profit_target_zar)
    sl_offset = calculate_price_offset(abs(loss_limit_zar))  # loss_limit_zar is negative

    if is_buy:
        tp_price = entry_price + tp_offset
        sl_price = entry_price - sl_offset
    else:
        tp_price = entry_price - tp_offset
        sl_price = entry_price + sl_offset

    order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": entry_price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"Sending order for {symbol}: {request}")
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"OrderSend failed for {symbol}, retcode={result.retcode}")
        return None, None

    print(f"Trade opened on {symbol}. Order ID: {result.order}")
    trade_details = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "profit_target_zar": profit_target_zar,
        "loss_limit_zar": loss_limit_zar,
    }
    return result.order, trade_details

def modify_trade_sl(order_ticket, symbol, new_sl):
    """
    Attempt to modify the stop loss of an open trade.
    If the modification fails with retcode 10036 (or any other error),
    log the error.
    """
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "order": order_ticket,
        "symbol": symbol,
        "sl": new_sl,
        "tp": 0.0,  # TP remains unchanged
        "magic": MAGIC_NUMBER,
        "comment": "Modify SL",
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Modified SL for order {order_ticket} to {new_sl}")
    else:
        print(f"Failed to modify SL for order {order_ticket}, retcode={result.retcode}")
    return result.retcode

def close_trade(order_ticket, symbol, direction):
    """
    Close an open trade by sending an opposite order.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for closing {symbol}")
        return False

    is_buy = (direction == "BUY")
    close_price = tick.bid if is_buy else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": order_ticket,
        "symbol": symbol,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": close_price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "Close trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Successfully closed trade {order_ticket} on {symbol}")
        return True
    else:
        print(f"Failed to close trade {order_ticket} on {symbol}, retcode={result.retcode}")
        return False

def monitor_trade(order_ticket, trade_details):
    """
    Monitor an open trade:
      - Checks every 60-120 seconds.
      - Closes the trade if profit >= GLOBAL_WIN_TARGET or loss <= GLOBAL_LOSS_LIMIT.
      - Attempts to adjust SL in a trailing manner when in profit.
    Returns the trade's profit (or loss) once closed.
    """
    symbol = trade_details["symbol"]
    direction = trade_details["direction"]
    trade_open_time = datetime.now()

    while True:
        sleep_interval = random.uniform(MIN_POLL_INTERVAL, MAX_POLL_INTERVAL)
        time.sleep(sleep_interval)

        positions = mt5.positions_get(ticket=order_ticket)
        if positions is None or len(positions) == 0:
            # Trade is closed
            break

        pos = positions[0]
        current_profit = pos.profit  # Profit in account currency (ZAR)
        print(f"Monitoring trade {order_ticket} on {symbol}: Current profit {current_profit:.2f} ZAR")

        if current_profit >= GLOBAL_WIN_TARGET or current_profit <= GLOBAL_LOSS_LIMIT:
            print(f"Trade {order_ticket} reached exit criteria: {current_profit:.2f} ZAR")
            close_trade(order_ticket, symbol, direction)
            break
        else:
            # Adjust SL if in profit to lock gains.
            if current_profit > 0:
                adjustment = calculate_price_offset(current_profit * 0.5)
                tick = mt5.symbol_info_tick(symbol)
                if direction == "BUY":
                    new_sl = tick.bid - adjustment
                else:
                    new_sl = tick.ask + adjustment
                retcode = modify_trade_sl(order_ticket, symbol, new_sl)
                if retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Warning: SL modification error for order {order_ticket}, retcode={retcode}")
    # After closure, compute profit from history.
    now = datetime.now()
    deals = mt5.history_deals_get(trade_open_time, now)
    trade_profit = 0.0
    if deals:
        for deal in deals:
            if deal.order == order_ticket and deal.magic == MAGIC_NUMBER:
                trade_profit += deal.profit
    return trade_profit

# ===================== MAIN LOGIC =====================

def main():
    initialize_mt5()

    # Validate symbols.
    valid_symbols = []
    for sym in SYMBOLS:
        info = check_and_select_symbol(sym)
        if info is not None:
            valid_symbols.append(sym)
    if not valid_symbols:
        print("No valid symbols found. Exiting.")
        shutdown_mt5()
        return

    # Score symbols based on spread.
    symbol_scores = {sym: score_symbol(sym) for sym in valid_symbols}
    print("Symbol scores (lower is better):", symbol_scores)

    # For silver, check additional confidence.
    if "XAGUSD.r" in symbol_scores:
        if not silver_trade_confidence():
            print("Skipping silver due to low confidence.")
            symbol_scores.pop("XAGUSD.r")

    # Select the two symbols with the best scores.
    selected_symbols = sorted(symbol_scores, key=symbol_scores.get)[:2]
    print("Selected symbols for trading:", selected_symbols)

    # Create a master log header.
    master_log_header = f"--- Trading Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    append_to_master_log(master_log_header)
    
    overall_session_profit = 0.0
    trade_count = 0

    # Trade on each selected symbol until exit criteria is met per symbol.
    for symbol in selected_symbols:
        symbol_session_profit = 0.0
        print(f"\nStarting trading on {symbol}")
        # Open a new log file for this symbol's session.
        symbol_log_filename = os.path.join(LOG_FOLDER, f"{symbol.replace('.r','')}_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(symbol_log_filename, "w") as f:
            f.write(f"Trade Log for {symbol} session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        while symbol_session_profit < GLOBAL_WIN_TARGET and symbol_session_profit > GLOBAL_LOSS_LIMIT:
            trade_count += 1
            print(f"\n--- Trade #{trade_count} on {symbol} ---")
            # Decide trade direction.
            direction = decide_trade_direction(symbol)
            # Choose a random profit target between MIN_TRADE_PROFIT and MAX_TRADE_PROFIT.
            profit_target = random.uniform(MIN_TRADE_PROFIT, MAX_TRADE_PROFIT)
            loss_limit = GLOBAL_LOSS_LIMIT  # Fixed loss limit (will be adjusted in open_trade if necessary)

            order_ticket, trade_details = open_trade(symbol, direction, profit_target, loss_limit)
            if order_ticket is None:
                print("Trade failed to open. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            # Monitor the trade.
            trade_profit = monitor_trade(order_ticket, trade_details)
            symbol_session_profit += trade_profit
            overall_session_profit += trade_profit

            # Build a log line for this trade.
            trade_log_filename = get_trade_log_filename(trade_count)
            log_line = (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Symbol: {symbol}, Direction: {direction}, "
                f"Entry: {trade_details['entry_price']:.5f}, TP: {trade_details['tp_price']:.5f}, "
                f"SL: {trade_details['sl_price']:.5f}, Target: {profit_target:.2f} ZAR, "
                f"Trade Profit: {trade_profit:.2f} ZAR, "
                f"Session Profit for {symbol}: {symbol_session_profit:.2f} ZAR"
            )
            with open(trade_log_filename, "w") as f:
                f.write(log_line + "\n")
            append_to_master_log(log_line)
            print(log_line)
            time.sleep(2)

            # Stop trading on this symbol if limits reached.
            if symbol_session_profit >= GLOBAL_WIN_TARGET:
                print(f"{symbol}: Session profit reached +{symbol_session_profit:.2f} ZAR. Stopping trading on {symbol}.")
                break
            if symbol_session_profit <= GLOBAL_LOSS_LIMIT:
                print(f"{symbol}: Session loss reached {symbol_session_profit:.2f} ZAR. Stopping trading on {symbol}.")
                break

    session_summary = (
        f"--- Trading Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        f"Overall Session Profit: {overall_session_profit:.2f} ZAR across {trade_count} trades."
    )
    append_to_master_log(session_summary)
    print(session_summary)

    shutdown_mt5()

if __name__ == "__main__":
    main()
