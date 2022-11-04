import math
import time
import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.client import Client

# All necessary plotly libraries
import plotly as plotly
import plotly.io as plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from BinanceBot.reinforcement_learning_stock import StockTradingEnv

api_key = 'hPZcl6C7b1NFfMNc2whnh2OSyxsWxyEIEP2mKDypUYDrG70eeB4rniELcx0KnAwD'
api_secret = 'tF6fS7kskVztUbDA5jkXt8GAdXLoJ82C9BaspK0F9AbV2MuG2aGZOcLoboOOUgX6'

col = ['open_time', 'open', 'high', 'low', 'close',
       'volume', 'close_time', 'quote_asset_volume',
       'number_of_trades', 'taker_buy_base_asset_volume',
       'taker_buy_quote_asset_volume', 'ignore']

client = Client(api_key, api_secret, testnet=True)


def preprossing(df, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14, difference=1e-4):
    df = df.copy(deep=True)

    df['sma_short'] = df.ta.sma(length=sma_short)
    df['sma_long'] = df.ta.sma(length=sma_long)

    df['ema_short'] = df.ta.ema(length=ema_short)
    df['ema_long'] = df.ta.ema(length=ema_long)

    df['rsi'] = df.ta.rsi(length=rsi_period)

    df['single_rsi'] = df['rsi'].apply(lambda x: 1 if x < 30 else 0 if x >= 70 else -1)

    df['single_sma'] = ((df['sma_short'] - difference) > df['sma_long']).astype(int)
    df['single_ema'] = ((df['ema_short']) > df['ema_long']).astype(int)

    df['double_sma'] = (df['sma_short'] > df['sma_long']).astype(int).diff()

    adx = df.ta.adx(length=14, high='high', low='low', close='close')
    df['single_dmp'] = (adx['DMN_14'] > adx['DMP_14']).astype(int)

    required_col = ['open', 'high', 'low', 'close', 'volume', 'sma_short',
                    'sma_long', 'ema_short', 'ema_long', 'rsi', 'single_sma',
                    'single_ema', 'double_sma', 'single_dmp', 'single_rsi']
    #     required_col = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'single_sma', 'single_ema']

    return df[required_col]


def get_data(symbol, interval, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14, limit=None):
    if sma_short > sma_long:
        print("sma_short must be smaller than sma_long")
        return None

    if ema_short > ema_long:
        print("ema_short must be smaller than ema_long")
        return None

    if rsi_period < 1:
        print("rsi_period must be greater than 1")
        return None

    limit = limit if limit is not None else sma_long if sma_long > ema_long else ema_long

    col = ['open_time', 'open', 'high', 'low', 'close',
           'volume', 'close_time', 'quote_asset_volume',
           'number_of_trades', 'taker_buy_base_asset_volume',
           'taker_buy_quote_asset_volume', 'ignore']

    df = pd.DataFrame(client.get_klines(symbol=symbol, interval=interval, limit=limit), columns=col)

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df = df.set_index('open_time')
    df = df.drop(columns=['close_time', 'ignore'])
    df = df.astype(float)

    df = preprossing(df)

    return df

def get_klines(symbol, interval, start, end,limit=1000):
    col = ['open_time', 'open', 'high', 'low', 'close',
           'volume', 'close_time', 'quote_asset_volume',
           'number_of_trades', 'taker_buy_base_asset_volume',
           'taker_buy_quote_asset_volume', 'ignore']

    d = []
    for klines in client.get_historical_klines_generator(symbol, interval, start_str=start, end_str=end, limit=limit):
        d.append(klines)
    df = pd.DataFrame(d, columns=col)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df['open'] = np.float64(df['open'])
    df['close'] = np.float64(df['close'])
    df['volume'] = np.float64(df['volume'])

    df = df.set_index('open_time')

    return df

def get_min_qty(symbol, multiplier=1):
    info = client.get_symbol_info(symbol)
    ticker = client.get_ticker(symbol=symbol)
    filters = pd.DataFrame(info.get('filters'))
    min_lot_size = float(filters[filters.filterType == 'LOT_SIZE']['minQty'].values[0])
    min_step_size = float(filters[filters.filterType == 'LOT_SIZE']['stepSize'].values[0])
    min_notional = float(filters[filters.filterType == 'MIN_NOTIONAL']['minNotional'].values[0])
    current_price = float(ticker['lastPrice'])

    qty = min_notional / current_price
    qty = math.ceil(qty)
    return qty * multiplier


def get_cash(asset):
    info = client.get_asset_balance(asset=asset)
    return float(info['free'])


def order(symbol, qty, type):
    try:
        if type.lower() == 'buy'.lower() or type.lower() == 'b'.lower():
            order = client.order_market_buy(symbol=symbol, quantity=qty)

        elif type.lower() == 'sell'.lower() or type.lower() == 's'.lower():
            order = client.order_market_sell(symbol=symbol, quantity=qty)
        else:
            print("Wrong order type")
            return None
        return order
    except Exception as e:
        print(e)
        return None


def bot_binance_signel(df, symbol, asset_1, asset_2, stop_loss, take_profit, print_action, is_bought, profit,
                       last_bought_price, last_max_value, multiplier=1):
    sma_single = df.single_sma[0]
    ema_single = df.single_ema[0]
    rsi = df.rsi[0]
    close_price = df.close[0]
    asset_1_cash = get_cash(asset_1)
    asset_2_cash = get_cash(asset_2)
    min_qty = get_min_qty(symbol,multiplier)
    # open_price = df['open'].values
    # high_price = df['high'].values
    # low_price = df['low'].values
    # volume = df['volume'].values

    if not is_bought and sma_single == 1 :  # and ema_single == 1:
        if asset_1_cash > close_price:  # check if you have money

            is_order_placed = order(symbol, min_qty, 'buy')  # place Buy Order
            if is_order_placed is not None:
                is_bought = True
                last_bought_price = close_price  # set bought price
                last_max_value = close_price  # max price since bought, used for dynamic stop loss
                if print_action:
                    print_and_log(f"""buy {close_price}, Min Qty: {min_qty} """)
                    # print("buy", close_price, "Min Qty:", min_qty)
            else:
                if print_action:
                    print_and_log(f"""buy order failed {close_price}, Min Qty: {min_qty} """)
                    # print("buy order failed",close_price, "Min Qty:", min_qty)
        else:
            if print_action:
                print_and_log(f"""not enough money to buy Close Price: {close_price}, {asset_1}: {asset_1_cash} """)
                # print("not enough cash to buy")

    elif is_bought and sma_single == 0:  # and ema_single == 0 : # normal Sell
        if asset_2_cash > min_qty:  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            if close_price < (last_max_value * (1 - stop_loss)) or close_price > (
                    last_bought_price * (1 + take_profit)):

                is_order_placed = order(symbol, min_qty, 'sell')  # place Sell Order
                if is_order_placed is not None:
                    is_bought = False
                    value = (close_price - last_bought_price)
                    profit += value

                    if print_action:
                        print_and_log(f"""sell {close_price}, Min Qty: {min_qty}, Profit: {profit} """)
                        # print("Sell", close_price, "Min Qty:", min_qty, "Profit: ", profit)
                else:
                    if print_action:
                        print_and_log(f"""sell order failed {close_price}, Min Qty: {min_qty} """)
                        # print("sell order failed",close_price, "Min Qty:", min_qty)

    elif is_bought and \
            (close_price < (last_max_value * (1 - stop_loss)) or
             close_price > (last_bought_price * (1 + take_profit))):  # Emergencey Sell Stop Loss / Take Profit

        if asset_2_cash > min_qty:  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            is_order_placed = order(symbol, min_qty, 'sell')  # place Sell Order

            if is_order_placed is not None:
                is_bought = False
                value = (close_price - last_bought_price)
                profit += value

                if print_action:
                    print_and_log(f"""sell {close_price}, Min Qty: {min_qty}, Profit: {profit} """)
                    # print("Sell", close_price, "Min Qty:", min_qty, "Profit: ", profit)
            else:
                if print_action:
                    print_and_log(f"""sell order failed {close_price}, Min Qty: {min_qty} """)
                    # print("sell order failed",close_price, "Min Qty:", min_qty)

        else:
            if print_action:
                print_and_log(f"""not enough money to buy Min QTY: {min_qty}, {asset_1}: {asset_2_cash} """)
                # print("not enough cash to sell")

    return profit, is_bought, last_max_value, last_bought_price


def bot_binance_rl(df, symbol, asset_1, asset_2, stop_loss, take_profit, print_action, is_bought, profit,
                       last_bought_price, last_max_value, multiplier=1):
    sma_single = df.single_sma[0]
    ema_single = df.single_ema[0]
    rsi = df.rsi[0]
    close_price = df.close[0]
    asset_1_cash = get_cash(asset_1)
    asset_2_cash = get_cash(asset_2)
    min_qty = get_min_qty(symbol,multiplier)




    action = 0

    if not is_bought and action == str(1) : # buy
        if asset_1_cash > close_price:  # check if you have money

            is_order_placed = order(symbol, min_qty, 'buy')  # place Buy Order
            if is_order_placed is not None:
                is_bought = True
                last_bought_price = close_price  # set bought price
                last_max_value = close_price  # max price since bought, used for dynamic stop loss
                if print_action:
                    print_and_log(f"""buy {close_price}, Min Qty: {min_qty} """)
                    # print("buy", close_price, "Min Qty:", min_qty)
            else:
                if print_action:
                    print_and_log(f"""buy order failed {close_price}, Min Qty: {min_qty} """)
                    # print("buy order failed",close_price, "Min Qty:", min_qty)
        else:
            if print_action:
                print_and_log(f"""not enough money to buy Close Price: {close_price}, {asset_1}: {asset_1_cash} """)
                # print("not enough cash to buy")

    elif is_bought and action == str(-1):  # normal Sell
        if asset_2_cash > min_qty:  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            if close_price < (last_max_value * (1 - stop_loss)) or close_price > (
                    last_bought_price * (1 + take_profit)):

                is_order_placed = order(symbol, min_qty, 'sell')  # place Sell Order
                if is_order_placed is not None:
                    is_bought = False
                    value = (close_price - last_bought_price)
                    profit += value

                    if print_action:
                        print_and_log(f"""sell {close_price}, Min Qty: {min_qty}, Profit: {profit} """)
                        # print("Sell", close_price, "Min Qty:", min_qty, "Profit: ", profit)
                else:
                    if print_action:
                        print_and_log(f"""sell order failed {close_price}, Min Qty: {min_qty} """)
                        # print("sell order failed",close_price, "Min Qty:", min_qty)

    elif is_bought and \
            (close_price < (last_max_value * (1 - stop_loss)) or
             close_price > (last_bought_price * (1 + take_profit))):  # Emergencey Sell Stop Loss / Take Profit

        if asset_2_cash > min_qty:  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            is_order_placed = order(symbol, min_qty, 'sell')  # place Sell Order

            if is_order_placed is not None:
                is_bought = False
                value = (close_price - last_bought_price)
                profit += value

                if print_action:
                    print_and_log(f"""sell {close_price}, Min Qty: {min_qty}, Profit: {profit} """)
                    # print("Sell", close_price, "Min Qty:", min_qty, "Profit: ", profit)
            else:
                if print_action:
                    print_and_log(f"""sell order failed {close_price}, Min Qty: {min_qty} """)
                    # print("sell order failed",close_price, "Min Qty:", min_qty)

        else:
            if print_action:
                print_and_log(f"""not enough money to buy Min QTY: {min_qty}, {asset_1}: {asset_2_cash} """)
                # print("not enough cash to sell")

    return profit, is_bought, last_max_value, last_bought_price


def run(symbol, asset_1, asset_2, interval='1m',
        sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14,
        stop_loss=0.001, take_profit=0.2, print_action=True, is_bought=False,
        profit=0, last_bought_price=-1, last_max_value=-1, multiplier=1, ):

    limit = sma_long if sma_long > ema_long else ema_long

    df = get_data(symbol, interval, sma_short=sma_short, sma_long=sma_long, ema_short=ema_short, ema_long=ema_long,
                  rsi_period=rsi_period)[-limit:]


    # is_load = True
    # path = 'qtable.csv'
    # env = StockTradingEnv(df, possible_values, action_space, signal_column, is_load, path)


    last_open_time = df.index[-1]
    while True:
        df = get_data(symbol, interval, sma_short=sma_short, sma_long=sma_long, ema_short=ema_short, ema_long=ema_long,
                      rsi_period=rsi_period)[-1:]

        if df.index[-1] > last_open_time:
            last_open_time = df.index[-1]
            close_price = df.close[-1]
            profit, is_bought, last_max_value, last_bought_price = bot_binance_signel(df, symbol, asset_1, asset_2, stop_loss,
                                                                                      take_profit, print_action, is_bought,
                                                                                      profit, last_bought_price,
                                                                                      last_max_value, multiplier=multiplier)

            print_and_log(f"""price: {close_price}, profit: {profit}, {asset_1}: {get_cash(asset_1)}, {asset_2}: {get_cash(asset_1)}, DateTime: {last_open_time}""")

        time.sleep(5)

def print_and_log(message):
    print(message)
    logging.info(message)


def plot_order_stocks(df, symbol, fig):
    fig = go.Figure()
    order_df = pd.DataFrame(client.get_all_orders(symbol=symbol, limit=15))
    order_df['time'] = pd.to_datetime(order_df['time'], unit='ms')
    order_df['updateTime'] = pd.to_datetime(order_df['updateTime'], unit='ms')
    order_df['price_'] = np.float64(order_df.cummulativeQuoteQty) / np.float64(order_df.origQty)
    buy_orders = order_df[order_df.side == 'BUY']
    sell_orders = order_df[order_df.side == 'SELL']

    fig.update_traces(go.Scatter(x=df.index, y=df.close, name='Close Price'))
    fig.add_trace(go.Scatter(x=buy_orders.time, y=buy_orders.price_, mode='markers', name='Buy Order'))
    fig.add_trace(go.Scatter(x=sell_orders.time, y=sell_orders.price_, mode='markers', name='Sell Order'))


def main():
    possible_values = [[1, 0],  # single_sma(1,0)
                       [1, 0],  # single_ema
                       # [1,0], # single_dmp
                       # [-1,1,0], # double_dmp
                       [-1, 1, 0],  # single_rsi
                       # [-1,1,0], # double_ema
                       # [-1,1,0], # double_sma
                       [0, 1]  # is_bought
                       ]
    action_space = [-1, 1, 0]
    # signal_column = ['single_sma','single_ema', 'single_dmp', 'double_dmp', 'single_rsi', 'double_ema', 'double_sma']
    signal_column = ['single_sma', 'single_ema', 'single_rsi']


    stop_loss = 1e-2
    take_profit = 3e-3
    multiplier = 1
    Target = "BTC"
    Base_Currency = "USDT"
    symbol = Target + Base_Currency
    # current_balance = {'TRX': get_cash('TRX'), 'USDT': get_cash('USDT')}
    current_balance = {Target: get_cash(Target), Base_Currency: get_cash(Base_Currency)}
    print(current_balance)


    run(symbol=symbol, asset_1=Target, asset_2=Base_Currency, stop_loss=stop_loss,
        take_profit=take_profit,multiplier=multiplier,env=env, is_load=is_load)



    # df = get_data('TRXUSDT', '1m', limit=1000)
    # fig = go.Figure()
    # # plot_order_stocks(df, 'TRXUSDT')
    # for i in range(10):
    #     plot_order_stocks(df, 'TRXUSDT',fig)
    #     time.sleep(2)
    # fig.show()




if __name__ == '__main__':
    logging.basicConfig(filename='logs.log', format='%(filename)s: %(message)s',level=logging.INFO)
    # get_klines('TRXUSDT', '1m', '4 day ago UTC', 'now UTC')
    main()
