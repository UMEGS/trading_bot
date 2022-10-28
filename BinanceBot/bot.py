import math
import time

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

api_key = 'hPZcl6C7b1NFfMNc2whnh2OSyxsWxyEIEP2mKDypUYDrG70eeB4rniELcx0KnAwD'
api_secret = 'tF6fS7kskVztUbDA5jkXt8GAdXLoJ82C9BaspK0F9AbV2MuG2aGZOcLoboOOUgX6'

col = ['open_time', 'open', 'high', 'low', 'close',
       'volume', 'close_time', 'quote_asset_volume',
       'number_of_trades', 'taker_buy_base_asset_volume',
       'taker_buy_quote_asset_volume', 'ignore']

client = Client(api_key, api_secret, testnet=True)


def preprossing(df, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14):
    df = df.copy(deep=True)

    df['sma_short'] = df.ta.sma(length=sma_short)
    df['sma_long'] = df.ta.sma(length=sma_long)

    df['ema_short'] = df.ta.ema(length=ema_short)
    df['ema_long'] = df.ta.ema(length=ema_long)

    df['rsi'] = df.ta.rsi(length=rsi_period)

    df['single_sma'] = (df['sma_short'] > df['sma_long']).astype(int)
    df['single_ema'] = (df['ema_short'] > df['ema_long']).astype(int)

    df['double_sma'] = (df['sma_short'] > df['sma_long']).astype(int).diff()

    # required_col = ['open', 'high', 'low', 'close', 'volume', 'sma_short', 'sma_long', 'ema_short', 'ema_long', 'rsi', 'single_sma', 'single_ema', 'double_sma']
    required_col = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'single_sma', 'single_ema']

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


def get_min_qty(symbol):
    info = client.get_symbol_info(symbol)
    ticker = client.get_ticker(symbol=symbol)
    filters = pd.DataFrame(info.get('filters'))
    min_lot_size = float(filters[filters.filterType == 'LOT_SIZE']['minQty'].values[0])
    min_step_size = float(filters[filters.filterType == 'LOT_SIZE']['stepSize'].values[0])
    min_notional = float(filters[filters.filterType == 'MIN_NOTIONAL']['minNotional'].values[0])
    current_price = float(ticker['lastPrice'])

    qty = min_notional / current_price
    qty = math.ceil(qty)
    return qty


def get_cash(asset):
    info = client.get_asset_balance(asset=asset)
    return float(info['free'])


def order(symbol, qty, type):
    if type.lower() == 'buy'.lower() or type.lower() == 'b'.lower():
        order = client.order_market_buy(symbol=symbol, quantity=qty)

    elif type.lower() == 'sell'.lower() or type.lower() == 's'.lower():
        order = client.order_market_sell(symbol=symbol, quantity=qty)
    else:
        print("Wrong order type")
        return None
    return order


def bot_binance(df, symbol, asset_1, asset_2, stop_loss, take_profit, print_action, is_bought, profit,
                last_bought_price, last_max_value):
    sma_single = df.single_sma[0]
    ema_single = df.single_ema[0]
    rsi = df.rsi[0]
    close_price = df.close[0]
    # open_price = df['open'].values
    # high_price = df['high'].values
    # low_price = df['low'].values
    # volume = df['volume'].values

    if not is_bought and sma_single == 1:  # and ema_single == 1:
        if get_cash(asset_1) > close_price:  # check if you have money

            order(symbol, get_min_qty(symbol), 'buy')  # place Buy Order
            is_bought = True

            last_bought_price = close_price  # set bought price

            last_max_value = close_price  # max price since bought, used for dynamic stop loss

            # bot feed action back

            if print_action:
                print("buy", close_price)
        else:
            if print_action:
                print("not enough cash to buy")

    elif is_bought and sma_single == 0:  # and ema_single == 0 : # normal Sell
        if get_cash(asset_2) > get_min_qty(symbol):  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            if close_price < (last_max_value * (1 - stop_loss)) or close_price > (
                    last_bought_price * (1 + take_profit)):

                order(symbol, get_min_qty(symbol), 'sell')  # place Sell Order
                is_bought = False

                value = (close_price - last_bought_price)
                profit += value

                if print_action:
                    print("Sell", close_price, "Profit: ", profit)

    elif is_bought and \
            (close_price < (last_max_value * (1 - stop_loss)) or
             close_price > (last_bought_price * (1 + take_profit))):  # Emergencey Sell Stop Loss

        if get_cash(asset_2) > get_min_qty(symbol):  # check if you have money

            if close_price > last_max_value:  # update max value since bought, needed for dynamic stop loss
                last_max_value = close_price

            order(symbol, get_min_qty(symbol), 'sell')  # place Sell Order
            is_bought = False

            value = (close_price - last_bought_price)
            profit += value

            if print_action:
                print("Sell", close_price, "Profit: ", profit)

        else:
            if print_action:
                print("not enough cash to sell")

    return profit, is_bought, last_max_value, last_bought_price


def run(symbol, asset_1, asset_2, interval='1m',
        sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14,
        stop_loss=0.001, take_profit=0.2, print_action=True, is_bought=False,
        profit=0, last_bought_price=-1, last_max_value=-1):
    limit = sma_long if sma_long > ema_long else ema_long

    df = get_data(symbol, interval, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14)[-limit:]
    last_open_time = df.index[-1]
    while True:
        df = get_data(symbol, interval, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14)[-1:]
        if df.index[-1] > last_open_time:
            last_open_time = df.index[-1]
            close_price = df.close[-1]
            profit, is_bought, last_max_value, last_bought_price = bot_binance(df, symbol, asset_1, asset_2, stop_loss,
                                                                               take_profit, print_action, is_bought,
                                                                               profit,
                                                                               last_bought_price, last_max_value)

            print('price: ', close_price, 'profit: ', profit, 'TRX: ', get_cash('TRX'), 'USDT: ', get_cash('USDT'),
                  'DateTime:', last_open_time)
        time.sleep(5)


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
    stop_loss = 1e-2
    take_profit = 3e-3
    current_balance = {'TRX': get_cash('TRX'), 'USDT': get_cash('USDT')}
    print(current_balance)
    run(symbol='TRXUSDT', asset_1='TRX', asset_2='USDT', stop_loss=stop_loss, take_profit=take_profit)
    # df = get_data('TRXUSDT', '1m', limit=1000)
    # fig = go.Figure()
    # # plot_order_stocks(df, 'TRXUSDT')
    # for i in range(10):
    #     plot_order_stocks(df, 'TRXUSDT',fig)
    #     time.sleep(2)
    # fig.show()


if __name__ == '__main__':
    main()
