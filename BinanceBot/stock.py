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


class Stock():
    def __init__(self, api_key, api_secret, testnet=False):
        try:
            self.api_key = api_key
            self.api_secret = api_secret
            self.client = Client(api_key, api_secret, testnet=testnet)
        except Exception as e:
            self.client = None
            print(e)

    def preprossing(self, df, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14, difference=0):

        df = df.copy(deep=True)

        df['sma_short'] = df.ta.sma(length=sma_short)
        df['sma_long'] = df.ta.sma(length=sma_long)

        df['ema_short'] = df.ta.ema(length=ema_short)
        df['ema_long'] = df.ta.ema(length=ema_long)

        df['rsi'] = df.ta.rsi(length=rsi_period)

        df['single_sma'] = ((df['sma_short'] - difference) > df['sma_long']).astype(int)
        df['single_ema'] = ((df['ema_short']) > df['ema_long']).astype(int)

        df['single_rsi'] = df['rsi'].apply(lambda x: 1 if x < 30 else -1 if x >= 70 else 0)
        df['double_sma'] = (df['sma_short'] > df['sma_long']).astype(int).diff()
        df['double_ema'] = (df['ema_short'] > df['ema_long']).astype(int).diff()

        adx = df.ta.adx(length=14, high='high', low='low', close='close')
        df['single_dmp'] = (adx['DMN_14'] > adx['DMP_14']).astype(int)
        df['double_dmp'] = (df['ema_short'] > df['ema_long']).astype(int).diff()

        required_col = ['open', 'high', 'low', 'close', 'volume', 'sma_short',
                        'sma_long', 'ema_short', 'ema_long', 'rsi', 'single_sma',
                        'single_ema', 'double_sma', 'single_dmp', 'single_rsi']
        #     required_col = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'single_sma', 'single_ema']

        return df

    def get_klines(self, symbol, interval, start, end, sma_short=5, sma_long=20, ema_short=5, ema_long=20,
                   rsi_period=14,
                   limit=1000, difference=0):

        col = ['open_time', 'open', 'high', 'low', 'close',
               'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume',
               'taker_buy_quote_asset_volume', 'ignore']

        d = []
        for klines in self.client.get_historical_klines_generator(symbol, interval, start_str=start, end_str=end,
                                                                  limit=limit):
            d.append(klines)
        df = pd.DataFrame(d, columns=col)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df['high'] = np.float64(df['high'])
        df['low'] = np.float64(df['low'])
        df['open'] = np.float64(df['open'])
        df['close'] = np.float64(df['close'])
        df['volume'] = np.float64(df['volume'])
        df = df.set_index('open_time')
        df = df.drop(columns=['close_time', 'ignore'])
        df = df.astype(float)

        df = self.preprossing(df, sma_short, sma_long, ema_short, ema_long, rsi_period, difference)

        return df

    def get_data(self, symbol, interval, sma_short=5, sma_long=20, ema_short=5, ema_long=20, rsi_period=14, limit=None,
                 difference=0):
        col = ['open_time', 'open', 'high', 'low', 'close',
               'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume',
               'taker_buy_quote_asset_volume', 'ignore']

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

        df = pd.DataFrame(self.client.get_klines(symbol=symbol, interval=interval, limit=limit), columns=col)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df = df.set_index('open_time')
        df = df.drop(columns=['close_time', 'ignore'])
        df = df.astype(float)

        df = self.preprossing(df, sma_short, sma_long, ema_short, ema_long, rsi_period, difference)

        return df

    def get_min_qty(self, symbol, multiplier=1):
        info = self.client.get_symbol_info(symbol)
        ticker = self.client.get_ticker(symbol=symbol)
        filters = pd.DataFrame(info.get('filters'))
        min_lot_size = float(filters[filters.filterType == 'LOT_SIZE']['minQty'].values[0])
        min_step_size = float(filters[filters.filterType == 'LOT_SIZE']['stepSize'].values[0])
        min_notional = float(filters[filters.filterType == 'MIN_NOTIONAL']['minNotional'].values[0])
        current_price = float(ticker['lastPrice'])

        qty = min_notional / current_price
        qty = math.ceil(qty)
        return qty * multiplier

    def get_cash(self, asset):
        info = self.client.get_asset_balance(asset=asset)
        return float(info['free'])

    def order(self, symbol, qty, type):
        if type.lower() == 'buy'.lower() or type.lower() == 'b'.lower():
            order = self.client.order_market_buy(symbol=symbol, quantity=qty)

        elif type.lower() == 'sell'.lower() or type.lower() == 's'.lower():
            order = self.client.order_market_sell(symbol=symbol, quantity=qty)
        else:
            print("Wrong order type")
            return None
        return order

    def print_and_log(message):
        print(message)
        logging.info(message)
