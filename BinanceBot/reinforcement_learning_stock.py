import math
import os.path
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import itertools as itr
from IPython.display import clear_output
from time import sleep
import random
import tqdm
import json
import gym
from gym import spaces

# All necessary plotly libraries
import plotly as plotly
import plotly.io as plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from BinanceBot.stock import Stock

CONFIG_FILE = 'config.json'

def save_config(data):
    global CONFIG_FILE
    with open(CONFIG_FILE, 'w') as outfile:
        json.dump(data, outfile)

def load_config():
    global CONFIG_FILE
    with open(CONFIG_FILE, 'r') as outfile:
        return json.load(outfile)

class StockTradingEnv():
    """A stock trading environment for OpenAI gym"""

    def __init__(self, df, base_currency_amount, quote_currency_amount,possible_values, action_space, signal_column=None, is_load=False, qtable_path=None):
        super(StockTradingEnv, self).__init__()

        if signal_column is None:
            signal_column = ['rsi_signal', 'sma_diff_signal', 'ema_diff_signal']

        self.df = df
        self.signal_column = signal_column
        self.action_space = [str(i) for i in action_space]  # Buy: 1, Sell: -1, Hold: 0

        all_posible_states = list(itr.product(*possible_values))
        self.observation_space = dict(zip(range(len(all_posible_states)), all_posible_states))
        self.observation_size = len(self.observation_space)

        if is_load:
            self._qtable = pd.read_csv(qtable_path)
        else:
            self._qtable = pd.DataFrame(columns=self.action_space, index=self.observation_space, dtype=np.float64)
            self._qtable = self._qtable.fillna(0)

        self.state = None
        self.profit = 0
        self.trades = []
        self.steps = None
        self.last_bought_price = -1
        self.last_action = 0
        self.is_bought = False
        self.can_buy = True
        self.init_base_currency_amount = base_currency_amount
        self.base_currency_amount = base_currency_amount

    def _reset(self):
        self.profit = 0
        self.trades = []
        self.state = None
        self.steps = 0
        self.is_bought = False
        self.base_currency_amount = self.init_base_currency_amount
        return self.state

    def make_state(self, row):
        row = row[self.signal_column]
        row.fillna(0, inplace=True)
        temp_state = [row[i] for i in range(len(row))]
        return temp_state

    def _next_observation(self, state):
        # for i in range(len(self.observation_space)):
        #     if self.observation_space[i] == (*state, self.is_bought):
        #         return i

        for key, values in self.observation_space.items():
            if values == (*state, self.can_buy,self.is_bought):
                return key

        return np.random.randint(0, self.observation_size)

    def get_state(self, row):
        return self._next_observation(self.make_state(row))

    def _step(self, action, row):
        temp_state = self.make_state(row)
        reward = self._take_action(action)
        state = self._next_observation(temp_state)
        # print(temp_state, action)
        return state, reward

    def _take_action(self, action):
        reward = 0
        profit = 0
        self.trades.append(self.df.loc[self.steps, 'close'])

        if action == str(1):  # buy
            # self.profit = self.df.loc[self.steps, 'close'] - self.last_bought_price
            if self.is_bought:
                reward = -1
            else:
                if self.base_currency_amount > self.df.loc[self.steps, 'close']:
                    self.base_currency_amount -= self.df.loc[self.steps, 'close']
                    reward = 1
                    self.last_bought_price = self.df.loc[self.steps, 'close']
                    self.is_bought = True
                else:
                    reward = -1
        elif action == str(-1):  # sell
            if not self.is_bought:
                reward = -1
            else:
                profit = self.df.loc[self.steps, 'close'] - self.last_bought_price
                self.profit += profit
                self.last_bought_price = 0
                self.is_bought = False

        if self.base_currency_amount > self.df.loc[self.steps, 'close']:
            self.can_buy = True
        else:
            self.can_buy = False


        if profit > 0:
            reward = 1
        elif profit < 0:
            reward = -1

        return reward

    def learn(self, episodes, alpha, gamma, epsilon, epsilon_decay, save_freq=100):
        bs = {'1':'Buy', '-1':'Sell', '0':'Hold'}
        print_info = ''
        bar = tqdm.tqdm(total=episodes)
        for episode in range(episodes):
            reward_ = 0
            state = self._reset()
            for date_time_index, row in self.df.iterrows():
                self.steps = date_time_index
                if state == None:
                    action = 0
                    next_state, reward = self._step(action, row)
                    state = next_state
                    continue
                if np.random.random() < epsilon:
                    action = np.random.choice(self.action_space)
                    print_info = f'Random: {bs[action]}'
                else:
                    action = self._qtable.loc[state].idxmax()
                    print_info = f'Qtable: {bs[action]}'

                next_state, reward = self._step(action,row)

                reward_ += reward
                self._qtable.loc[state, str(action)] += alpha * (
                        reward + gamma * self._qtable.loc[next_state].max() - self._qtable.loc[state, str(action)])
                self._qtable.loc[state, str(action)] = np.round(self._qtable.loc[state, str(action)] , 4)
                state = next_state

                # print('Action: ', print_info, 'Reward: ', reward, 'Close: ', row['close'], 'Profit: ', self.profit)
                # sleep(0.2)

            if epsilon != 0.01:
                epsilon *= epsilon_decay

            bar.n = episode
            bar.update()
            bar.set_description(f'Episode: {episode} Reward: {reward_}, Profit: {self.profit} Epsilon: {epsilon} Alpha: {alpha} Gamma: {gamma}')
            # clear_output(wait=True)
            # sleep(0.3)

            if (episode + 1) % save_freq == 0:
                save_config({'epsilon': epsilon})
                self._qtable.to_csv('qtable.csv', index=False, header=True)

        return self._qtable

    def test(self):
        reward_ = 0
        total_rows = self.df.shape[0]
        bar = tqdm.tqdm(total=total_rows)
        for date_time_index,row in self.df.iterrows():
            self.steps = date_time_index
            state = self.get_state(row)
            action = self._qtable.loc[state].idxmax()
            reward = self._take_action(str(action))
            reward_ += reward

            bar.update(1)
            bar.set_description(f'Reward: {reward_}, Profit: {self.profit}')

    def render(self, mode='human', close=False):
        profit = 0
        if len(self.trades) > 0:
            profit = self.trades[-1] - self.trades[0]
        print(f'Step: {self.steps} Profit: {profit}')


        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        #                     subplot_titles=('Price', 'RSI'))
        # fig.add_trace(go.Scatter(x=self.df.loc[self.frame_bound[0]:self.steps, 'open_time'],
        #                             y=self.df.loc[self.frame_bound[0]:self.steps, 'close'],
        #                             name='Price'), row=1, col=1)
        # fig.add_trace(go.Scatter(x=self.df.loc[self.frame_bound[0]:self.steps, 'open_time'],
        #                             y=self.df.loc[self.frame_bound[0]:self.steps, 'rsi'],
        #                             name='RSI'), row=2, col=1)
        # fig.update_layout(height=600, width=1000, title_text="Stock Trading")
        # fig.show()

        clear_output(wait=True)
        # sleep(0.5)


if __name__ == "__main__":
    # Binance API
    api_key = 'hPZcl6C7b1NFfMNc2whnh2OSyxsWxyEIEP2mKDypUYDrG70eeB4rniELcx0KnAwD'
    api_secret = 'tF6fS7kskVztUbDA5jkXt8GAdXLoJ82C9BaspK0F9AbV2MuG2aGZOcLoboOOUgX6'
    stock = Stock(api_key, api_secret, testnet=True)

    symbol = 'BTCUSDT'

    df = stock.get_klines(symbol, '1m', '1 day ago UTC', 'now UTC')


    # df = pd.read_pickle('../datafram.pkl')[:200]
    # df = stock.preprossing(df)
    del stock

    # single_sma(1,0), single_ema(1,0) , single_dmp(1,0) , double_dmp(1,-1,0)   single_rsi(-1,1,0) , double_ema(-1,1,0), double_sma(-1,1,0), Actions(-1,1,0)

    possible_values =  [ [1,0], #single_sma(1,0)
                         [1,0], # single_ema
                         # [1,0], # single_dmp
                         # [-1,1,0], # double_dmp
                         [-1,1,0], # single_rsi
                         # [-1,1,0], # double_ema
                         # [-1,1,0], # double_sma
                         [1,0], # Can Buy
                         [0,1] # is_bought
                         ]
    action_space = [-1,1,0]
    # signal_column = ['single_sma','single_ema', 'single_dmp', 'double_dmp', 'single_rsi', 'double_ema', 'double_sma']
    signal_column = ['single_sma', 'single_ema', 'single_rsi' ]

    base_currency_amount = 100000
    quote_currency_amount = 2

    episodes, alpha, gamma, epsilon, epsilon_decay = 1000, 0.9, 0.9, 0.9, 0.99

    RESET = True

    if not RESET:
        if os.path.exists(CONFIG_FILE):
            data = load_config()
            epsilon = data["epsilon"]
        is_load = True
    else:
        is_load = False

    # Create environment
    path = f'qtable{symbol}.csv'

    env = StockTradingEnv(df, base_currency_amount, quote_currency_amount,possible_values,
                          action_space, signal_column, is_load, path)

    # env.test()
    qtable = env.learn(episodes, alpha, gamma, epsilon, epsilon_decay)
    qtable.to_csv(path, index=False, header=True)
    print(qtable)

