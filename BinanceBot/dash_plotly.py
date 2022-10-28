import datetime

import dash
from dash import dcc, html
import plotly
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

google = yf.Ticker("GOOG")
google_hist = google.history(period="1d", interval="1m")

data = {
    'Date': google_hist.index.values,
    'Open': google_hist.Open.values,
    'High': google_hist.High.values,
    'Low': google_hist.Low.values,
    'Close': google_hist.Close.values,
    'Volume': google_hist.Volume.values
}


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Stock Trand Bot Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),Input('interval-component', 'n_intervals'))
def update_metrics(n):
    google_hist = google.history(period="1d", interval="1m")[-1:]
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Open: {0:.2f}'.format(google_hist.Open[0]), style=style),
        html.Span('Close: {0:.2f}'.format(google_hist.Close[0]), style=style),
        html.Span('High: {0:0.2f}'.format(google_hist.High[0]), style=style),
        html.Span('Low: {0:0.2f}'.format(google_hist.Low[0]), style=style),
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global data

    fig = plotly.subplots.make_subplots(rows=1, cols=1, row_heights=[100])
    fig.append_trace({
        'x': data['Date'],
        'y': data['Close'],
        'name': 'GOOG',
        'type': 'scatter'
    }, 1, 1)

    google_hist = google.history(period="1d", interval="1m")
    data = {
        'Date': google_hist.index.values,
        'Open': google_hist.Open.values,
        'High': google_hist.High.values,
        'Low': google_hist.Low.values,
        'Close': google_hist.Close.values,
        'Volume': google_hist.Volume.values
    }

    return fig


if __name__ == '__main__':
    import gym

    from stable_baselines3 import A2C

    env = gym.make("CartPole-v1")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    # app.run_server(debug=True)