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
from stock import Stock



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

api_key = 'hPZcl6C7b1NFfMNc2whnh2OSyxsWxyEIEP2mKDypUYDrG70eeB4rniELcx0KnAwD'
api_secret = 'tF6fS7kskVztUbDA5jkXt8GAdXLoJ82C9BaspK0F9AbV2MuG2aGZOcLoboOOUgX6'
stock = Stock(api_key, api_secret, testnet=True)
df = stock.get_klines('TRXUSDT', '1m', '1 day ago UTC', 'now UTC')

data = {
    'Date': df.index.values,
    'Open': df.open.values,
    'High': df.high.values,
    'Low': df.low.values,
    'Close': df.close.values,
    'Volume': df.volume.values
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
    df = stock.get_klines('TRXUSDT', '1m', '1 day ago UTC', 'now UTC')[-1:]
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Open: {0:.5f}'.format(df.open[0]), style=style),
        html.Span('Close: {0:.5f}'.format(df.close[0]), style=style),
        html.Span('High: {0:0.5f}'.format(df.high[0]), style=style),
        html.Span('Low: {0:0.5f}'.format(df.low[0]), style=style),
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    global data

    fig = plotly.subplots.make_subplots(rows=1, cols=1, row_heights=[100])
    fig.add_trace({
        'x': data['Date'],
        'y': data['Close'],
        'name': 'GOOG',
        'type': 'scatter'
    }, 1, 1)

    df = stock.get_klines('TRXUSDT', '1m', '1 day ago UTC', 'now UTC')[-1:]
    data = {
        'Date': df.index.values,
        'Open': df.open.values,
        'High': df.high.values,
        'Low': df.low.values,
        'Close': df.close.values,
        'Volume': df.volume.values
    }

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)