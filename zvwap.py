# Import necessary modules
import copy
import logging
import pathlib
import rapidjson
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade, LocalTrade
from datetime import datetime, timedelta
import time
from typing import Optional, Union
import warnings

class ZVWAPLBStrategy(IStrategy):
    # Define hyperparameters
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    timeframe = '5m'
    can_short: bool = False

    # Define strategy parameters
    strategy_params = {
        'length': 23,
        'lowerBottom': -1,
        'sellLine': 1.0,
        'stopLoss': 5,
        'rsi_length': 9,
        'rsi_sma_length': 14
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate ZVWAP
        dataframe['zvwap'] = self.calc_zvwap(dataframe, length=self.strategy_params['length'])

        # RSI
        dataframe['rsiValue'] = ta.RSI(dataframe['close'], timeperiod=self.strategy_params['rsi_length'])
        dataframe['rsi_sma'] = ta.SMA(dataframe['rsiValue'], timeperiod=self.strategy_params['rsi_sma_length'])

        # EMA
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        return dataframe

    def calc_zvwap(self, dataframe: DataFrame, length: int) -> Series:
        mean = (dataframe['volume'] * dataframe['close']).rolling(window=length).sum() / dataframe['volume'].rolling(window=length).sum()

        var_pow = (dataframe['close'] - mean).pow(2)
        var_sma = ta.SMA(var_pow, timeperiod=length)
        vwapsd = np.sqrt(var_sma)
        
        return (dataframe['close'] - mean) / vwapsd

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['zvwap'], self.strategy_params['lowerBottom'])) &
                (dataframe['close'] > dataframe["ema_100"]) &
                (dataframe['rsiValue'] > dataframe['rsi_sma'])
            ),
            'enter_long'] = 1

        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['rsiValue'], 80)) |
                ((dataframe['zvwap'] > 1.1) & (dataframe['rsiValue'] > 60))
            ),
            'exit_long'] = 1

        return dataframe