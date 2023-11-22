#%%
import sys
import argparse
import subprocess
import os
import glob
import re
from dataclasses import dataclass
import json

#parser = argparse.ArgumentParser()
#parser.add_argument("-d", "--days", help="Number of days to optimize for (backtesting)", type=int)
#args = parser.parse_args()

#days = args.days
exchange = 'binance'
days = 360
timeframes = ['1m', '5m', '15m', '1h']
strategies = []
spot_pairs = [
      "1INCH/USDT",
      "AAVE/USDT",
      "ACM/USDT",
      "ADA/USDT",
      "ADX/USDT",
      "AGLD/USDT",
      "AION/USDT",
      "AKRO/USDT",
      "ALGO/USDT",
      "ALICE/USDT",
      "ALPACA/USDT",
      "ALPHA/USDT",
      "ANKR/USDT",
      "AR/USDT",
      "ARDR/USDT",
      "ARPA/USDT",
      "ATM/USDT",
      "ATOM/USDT",
      "AUCTION/USDT",
      "AUDIO/USDT",
      "AVA/USDT",
      "AVAX/USDT",
      "AXS/USDT",
      "BADGER/USDT",
      "BAKE/USDT",
      "BAND/USDT",
      "BAR/USDT",
      "BAT/USDT",
      "BCH/USDT",
      "BEL/USDT",
      "BETA/USDT",
      "BICO/USDT",
      "BLZ/USDT",
      "BNB/USDT",
      "BNT/USDT",
      "BNX/USDT",
      "BOND/USDT",
      "BTC/USDT",
      "BTS/USDT",
      "BURGER/USDT",
      "C98/USDT",
      "CAKE/USDT",
      "CELO/USDT",
      "CELR/USDT",
      "CFX/USDT",
      "CHR/USDT",
      "CHZ/USDT",
      "CKB/USDT",
      "COMP/USDT",
      "COS/USDT",
      "COTI/USDT",
      "CRV/USDT",
      "CTSI/USDT",
      "CTXC/USDT",
      "CVP/USDT",
      "DASH/USDT",
      "DATA/USDT",
      "DEGO/USDT",
      "DENT/USDT",
      "DEXE/USDT",
      "DF/USDT",
      "DGB/USDT",
      "DOCK/USDT",
      "DODO/USDT",
      "DOGE/USDT",
      "DOT/USDT",
      "DREP/USDT",
      "DUSK/USDT",
      "DYDX/USDT",
      "EGLD/USDT",
      "ELF/USDT",
      "ENJ/USDT",
      "ENS/USDT",
      "EOS/USDT",
      "ERN/USDT",
      "ETC/USDT",
      "ETH/USDT",
      "FARM/USDT",
      "FET/USDT",
      "FIL/USDT",
      "FIO/USDT",
      "FIRO/USDT",
      "FIS/USDT",
      "FLM/USDT",
      "FLOW/USDT",
      "FLUX/USDT",
      "FOR/USDT",
      "FORTH/USDT",
      "FRONT/USDT",
      "FTM/USDT",
      "FUN/USDT",
      "FXS/USDT",
      "GALA/USDT",
      "GRT/USDT",
      "GTC/USDT",
      "HARD/USDT",
      "HBAR/USDT",
      "HIGH/USDT",
      "HIVE/USDT",
      "HOT/USDT",
      "ICP/USDT",
      "IDEX/USDT",
      "INJ/USDT",
      "IOTA/USDT",
      "IOTX/USDT",
      "KAVA/USDT",
      "KEY/USDT",
      "KMD/USDT",
      "KNC/USDT",
      "KP3R/USDT",
      "KSM/USDT",
      "LAZIO/USDT",
      "LINA/USDT",
      "LINK/USDT",
      "LIT/USDT",
      "LRC/USDT",
      "LSK/USDT",
      "LTC/USDT",
      "LTO/USDT",
      "LUNA/USDT",
      "MANA/USDT",
      "MASK/USDT",
      "MATIC/USDT",
      "MBL/USDT",
      "MBOX/USDT",
      "MDT/USDT",
      "MDX/USDT",
      "MINA/USDT",
      "MKR/USDT",
      "MLN/USDT",
      "MOVR/USDT",
      "MTL/USDT",
      "NEAR/USDT",
      "NEO/USDT",
      "NKN/USDT",
      "NMR/USDT",
      "OCEAN/USDT",
      "OG/USDT",
      "OGN/USDT",
      "OM/USDT",
      "ONE/USDT",
      "ONG/USDT",
      "ONT/USDT",
      "ORN/USDT",
      "OXT/USDT",
      "PEOPLE/USDT",
      "PERL/USDT",
      "PERP/USDT",
      "PHA/USDT",
      "PNT/USDT",
      "POLS/USDT",
      "POND/USDT",
      "PSG/USDT",
      "PUNDIX/USDT",
      "PYR/USDT",
      "QI/USDT",
      "QNT/USDT",
      "QTUM/USDT",
      "RAD/USDT",
      "RARE/USDT",
      "RAY/USDT",
      "REEF/USDT",
      "REN/USDT",
      "REQ/USDT",
      "RIF/USDT",
      "RLC/USDT",
      "RNDR/USDT",
      "ROSE/USDT",
      "RSR/USDT",
      "RUNE/USDT",
      "RVN/USDT",
      "SAND/USDT",
      "SANTOS/USDT",
      "SFP/USDT",
      "SHIB/USDT",
      "SKL/USDT",
      "SLP/USDT",
      "SNX/USDT",
      "SOL/USDT",
      "SPELL/USDT",
      "STMX/USDT",
      "STORJ/USDT",
      "STPT/USDT",
      "STRAX/USDT",
      "STX/USDT",
      "SUN/USDT",
      "SUPER/USDT",
      "SUSHI/USDT",
      "SXP/USDT",
      "TFUEL/USDT",
      "THETA/USDT",
      "TKO/USDT",
      "TLM/USDT",
      "TOMO/USDT",
      "TRB/USDT",
      "TROY/USDT",
      "TRU/USDT",
      "TRX/USDT",
      "TWT/USDT",
      "UMA/USDT",
      "UNFI/USDT",
      "UNI/USDT",
      "VET/USDT",
      "VGX/USDT",
      "VIDT/USDT",
      "VITE/USDT",
      "VTHO/USDT",
      "WAN/USDT",
      "WAVES/USDT",
      "WAXP/USDT",
      "WIN/USDT",
      "WING/USDT",
      "WNXM/USDT",
      "WRX/USDT",
      "XEC/USDT",
      "XEM/USDT",
      "XLM/USDT",
      "XMR/USDT",
      "XRP/USDT",
      "XTZ/USDT",
      "XVG/USDT",
      "XVS/USDT",
      "YFI/USDT",
      "YFII/USDT",
      "YGG/USDT",
      "ZEC/USDT",
      "ZEN/USDT",
      "ZIL/USDT",
      "ZRX/USDT"
      ]
futures_pairs = [
      "1INCH/USDT:USDT",
      "AAVE/USDT:USDT",
      "ACH/USDT:USDT",
      "ADA/USDT:USDT",
      "AGIX/USDT:USDT",
      "ALGO/USDT:USDT",
      "ALICE/USDT:USDT",
      "ALPHA/USDT:USDT",
      "ANKR/USDT:USDT",
      "ANT/USDT:USDT",
      "APE/USDT:USDT",
      "API3/USDT:USDT",
      "APT/USDT:USDT",
      "AR/USDT:USDT",
      "ARB/USDT:USDT",
      "ARPA/USDT:USDT",
      "ASTR/USDT:USDT",
      "ATA/USDT:USDT",
      "ATOM/USDT:USDT",
      "AUDIO/USDT:USDT",
      "AVAX/USDT:USDT",
      "AXS/USDT:USDT",
      "BAKE/USDT:USDT",
      "BAL/USDT:USDT",
      "BAND/USDT:USDT",
      "BAT/USDT:USDT",
      "BCH/USDT:USDT",
      "BEL/USDT:USDT",
      "BLUEBIRD/USDT:USDT",
      "BLZ/USDT:USDT",
      "BNB/USDT:USDT",
      "BTC/USDT:USDT",
      "BTCDOM/USDT:USDT",
      "C98/USDT:USDT",
      "CELO/USDT:USDT",
      "CELR/USDT:USDT",
      "CFX/USDT:USDT",
      "CHR/USDT:USDT",
      "CKB/USDT:USDT",
      "COMP/USDT:USDT",
      "COTI/USDT:USDT",
      "CRV/USDT:USDT",
      "CTK/USDT:USDT",
      "CTSI/USDT:USDT",
      "CVX/USDT:USDT",
      "DAR/USDT:USDT",
      "DASH/USDT:USDT",
      "DEFI/USDT:USDT",
      "DENT/USDT:USDT",
      "DGB/USDT:USDT",
      "DOGE/USDT:USDT",
      "DOT/USDT:USDT",
      "DUSK/USDT:USDT",
      "DYDX/USDT:USDT",
      "EDU/USDT:USDT",
      "EGLD/USDT:USDT",
      "ENJ/USDT:USDT",
      "ENS/USDT:USDT",
      "EOS/USDT:USDT",
      "ETC/USDT:USDT",
      "ETH/USDT:USDT",
      "FET/USDT:USDT",
      "FIL/USDT:USDT",
      "FLOW/USDT:USDT",
      "FOOTBALL/USDT:USDT",
      "FTM/USDT:USDT",
      "FXS/USDT:USDT",
      "GAL/USDT:USDT",
      "GALA/USDT:USDT",
      "GMT/USDT:USDT",
      "GMX/USDT:USDT",
      "GRT/USDT:USDT",
      "GTC/USDT:USDT",
      "HBAR/USDT:USDT",
      "HIGH/USDT:USDT",
      "HOOK/USDT:USDT",
      "HOT/USDT:USDT",
      "ICP/USDT:USDT",
      "ICX/USDT:USDT",
      "IMX/USDT:USDT",
      "INJ/USDT:USDT",
      "IOST/USDT:USDT",
      "IOTA/USDT:USDT",
      "IOTX/USDT:USDT",
      "JOE/USDT:USDT",
      "KAVA/USDT:USDT",
      "KLAY/USDT:USDT",
      "KNC/USDT:USDT",
      "KSM/USDT:USDT",
      "LDO/USDT:USDT",
      "LEVER/USDT:USDT",
      "LINA/USDT:USDT",
      "LINK/USDT:USDT",
      "LIT/USDT:USDT",
      "LPT/USDT:USDT",
      "LQTY/USDT:USDT",
      "LRC/USDT:USDT",
      "LTC/USDT:USDT",
      "LUNA2/USDT:USDT",
      "MAGIC/USDT:USDT",
      "MANA/USDT:USDT",
      "MASK/USDT:USDT",
      "MATIC/USDT:USDT",
      "MINA/USDT:USDT",
      "MKR/USDT:USDT",
      "MTL/USDT:USDT",
      "NEAR/USDT:USDT",
      "NEO/USDT:USDT",
      "NMR/USDT:USDT",
      "NKN/USDT:USDT",
      "OCEAN/USDT:USDT",
      "OGN/USDT:USDT",
      "ONE/USDT:USDT",
      "ONT/USDT:USDT",
      "OP/USDT:USDT",
      "PEOPLE/USDT:USDT",
      "PERP/USDT:USDT",
      "PHB/USDT:USDT",
      "QNT/USDT:USDT",
      "QTUM/USDT:USDT",
      "RDNT/USDT:USDT",
      "REEF/USDT:USDT",
      "REN/USDT:USDT",
      "RLC/USDT:USDT",
      "RNDR/USDT:USDT",
      "ROSE/USDT:USDT",
      "RSR/USDT:USDT",
      "RUNE/USDT:USDT",
      "RVN/USDT:USDT",
      "SAND/USDT:USDT",
      "SFP/USDT:USDT",
      "SKL/USDT:USDT",
      "SNX/USDT:USDT",
      "SOL/USDT:USDT",
      "SPELL/USDT:USDT",
      "SSV/USDT:USDT",
      "STMX/USDT:USDT",
      "STORJ/USDT:USDT",
      "STX/USDT:USDT",
      "SUI/USDT:USDT",
      "SUSHI/USDT:USDT",
      "SXP/USDT:USDT",
      "T/USDT:USDT",
      "THETA/USDT:USDT",
      "TLM/USDT:USDT",
      "TOMO/USDT:USDT",
      "TRB/USDT:USDT",
      "TRU/USDT:USDT",
      "TRX/USDT:USDT",
      "UNFI/USDT:USDT",
      "UNI/USDT:USDT",
      "WAVES/USDT:USDT",
      "WOO/USDT:USDT",
      "XLM/USDT:USDT",
      "XMR/USDT:USDT",
      "XRP/USDT:USDT",
      "XTZ/USDT:USDT",
      "XVS/USDT:USDT",
      "YFI/USDT:USDT",
      "ZEC/USDT:USDT",
      "ZEN/USDT:USDT",
      "ZIL/USDT:USDT",
      "ZRX/USDT:USDT"
    ]

@dataclass
class Strategy:
    name: str
    profit: float
    timeframe: str

#%%
def setup_pairs_json():
    print("Setting up pairs...")
    print(spot_pairs)
    folder = 'user_data/data/binance/'
    with open(folder+'spot_pairs.json', 'w+') as f:
        json.dump(spot_pairs, f)

#%%
def download_data():
    print("Deleting existing backtesting data...")
    folder = 'user_data/data/binance/*'
    files = glob.glob(folder)
    for f in files:
        os.remove(f)

    setup_pairs_json()

    print("Downloading new backtesting data for last " + str(days) + " days on follwing timeframes: " + ' '.join(timeframes) + "...")
    print("Exchange: " + exchange)

    arguments = ['freqtrade', 'download-data', '--days', str(days), '--exchange', str(exchange), '--erase', '--pairs-file', 'user_data/data/binance/spot_pairs.json', '-t']
    arguments.extend(timeframes)

    result = subprocess.run(arguments, capture_output=True, text=True).stdout

#%%
def backtest_strategies():
    folder = 'user_data/strategies/*'
    files = glob.glob(folder)
    for f in files:
        best_profit = -100
        best_strategy = None
        try:

            for timeframe in timeframes:
                name = re.search('strategies/(.+?)\.py', f).group(1)

                result = subprocess.run(['freqtrade', 'backtesting', '-s', name, '-i', timeframe], capture_output=True, text=True).stdout
                profit = float(re.search('(Total profit %)(.+?)\|\s(.+?)\%', result).group(3))

                if profit > best_profit:
                    best_strategy = Strategy(name, profit, timeframe)
                    best_profit = profit

            if best_strategy != None:
                strategies.append(best_strategy)
        except Exception as e:
            print("Could not backtest: " + str(name))
            print(e)
            continue

#%%
download_data()
##backtest_strategies()

#strategies.sort(key=lambda x: x.profit, reverse=True)
#for strat in strategies:
#    print(strat)
