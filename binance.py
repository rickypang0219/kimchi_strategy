from datetime import datetime, timedelta
from typing import TypedDict
import polars as pl
import requests


class CryptoFuturesPriceItem(TypedDict):
    PK: str  # Symbol
    SK: str  # Timestamp
    O: float  # open # noqa: E741
    H: float  # high
    L: float  # low
    C: float  # close
    V: float  # volume
    T: str  # close time
    BAV: float  # base asset volume
    N_TRADE: int  # number of trade
    TBV: float  # taker buy volume
    TBAV: float  # taker buy asset volume


def convert_perp_futures_to_db_items(
    fetched_crypto_price_data: list[list], symbol: str
) -> list[CryptoFuturesPriceItem]:
    """A function that turns the fetced dictionary to
    CryptoFuturesPirceItem

    Attributes:
        crypto_price: list[list[str|float]]

    Output:
        crypto_price_db_item: list[CryptoFuturesPirceItem]
    """
    return [
        CryptoFuturesPriceItem(
            PK=f"{symbol}#PERP",
            SK=str(item[0]),
            O=float(item[1]),
            H=float(item[2]),
            L=float(item[3]),
            C=float(item[4]),
            V=float(item[5]),
            T=str(item[6]),
            BAV=float(item[7]),
            N_TRADE=int(item[8]),
            TBV=float(item[9]),
            TBAV=float(item[10]),
        )
        for item in fetched_crypto_price_data
    ]


def fetch_binance_perpetual_candles_custom(
    symbol: str,
    start_timestamp: int,
    end_timestamp: int,
    interval: str,
    limit: int = 10,
) -> list:
    res = requests.get(
        url="https://fapi.binance.com/fapi/v1/continuousKlines",
        params={
            "pair": symbol,
            "startTime": int(start_timestamp),
            "endTime": int(end_timestamp),
            "contractType": "PERPETUAL",
            "interval": interval,
            "limit": limit,
        },
        timeout=60,
    )

    res.raise_for_status()  # Raise an exception for HTTP errors
    return res.json()


def get_binance_perp_hist_data_custom(
    symbol: str, start_date: str, end_date: str, interval: str
) -> list:
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    start_time = int(start_datetime.timestamp() * 1000)
    end_time = int(end_datetime.timestamp() * 1000)
    all_klines = []

    while start_time < end_time:
        klines = fetch_binance_perpetual_candles_custom(
            symbol, start_time, end_time, interval, limit=1500
        )
        if not klines:
            print("No klines returned, Complete Binance Data Fetching.")
            break
        all_klines.extend(klines)
        start_time = klines[-1][0] + 1  # Move to the next timestamp
    return all_klines


def get_binance_custom_perp_hist_df(symbol: str, interval: str) -> pl.DataFrame:
    # start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    start_date = "2020-01-01"
    end_date_df = datetime.now().date() + timedelta(days=1)
    end_date = end_date_df.strftime("%Y-%m-%d")
    data = get_binance_perp_hist_data_custom(symbol, start_date, end_date, interval)
    db_item = convert_perp_futures_to_db_items(data, symbol)
    df = pl.DataFrame(db_item)[:-1, :]  # exclude last row
    return df.select(["SK", "C"]).sort("SK")


if __name__ == "__main__":
    resolution = "1h"
    res = get_binance_custom_perp_hist_df("BTCUSDT", resolution)
    res.write_csv("binance_btc.csv")
    print(res)
