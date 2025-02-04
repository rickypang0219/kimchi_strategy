import polars as pl
import datetime
from binance import get_binance_custom_perp_hist_df
from upbit import fetch_historical_candles


def init_data() -> pl.DataFrame:
    print("Data Fetching")
    binance = get_binance_custom_perp_hist_df("BTCUSDT", "1h")
    upbit = fetch_historical_candles(
        "USDT-BTC", 60, datetime.datetime(2020, 1, 1), datetime.datetime.now()
    )
    binance = binance.rename({"SK": "timestamp", "C": "binance_close"}).with_columns(
        pl.col("timestamp").cast(pl.Int64)
    )
    upbit = upbit.rename({"BTC_close": "upbit_close"}).sort("timestamp")
    return merge_price_dataframe(upbit, binance)


def merge_price_dataframe(
    upbit_df: pl.DataFrame, binance_df: pl.DataFrame
) -> pl.DataFrame:
    return upbit_df.join(binance_df, on="timestamp", how="inner")


if __name__ == "__main__":
    merged_df = init_data()
    print(merged_df)
