import polars as pl


def merge_price_dataframe(
    upbit_df: pl.DataFrame, binance_df: pl.DataFrame
) -> pl.DataFrame:
    return upbit_df.join(binance_df, on="timestamp", how="inner")


if __name__ == "__main__":
    upbit_df = (
        pl.read_csv("./upbit_btc.csv")
        .rename({"BTC_close": "upbit_close"})
        .sort("timestamp")
    )
    binance_df = pl.read_csv("./binance_btc.csv").rename(
        {"SK": "timestamp", "C": "binance_close"}
    )
    print(upbit_df)
    merged_df = merge_price_dataframe(upbit_df, binance_df)
    print(merged_df)
    merged_df.write_csv("factor.csv")
