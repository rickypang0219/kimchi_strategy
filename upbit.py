import requests
import datetime
import time
import polars as pl
import datetime


def convert_to_utc_timestamp(date_string):
    local_datetime = datetime.datetime.fromisoformat(date_string)
    utc_datetime = local_datetime.replace(tzinfo=datetime.timezone.utc)
    utc_timestamp = utc_datetime.timestamp()
    return int(utc_timestamp * 1000)


def fetch_historical_candles(market, unit, from_date, to_date):
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    headers = {"Accept": "application/json"}
    candles = []
    to = to_date

    while to > from_date:
        params = {
            "market": market,
            "to": to.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": 200,  # Maximum count Upbit allows in one request
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()
        if not data:
            print("Upbit Data Fetching Completed")
            break

        candles.extend(
            [
                {
                    "timestamp": convert_to_utc_timestamp(_i["candle_date_time_utc"]),
                    f"{market.replace('USDT-', '')}_close": _i["trade_price"],
                    # f"{market.replace('KRW-', '')}_volume": _i[
                    # "candle_acc_trade_volume"
                    # ],
                }
                for _i in data
            ]
        )
        to = datetime.datetime.strptime(
            data[-1]["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S"
        )
    return pl.DataFrame(candles)


def fetch_all_coins_dataframe(
    market_list: list[str], unit, from_data, to_data
) -> pl.DataFrame:
    df_list: list[pl.DataFrame] = []
    for market in market_list:
        candles: pl.DataFrame = fetch_historical_candles(
            market, unit, from_date, to_date
        )
        df_list.append(candles)

    if df_list:
        result = df_list[0]
        for i, df in enumerate(df_list[1:]):
            result = result.join(df, on="timestamp", how="outer", suffix=f"_df{i + 2}")
        return result
    return pl.DataFrame()


if __name__ == "__main__":
    market = "USDT-BTC"
    unit = 15  # 60-minute intervals
    from_date = datetime.datetime(2020, 1, 1)
    to_date = datetime.datetime.now()
    candles = fetch_historical_candles(market, unit, from_date, to_date)
    print(candles)
    candles.write_csv("upbit_btc.csv")
