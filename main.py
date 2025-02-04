import polars as pl
import numpy as np
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from data_merging import init_data


def update_position(
    position: np.ndarray, long_entry: np.ndarray, short_entry: np.ndarray
) -> np.ndarray:
    for i in range(1, len(position)):
        if position[i - 1] == 0:
            if long_entry[i]:
                position[i] = 1
            elif short_entry[i]:
                position[i] = -1
        elif position[i - 1] == 1:
            if short_entry[i]:
                position[i] = -1
            else:
                position[i] = 1
        elif position[i - 1] == -1:
            if long_entry[i]:
                position[i] = 1
            else:
                position[i] = -1
    return position


class BackTester:
    def __init__(self, factors_df: pl.DataFrame):
        self.factors = factors_df.drop_nulls().sort("timestamp")
        self.TRANSACTION_COST: float = 0.06 / 100

    def _kimchi_strategy(self, x: float, y: float) -> pl.DataFrame:
        trade_info = pl.DataFrame()
        trade_info = trade_info.with_columns(
            self.factors["timestamp"].alias("timestamp")
        )
        upbit_factor = self.factors["upbit_close"].pct_change().fill_nan(0)

        position = np.zeros(len(self.factors["timestamp"]))
        long_entry = (upbit_factor.to_numpy() >= x).astype(int)
        short_entry = (upbit_factor.to_numpy() <= -y).astype(int) * -1
        updated_position = update_position(position, long_entry, short_entry)
        trade_info = trade_info.with_columns([pl.Series("position", updated_position)])
        return trade_info

    def _compute_trans_cost(self, trade_info: pl.DataFrame) -> pl.DataFrame:
        trade_info = trade_info.with_columns(
            [
                (
                    abs(pl.col("position") - pl.col("position").shift(1))
                    * self.TRANSACTION_COST
                ).alias("trans_cost")
            ]
        )
        return trade_info

    def _compute_PnL(self, trade_info: pl.DataFrame) -> pl.DataFrame:
        trade_info = self._compute_trans_cost(trade_info)
        trade_info = trade_info.with_columns(
            [
                self.factors["binance_close"].pct_change().alias("returns"),
            ]
        )
        trade_info = trade_info.with_columns(
            [
                (
                    pl.col("position").shift(1) * pl.col("returns")
                    - pl.col("trans_cost")
                ).alias("PnL")
            ]
        )
        return trade_info

    def _compute_cum_PnL(self, trade_info: pl.DataFrame) -> pl.DataFrame:
        trade_info = trade_info.with_columns(
            [
                pl.col("PnL").cum_sum().alias("strategy_cumPnL"),
                pl.col("returns").cum_sum().alias("benchmark_cumPnL"),
            ]
        )
        return trade_info

    def _compute_trade_statistics(self, x: float, y: float) -> pl.DataFrame:
        if self.factors is None:
            raise ValueError("Factors DataFrame is not provided")
        trade_info: pl.DataFrame = self._kimchi_strategy(x, y)
        trade_info = self._compute_PnL(trade_info)
        trade_info = self._compute_cum_PnL(trade_info)
        return trade_info

    def _convert_humanized_timestamp(self, df: pl.dataframe) -> pl.dataframe:
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("humanized_timestamp")
        )
        return df

    def compute_sharpe_ratio(
        self, trade_info: pl.DataFrame, trading_days: int
    ) -> float:
        trade_info = self._convert_humanized_timestamp(trade_info)
        trade_info = (
            trade_info.with_columns(pl.col("humanized_timestamp").dt.truncate("1d"))
            .group_by("humanized_timestamp")
            .agg(pl.col("PnL").sum().alias("aggPnL"))
        )
        agg_pnl = trade_info["aggPnL"].drop_nulls().to_list()
        if agg_pnl:
            mean = np.mean(agg_pnl)
            sd = np.std(agg_pnl, ddof=1)
            if sd == 0:
                return 0
            return (mean / sd) * np.sqrt(trading_days)
        return 0

    def compute_information_ratio(
        self, trade_info: pl.DataFrame, trading_days: int
    ) -> float:
        trade_info = self._convert_humanized_timestamp(trade_info)
        trade_info = (
            trade_info.with_columns(pl.col("humanized_timestamp").dt.truncate("1d"))
            .group_by("humanized_timestamp")
            .agg(
                [
                    pl.col("PnL").sum().alias("aggPnL"),
                    pl.col("returns").sum().alias("aggReturns"),
                ]
            )
        )
        agg_pnl = trade_info["aggPnL"].drop_nulls().to_numpy()
        agg_returns = trade_info["aggReturns"].drop_nulls().to_numpy()

        if (len(agg_pnl) != 0) and (len(agg_returns) != 0):
            excess = agg_pnl - agg_returns
            mean = np.mean(excess)
            sd = np.std(excess, ddof=1)
            if sd == 0:
                return 0
            return (mean / sd) * np.sqrt(trading_days)
        return 0

    def _compute_max_drawdown(self, trade_info: pl.DataFrame) -> float:
        returns = trade_info.select("PnL").drop_nulls().to_numpy()
        cum_prod_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_prod_returns)
        drawdown = cum_prod_returns / running_max - 1
        return np.min(drawdown)

    def _compute_long_short_ratio(self, trade_info: pl.DataFrame) -> float | None:
        long_count = trade_info.filter(pl.col("position") == 1).shape[0]
        short_count = trade_info.filter(pl.col("position") == -1).shape[0]
        if (short_count != 0) and (long_count != 0):
            return long_count / short_count
        return None

    def _compute_cagr(self, trade_info: pl.DataFrame, days=365) -> float | None:
        trade_info = self._convert_humanized_timestamp(trade_info)
        trade_info = (
            trade_info.with_columns(pl.col("humanized_timestamp").dt.truncate("1d"))
            .group_by("humanized_timestamp")
            .agg(
                [
                    pl.col("PnL").sum().alias("aggPnL"),
                    pl.col("returns").sum().alias("aggReturns"),
                ]
            )
        )
        agg_pnl = trade_info["aggPnL"].drop_nulls().to_numpy()
        cum_pnl = (1 + agg_pnl).cumprod()
        start_val, end_val = cum_pnl[0], cum_pnl[-1]
        n_years = len(cum_pnl) / days
        return (end_val / start_val) ** (1 / n_years) - 1 if start_val != 0 else None

    def print_trade_summary_stats(
        self, rolling_window: float | int, multiplier: float
    ) -> None:
        trade_info = self._compute_trade_statistics(rolling_window, multiplier)
        sharpe: float = self.compute_sharpe_ratio(trade_info, 365)
        ir: float = self.compute_information_ratio(trade_info, 365)
        mdd = self._compute_max_drawdown(trade_info)
        ls_ratio = self._compute_long_short_ratio(trade_info)
        cagr = self._compute_cagr(trade_info, 365)
        print(
            f"### Trade Summary Statistics ### \n"
            f"Params Set {rolling_window, multiplier} \n"
            f"Strategy Cum PnL: {trade_info['strategy_cumPnL'][-1]:.3f} \n"
            f"Benchmark Cum PnL {trade_info['benchmark_cumPnL'][-1]:.3f} \n"
            f"Annualized Sharpe Ratio: {sharpe:.3f} \n"
            f"Annualized Information Ratio: {ir:.3f} \n"
            f"Maximum Drawdown: {mdd * 100:.0f}% \n"
            f"Long Short Ratio: {ls_ratio if ls_ratio is not None else 'error'}  \n"
            f"CAGR :{cagr * 100 if cagr is not None else 'error'} % \n"
            f"################################ \n"
        )

    def _compute_sharpe_in_optimization(self, params: tuple[int, float]) -> float:
        trade_stat = self._compute_trade_statistics(params[0], params[1])
        return self.compute_sharpe_ratio(trade_stat, 365)

    def _compute_sharpe_with_params(self, params: tuple[int, float]) -> tuple:
        sharpe_ratio = self._compute_sharpe_in_optimization(params)
        return (params, sharpe_ratio)

    def optimize_params_and_plot_heatmap(
        self, rolling_windows: np.ndarray, multipliers: np.ndarray
    ) -> None:
        xy_pairs = [(xi, yi) for yi in multipliers for xi in rolling_windows]
        with Pool() as pool:
            results = pool.map(self._compute_sharpe_with_params, xy_pairs)
        xy_pairs, z_values = zip(*results)
        z = np.array(z_values).reshape(len(multipliers), len(rolling_windows))
        rolling_windows_expanded = np.append(
            rolling_windows,
            rolling_windows[-1] + (rolling_windows[-1] - rolling_windows[-2]),
        )
        multipliers_expanded = np.append(
            multipliers, multipliers[-1] + (multipliers[-1] - multipliers[-2])
        )
        rolling_windows_expanded = np.array(rolling_windows_expanded).flatten()
        multipliers_expanded = np.array(multipliers_expanded).flatten()
        xticklabels = [f"{x * 100:.0f}%" for x in rolling_windows_expanded]
        yticklabels = [f"{y * 100:.1f}%" for y in multipliers_expanded]
        ax = sns.heatmap(
            z,
            annot=True,
            fmt=".2f",
            cmap="coolwarm_r",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            vmin=-2,  # Set minimum bound of color scale
            vmax=3.5,  # Set maximum bound of color scale
            center=1,  # Set midpoint of the color scale
            annot_kws={"size": 5},
        )
        ax.set_xlabel("X%")
        ax.set_ylabel("Y%")
        ax.set_title("Heatmap of Params Set")
        plt.show()

    def plot_returns(self, trade_info: pl.DataFrame) -> None:
        trade_info = self._convert_humanized_timestamp(trade_info)
        trade_info_pd = trade_info.to_pandas()
        plt.title("Cumulative PnL of Market VS Strategy")
        plt.plot(
            trade_info_pd["humanized_timestamp"],
            trade_info_pd["benchmark_cumPnL"],
            label="Market",
        )
        plt.plot(
            trade_info_pd["humanized_timestamp"],
            trade_info_pd["strategy_cumPnL"],
            label="Strategy",
        )
        plt.legend()
        plt.xlabel("timestamp")
        plt.ylabel("Cumulative PnL")
        plt.show()


if __name__ == "__main__":
    factors_df = pl.read_csv("factor.csv")
    # factors_df = init_data()
    backtest_df = factors_df[: len(factors_df) // 2, :]
    backtester = BackTester(backtest_df)
    x = np.linspace(0.01, 0.15, 20)
    y = np.linspace(0.01, 0.3, 20)
    backtester.optimize_params_and_plot_heatmap(x, y)

    # Forward Test
    forward_df = factors_df[len(factors_df) // 2 :, :]
    backtester_forward = BackTester(forward_df)
    forward_x = np.linspace(0.01, 0.15, 20)
    forward_y = np.linspace(0.01, 0.3, 40)
    backtester_forward.optimize_params_and_plot_heatmap(forward_x, forward_y)

    trade_info = backtester_forward._compute_trade_statistics(0.1, 0.25)
    backtester_forward.print_trade_summary_stats(0.01, 0.25)
    backtester_forward.plot_returns(trade_info)
