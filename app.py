import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json  

st.set_page_config(page_title="MMM Portfolio", initial_sidebar_state="collapsed", layout="wide")

def to_percentage(val):
    """Format a float value as a raw number string without a percentage symbol."""
    try:
        return f"{val:.2f}"
    except Exception:
        return "N/A"

def detect_time_interval(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    time_diffs = df[time_col].diff().dropna()
    if time_diffs.empty:
        return None
    median_diff = time_diffs.median().total_seconds()
    minutes = round(median_diff / 60)
    if minutes < 60:
        return f"{minutes}min"
    hours = round(minutes / 60)
    if hours < 24:
        return f"{hours}h"
    days = round(hours / 24)
    return f"{days}d"

def get_periods_per_year(interval):
    m = re.match(r"(\d+)\s*(s|sec|second|seconds|m|min|minute|minutes|h|hr|hour|hours|d|day|days)", interval, re.IGNORECASE)
    if m:
        number, unit = m.groups()
        number = float(number)
        unit = unit.lower()
        if unit in ['s', 'sec', 'second', 'seconds']:
            return 31536000 / number
        elif unit in ['m', 'min', 'minute', 'minutes']:
            return 525600 / number
        elif unit in ['h', 'hr', 'hour', 'hours']:
            return 8760 / number
        elif unit in ['d', 'day', 'days']:
            return 365 / number
    else:
        mapping = {
            'seconds': 31536000,
            'minute': 525600,
            'hourly': 8760,
            'daily': 365,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        return mapping.get(interval.lower(), 365)

def calculate_metrics(df, interval, annualization_factor=None):
    periods_per_year = get_periods_per_year(interval)
    if annualization_factor is None:
        annualization_factor = np.sqrt(periods_per_year)
        
    pnl_std = df['pnl'].std()
    mean_return = df['pnl'].mean()
    sharpe_ratio = (mean_return / pnl_std * annualization_factor) if pnl_std != 0 else 0
    cumulative_pnl = df['pnl'].cumsum()
    max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()
    num_trades = int(df['trade'].sum())
    total_returns = float(df['pnl'].sum())
    annualized_avg_return = mean_return * periods_per_year
    trades_per_interval = num_trades / len(df) if len(df) else 0

    trade_mask = df['trade'] == 1
    trade_pnls = df.loc[trade_mask, 'pnl']
    if num_trades > 0:
        average_trade_return = trade_pnls.mean()
        win_trades = trade_pnls[trade_pnls > 0]
        loss_trades = trade_pnls[trade_pnls < 0]
        win_rate = len(win_trades) / num_trades
        average_winning_trade = win_trades.mean() if not win_trades.empty else 0
        average_losing_trade = loss_trades.mean() if not loss_trades.empty else 0
        profit_factor = win_trades.sum() / abs(loss_trades.sum()) if loss_trades.sum() != 0 else float('inf')
    else:
        average_trade_return = 0
        win_rate = 0
        average_winning_trade = 0
        average_losing_trade = 0
        profit_factor = 0

    return {
        "num_of_trades": num_trades,
        "total_returns": total_returns,
        "annualized_avg_return": annualized_avg_return,
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "trades_per_interval": trades_per_interval,
        "average_trade_return": average_trade_return,
        "win_rate": win_rate,
        "average_winning_trade": average_winning_trade,
        "average_losing_trade": average_losing_trade,
        "profit_factor": profit_factor,
        "cumulative_pnL": cumulative_pnl  
    }

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def color_negative_positive(val):
    try:
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
    except Exception:
        return ''

def select_diversified_strategies(corr_matrix, threshold=0.65):
    strategies = list(corr_matrix.index)
    while True:
        high_corr_pairs = [
            (i, j) for i in strategies for j in strategies 
            if i != j and corr_matrix.loc[i, j] > threshold
        ]
        if not high_corr_pairs:
            break
        count = {strategy: 0 for strategy in strategies}
        for i, j in high_corr_pairs:
            count[i] += 1
            count[j] += 1
        worst_strategy = max(count, key=count.get)
        strategies.remove(worst_strategy)
    return strategies

def main():
    st.title("MMM Portfolio")
    
    uploaded_files = st.sidebar.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)
    
    strategies = {}
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = load_csv(uploaded_file)
                if 'pnl' not in df.columns or 'trade' not in df.columns:
                    st.error(f"File {uploaded_file.name} does not contain required 'pnl' and 'trade' columns.")
                    continue
                strategies[uploaded_file.name] = df
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
    
    if not strategies:
        st.info("Please upload one or more CSV files from the sidebar.")
    else:
        all_strategy_names = list(strategies.keys())
        tab1, tab2, tab3 = st.tabs(["Individual Strategy", "Portfolio", "Correlation"])
        
        # --- Individual Strategy Tab ---
        with tab1:
            st.header("📊 Individual Strategy Performance")
            selected_strategy = st.selectbox("Select a strategy to view its performance:", options=["None"] + all_strategy_names)
            if selected_strategy != "None":
                strategy_trades = strategies[selected_strategy]
                candidate_columns = ['Entry Date', 'Exit Date', 'Trade Date', 'Date/Time', 'timestamp', 'datetime', 'date']
                date_column = next((col for col in strategy_trades.columns if col in candidate_columns), None)
                if not date_column:
                    st.warning("No date column found. Defaulting to index-based analysis with 'daily' interval.")
                    detected_interval = 'daily'
                else:
                    st.info(f"Using '{date_column}' column for time detection.")
                    detected_interval = detect_time_interval(strategy_trades, date_column) or 'daily'
                    st.info(f"Detected interval: {detected_interval} — Periods per year: {get_periods_per_year(detected_interval)}")
                
                metrics = calculate_metrics(strategy_trades, detected_interval)
                
                st.write("### Performance Metrics")
                display_metrics = {}
                for key, value in metrics.items():
                    if key == "cumulative_pnL":
                        continue
                    if key in ["total_returns", "annualized_avg_return", "max_drawdown", "trades_per_interval",
                               "average_trade_return", "win_rate", "average_winning_trade", "average_losing_trade"]:
                        display_metrics[key] = to_percentage(value)
                    elif key == "sharpe_ratio":
                        display_metrics[key] = f"{value:.2f}"
                    elif key == "profit_factor":
                        display_metrics[key] = f"{value:.2f}"
                    elif key == "num_of_trades":
                        display_metrics[key] = f"{value}"
                    else:
                        display_metrics[key] = f"{value}"
                
                cols = st.columns(3)
                i = 0
                for metric, val in display_metrics.items():
                    cols[i % 3].metric(metric.replace("_", " ").title(), val)
                    i += 1
                
                if date_column:
                    strategy_trades[date_column] = pd.to_datetime(strategy_trades[date_column])
                    strategy_trades = strategy_trades.sort_values(date_column)
                    equity_series = strategy_trades['pnl'].cumsum()
                    fig_ind = go.Figure()
                    fig_ind.add_trace(go.Scatter(x=strategy_trades[date_column], y=equity_series, mode='lines', name="Equity Curve"))
                    fig_ind.update_layout(title="Cumulative Profit Over Time", xaxis_title="Time", yaxis_title="Cumulative PnL")
                    st.plotly_chart(fig_ind, use_container_width=True)
                else:
                    st.line_chart(strategy_trades['pnl'].cumsum())
                
                if date_column:
                    st.write("### Monthly Performance")
                    monthly_return = strategy_trades.set_index(pd.to_datetime(strategy_trades[date_column]))['pnl'].resample('ME').sum()
                    monthly_return_df = monthly_return.to_frame(name='Return')
                    monthly_return_df['Year'] = monthly_return_df.index.year
                    monthly_return_df['Month'] = monthly_return_df.index.strftime('%b')
                    pivot_table = monthly_return_df.pivot(index='Year', columns='Month', values='Return')
                    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    pivot_table = pivot_table.reindex(columns=month_order)
                    ytd_series = strategy_trades.set_index(pd.to_datetime(strategy_trades[date_column]))['pnl'].resample('YE').sum()
                    ytd_series.index = ytd_series.index.year
                    pivot_table["YTD"] = ytd_series
                    styled_table = pivot_table.style.format("{:.2f}").map(color_negative_positive)
                    st.dataframe(styled_table, use_container_width=True)
        
        # --- Portfolio Tab ---
        with tab2:
            st.header("📊 Portfolio Performance")
            selected_portfolio = st.multiselect("Select strategies to include in the portfolio:", options=all_strategy_names)
            if selected_portfolio:
                candidate_columns = ['Entry Date', 'Exit Date', 'Trade Date', 'Date/Time', 'timestamp', 'datetime', 'date']
                daily_pnl_list = []
                daily_trade_list = []
                for strategy_name in selected_portfolio:
                    df = strategies[strategy_name]
                    date_column = next((col for col in df.columns if col in candidate_columns), None)
                    if date_column is None:
                        continue
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.sort_values(date_column)
                    daily_pnl = df.set_index(date_column)['pnl'].resample('D').sum()
                    daily_trade = df.set_index(date_column)['trade'].resample('D').sum()
                    daily_pnl_list.append(daily_pnl)
                    daily_trade_list.append(daily_trade)
                if not daily_pnl_list:
                    st.error("None of the selected strategies have a valid date column for daily resampling.")
                else:
                    pnl_df = pd.concat(daily_pnl_list, axis=1)
                    portfolio_daily_pnl = pnl_df.mean(axis=1)
                    trade_df = pd.concat(daily_trade_list, axis=1)
                    portfolio_daily_trade = trade_df.sum(axis=1)
                    
                    portfolio_df = pd.DataFrame({"pnl": portfolio_daily_pnl, "trade": portfolio_daily_trade})
                    aggregated_metrics = calculate_metrics(portfolio_df, "daily")
                    
                    rolling_window = 180
                    rolling_mean = portfolio_daily_pnl.rolling(window=rolling_window).mean()
                    rolling_std = portfolio_daily_pnl.rolling(window=rolling_window).std()
                    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(365)).replace([np.inf, -np.inf], np.nan)
                    avg_rolling_sharpe = rolling_sharpe.mean()
                    
                    raw_list = []
                    for strategy_name in selected_portfolio:
                        df = strategies[strategy_name].copy()
                        date_column = next((col for col in df.columns if col in candidate_columns), None)
                        if date_column is None:
                            continue
                        df["date"] = pd.to_datetime(df[date_column])
                        raw_list.append(df[["pnl", "trade", "date"]])
                    if raw_list:
                        combined_raw = pd.concat(raw_list)
                        combined_raw = combined_raw.sort_values("date")
                        raw_metrics = calculate_metrics(combined_raw, "daily")
                    else:
                        raw_metrics = {}
                    
                    display_metrics = {}
                    display_metrics["Total Returns"] = aggregated_metrics["total_returns"]
                    display_metrics["Annualized Avg Return"] = aggregated_metrics["annualized_avg_return"]
                    display_metrics["Max Drawdown"] = aggregated_metrics["max_drawdown"]
                    display_metrics["Sharpe Ratio"] = aggregated_metrics["sharpe_ratio"]
                    display_metrics["Trades Per Interval"] = aggregated_metrics["trades_per_interval"]
                    display_metrics["Number of Trades"] = raw_metrics.get("num_of_trades", 0)
                    display_metrics["Average Trade Return"] = raw_metrics.get("average_trade_return", 0)
                    display_metrics["Win Rate"] = raw_metrics.get("win_rate", 0)
                    display_metrics["Average Winning Trade"] = raw_metrics.get("average_winning_trade", 0)
                    display_metrics["Average Losing Trade"] = raw_metrics.get("average_losing_trade", 0)
                    display_metrics["Profit Factor"] = raw_metrics.get("profit_factor", 0)
                    display_metrics["Rolling Sharpe"] = avg_rolling_sharpe
                    
                    formatted_metrics = {}
                    for k, v in display_metrics.items():
                        if k in ["Total Returns", "Annualized Avg Return", "Max Drawdown", "Trades Per Interval", "Win Rate"]:
                            formatted_metrics[k] = to_percentage(v)
                        elif k in ["Sharpe Ratio", "Rolling Sharpe"]:
                            formatted_metrics[k] = f"{v:.2f}"
                        elif k in ["Average Trade Return", "Average Winning Trade", "Average Losing Trade", "Profit Factor"]:
                            formatted_metrics[k] = f"{v:.2f}"
                        elif k == "Number of Trades":
                            formatted_metrics[k] = f"{v}"
                        else:
                            formatted_metrics[k] = str(v)
                    
                    cols = st.columns(3)
                    i = 0
                    for metric, val in formatted_metrics.items():
                        cols[i % 3].metric(metric, val)
                        i += 1
                    
                    metrics_json = json.dumps(formatted_metrics, indent=2)
                    st.download_button("Export Performance Metrics (JSON)", data=metrics_json, file_name="portfolio_metrics.json", mime="application/json")
                    
                    portfolio_equity = portfolio_daily_pnl.cumsum()
                    fig_port = go.Figure()
                    fig_port.add_trace(go.Scatter(x=portfolio_equity.index, y=portfolio_equity, mode='lines', name="Portfolio Equity Curve"))
                    fig_port.update_layout(title="Cumulative Portfolio Profit Over Time", xaxis_title="Time", yaxis_title="Cumulative PnL")
                    st.plotly_chart(fig_port, use_container_width=True)
                    
                    st.write("### Monthly Performance")
                    monthly_portfolio_return = portfolio_daily_pnl.resample('ME').sum()
                    monthly_portfolio_df = monthly_portfolio_return.to_frame(name='Return')
                    monthly_portfolio_df['Year'] = monthly_portfolio_df.index.year
                    monthly_portfolio_df['Month'] = monthly_portfolio_df.index.strftime('%b')
                    pivot_table_portfolio = monthly_portfolio_df.pivot(index='Year', columns='Month', values='Return')
                    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    pivot_table_portfolio = pivot_table_portfolio.reindex(columns=month_order)
                    ytd_series_portfolio = portfolio_daily_pnl.resample('YE').sum()
                    ytd_series_portfolio.index = ytd_series_portfolio.index.year
                    pivot_table_portfolio["YTD"] = ytd_series_portfolio
                    styled_table_portfolio = pivot_table_portfolio.style.format("{:.2f}").map(color_negative_positive)
                    st.dataframe(styled_table_portfolio, use_container_width=True)
            else:
                st.info("Select at least one strategy to view portfolio performance.")
        
        # --- Correlation Tab ---
        with tab3:
            st.header("📊 Strategy Correlation")
            selected_corr = st.multiselect("Select strategies for correlation analysis:", options=all_strategy_names)
            if len(selected_corr) < 2:
                st.info("Please select at least two strategies to view correlation.")
            else:
                candidate_columns = ['Entry Date', 'Exit Date', 'Trade Date', 'Date/Time', 'timestamp', 'datetime', 'date']
                daily_series_list = []
                for strategy_name in selected_corr:
                    df = strategies[strategy_name]
                    date_column = next((col for col in df.columns if col in candidate_columns), None)
                    if date_column is None:
                        st.warning(f"Strategy {strategy_name} does not have a valid date column and will be skipped.")
                        continue
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.sort_values(date_column)
                    daily_pnl = df.set_index(date_column)['pnl'].resample('D').sum()
                    daily_pnl.name = strategy_name
                    daily_series_list.append(daily_pnl)
                if len(daily_series_list) < 2:
                    st.error("Not enough valid strategies for correlation analysis.")
                else:
                    pnl_df = pd.concat(daily_series_list, axis=1)
                    pnl_df = pnl_df.fillna(0)
                    corr_matrix = pnl_df.corr()
                    custom_color_scale = [(0, "red"), (0.5, "white"), (1, "darkblue")]
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        aspect="auto",
                        title="Correlation Heatmap",
                        color_continuous_scale=custom_color_scale,
                        zmin=-1,
                        zmax=1
                    )
                    fig_corr.update_xaxes(tickangle=90)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.write("### Diversified Portfolio Optimization")
                    threshold = st.slider("Set correlation threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
                    if st.button("Run Portfolio Optimization"):
                        selected_strategies = select_diversified_strategies(corr_matrix, threshold)
                        eliminated = [s for s in corr_matrix.index if s not in selected_strategies]
                        st.write(f"**Selected Strategies (total: {len(selected_strategies)}):** {selected_strategies}")
                        st.write(f"**Eliminated Strategies:** {eliminated}")
                        
                        if len(selected_strategies) > 1:
                            diversified_corr = corr_matrix.loc[selected_strategies, selected_strategies]
                            fig_div = px.imshow(
                                diversified_corr,
                                text_auto=".2f",
                                aspect="auto",
                                title="Diversified Strategies Correlation Heatmap",
                                color_continuous_scale=custom_color_scale,
                                zmin=-1,
                                zmax=1
                            )
                            fig_div.update_xaxes(tickangle=90)
                            st.plotly_chart(fig_div, use_container_width=True)

if __name__ == "__main__":
    main()
