import re
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import networkx as nx
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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

def check_nan_rows(df, threshold=0.03):
    """
    Check if the percentage of rows with any NaN values exceeds the threshold.
    Returns a tuple (exceeds_threshold, ratio) where:
      - exceeds_threshold: True if missing rows percentage > threshold.
      - ratio: the fraction of rows with NaN values.
    """
    total_rows = len(df)
    if total_rows == 0:
        return False, 0.0
    nan_rows = df.isna().any(axis=1).sum()
    ratio = nan_rows / total_rows
    return ratio > threshold, ratio

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def color_negative_positive(val):
    try:
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
    except Exception:
        return ''

def select_diversified_strategies_graph(corr_matrix, threshold=0.65):
    """
    Build a graph where each node is a strategy.
    Add an edge between two strategies if their correlation exceeds the threshold.
    Then, return an approximate maximum independent set.
    """
    strategies = list(corr_matrix.index)
    G = nx.Graph()
    G.add_nodes_from(strategies)
    
    # Add edges for high correlations
    for i in strategies:
        for j in strategies:
            if i != j and corr_matrix.loc[i, j] > threshold:
                G.add_edge(i, j)
    
    # Approximate maximum independent set: strategies that are not connected by high correlation.
    independent_set = nx.algorithms.approximation.maximum_independent_set(G)
    return list(independent_set), G

def visualize_network(G, independent_set):
    """
    Create a Plotly visualization of the network graph.
    Nodes in the independent_set are colored green and the eliminated ones are red.
    """
    pos = nx.spring_layout(G, seed=42)  # fixed seed for consistency
    # Extract edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node coordinates and styling
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in independent_set:
            node_color.append('green')
        else:
            node_color.append('red')
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=20,
            line_width=2
        ),
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Network Graph of Strategy Correlations",
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

# Helper function to save the uploaded file to a local folder.
def save_uploaded_file(uploaded_file):
    folder = "alpha"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("MMM Portfolio")
    
    # Ensure the 'alpha' folder exists for saving and loading CSV files.
    alpha_folder = "alpha"
    if not os.path.exists(alpha_folder):
        os.makedirs(alpha_folder)
    
    # Define the desired columns for the metrics table exactly as specified.
    desired_columns = [
        "alpha_id", "custom_id", "alpha_formula", "manual_alpha_formula", "data_preprocessing",
        "preprocessing_window", "model", "entry_exit_logic", "data_asset", "trade_asset",
        "Alpha Details & Logic", "alpha remarks", "backtested_period", "timeframe",
        "rolling_window_1", "rolling_window_2", "long_entry_threshold", "long_exit_threshold",
        "short_entry_threshold", "short_exit_threshold", "SR", "CR", "MDD", "AR", "trade_numbers",
        "datasource_structure", "shift_backtest_candle_minute"
    ]
    
    # Load previously saved metrics from CSV, if available.
    metrics_file_path = "metrics.csv"
    if os.path.exists(metrics_file_path):
        saved_metrics_df = pd.read_csv(metrics_file_path)
    else:
        saved_metrics_df = pd.DataFrame(columns=desired_columns)
    
    # Initialize the strategies dictionary.
    strategies = {}

    # --------------------------------------------------
    # Auto load CSV files from the alpha folder
    # --------------------------------------------------
    for file in os.listdir(alpha_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(alpha_folder, file)
            try:
                df = pd.read_csv(file_path)
                file_basename = os.path.splitext(file)[0]
                strategies[file_basename] = df
            except Exception as e:
                st.error(f"Error loading file {file}: {e}")
    
    # Allow multiple CSV file uploads for strategy data.
    uploaded_files = st.sidebar.file_uploader("Upload your CSV files", type=["csv"], accept_multiple_files=True)
    
    new_metrics_records = []
    # Candidate columns to detect dates.
    candidate_columns = ['Entry Date', 'Exit Date', 'Trade Date', 'Date/Time', 'timestamp', 'datetime', 'date']
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save the uploaded file to the 'alpha' folder.
            save_uploaded_file(uploaded_file)
            # Reset the file pointer before reading.
            uploaded_file.seek(0)
            try:
                df = load_csv(uploaded_file)
                
                # Check for NaN rows exceeding the 3% threshold.
                exceeds_nan, nan_ratio = check_nan_rows(df, threshold=0.03)
                if exceeds_nan:
                    st.error(f"File {uploaded_file.name} has {nan_ratio*100:.2f}% missing rows, which exceeds the 3% threshold. This file will be skipped.")
                    continue
                
                if 'pnl' not in df.columns or 'trade' not in df.columns:
                    st.error(f"File {uploaded_file.name} does not contain required 'pnl' and 'trade' columns.")
                    continue
                # Use the basename for consistency.
                file_basename = os.path.splitext(uploaded_file.name)[0]
                strategies[file_basename] = df
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
        
        # Compute metrics for each uploaded file.
        for file_name, df in strategies.items():
            date_column = next((col for col in df.columns if col in candidate_columns), None)
            if date_column:
                # Detect time interval and calculate backtested period.
                detected_interval = detect_time_interval(df, date_column) or 'daily'
                df[date_column] = pd.to_datetime(df[date_column])
                start_date = df[date_column].min()
                end_date = df[date_column].max()
                backtested_period = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            else:
                detected_interval = 'daily'
                backtested_period = ""
            metrics = calculate_metrics(df, detected_interval)
            sharpe_ratio = metrics["sharpe_ratio"]
            annualized_avg_return = metrics["annualized_avg_return"]
            max_drawdown = metrics["max_drawdown"]
            num_of_trades = metrics["num_of_trades"]
            if max_drawdown != 0:
                calmar_ratio = annualized_avg_return / abs(max_drawdown)
            else:
                calmar_ratio = np.nan
            record = {
                "custom_id": file_name,
                "alpha_formula": "",
                "manual_alpha_formula": "",
                "data_preprocessing": "",
                "preprocessing_window": "",
                "model": "",
                "entry_exit_logic": "",
                "data_asset": "",
                "trade_asset": "",
                "Alpha Details & Logic": "",
                "alpha remarks": "",
                "backtested_period": backtested_period,
                "timeframe": detected_interval,
                "rolling_window_1": "",
                "rolling_window_2": "",
                "long_entry_threshold": "",
                "long_exit_threshold": "",
                "short_entry_threshold": "",
                "short_exit_threshold": "",
                "SR": f"{sharpe_ratio:.2f}",
                "CR": f"{calmar_ratio:.2f}" if not np.isnan(calmar_ratio) else "N/A",
                "MDD": to_percentage(max_drawdown),
                "AR": to_percentage(annualized_avg_return),
                "trade_numbers": num_of_trades,
                "datasource_structure": "",
                "shift_backtest_candle_minute": ""
            }
            new_metrics_records.append(record)
    
    # Merge new metrics with previously saved metrics using 'custom_id' as unique key.
    if new_metrics_records:
        new_metrics_df = pd.DataFrame(new_metrics_records)
        combined_df = pd.concat([saved_metrics_df, new_metrics_df]).drop_duplicates(subset=['custom_id'], keep='last')
    else:
        combined_df = saved_metrics_df.copy()
    
    # Reset index and reassign alpha_ids.
    combined_df.reset_index(drop=True, inplace=True)
    if "alpha_id" in combined_df.columns:
        combined_df.drop(columns=["alpha_id"], inplace=True)
    alpha_ids = ["TURTLE999_{:04d}".format(i) for i in range(1, len(combined_df) + 1)]
    combined_df.insert(0, "alpha_id", alpha_ids)
    metrics_df = combined_df[desired_columns]
    
    # ------------------------------
    # Display and Automatically Update the Metrics Table
    # ------------------------------
    st.subheader("📄 Files Metrics Overview")
    gb = GridOptionsBuilder.from_dataframe(metrics_df)
    gb.configure_default_column(editable=True, resizable=True, minWidth=150)
    # Configure the data_asset and trade_asset columns as dropdowns with BTC and ETH options.
    gb.configure_column("data_asset", cellEditor="agSelectCellEditor", cellEditorParams={"values": ["BTC", "ETH"]})
    gb.configure_column("trade_asset", cellEditor="agSelectCellEditor", cellEditorParams={"values": ["BTC", "ETH"]})
    gb.configure_selection("single", use_checkbox=False)
    gb.configure_grid_options(
        onGridReady=""" 
        function(params) {
            var allColumnIds = params.columnApi.getAllColumns().map(function(col) { return col.colId; });
            params.columnApi.autoSizeColumns(allColumnIds, false);
        }
        """
    )
    gridOptions = gb.build()
    ag_response = AgGrid(
        metrics_df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        height=300,
        fit_columns_on_grid_load=True,
    )
    # Auto-update the CSV file on every change in the table.
    updated_metrics = ag_response['data']
    updated_df = pd.DataFrame(updated_metrics)
    updated_df.to_csv(metrics_file_path, index=False)
    st.info("Metrics updated automatically to metrics.csv.")
    
    # Add a download button for the current metrics CSV.
    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV File",
        data=csv_data,
        file_name="metrics.csv",
        mime="text/csv"
    )
    
    # ------------------------------
    # Main Tabs
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["Individual Strategy", "Portfolio", "Correlation"])
    
    # --- Individual Strategy Tab ---
    with tab1:
        st.header("📊 Individual Strategy Performance")
        all_strategy_names = list(strategies.keys())
        selected_strategy = st.selectbox("Select a strategy to view its performance:", options=["None"] + all_strategy_names)
        if selected_strategy != "None":
            strategy_trades = strategies[selected_strategy]
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
        all_strategy_names = list(strategies.keys())
        selected_portfolio = st.multiselect("Select strategies to include in the portfolio:", options=all_strategy_names)
        if selected_portfolio:
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
        all_strategy_names = list(strategies.keys())
        select_all_corr = st.checkbox("Select All Strategies for Correlation", key="select_all_corr")
        if select_all_corr:
            selected_corr = st.multiselect(
                "Select strategies for correlation analysis:", 
                options=all_strategy_names, 
                default=all_strategy_names,
                key="selected_corr"
            )
        else:
            selected_corr = st.multiselect(
                "Select strategies for correlation analysis:", 
                options=all_strategy_names,
                key="selected_corr"
            )
            
        if len(selected_corr) < 2:
            st.info("Please select at least two strategies to view correlation.")
        else:
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
                
                st.write("### Diversified Portfolio Optimization (Graph-based)")
                threshold = st.slider("Set correlation threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01, key="opt_threshold")
                
                if st.button("Run Portfolio Optimization", key="run_opt_button"):
                    selected_strategies, G = select_diversified_strategies_graph(corr_matrix, threshold)
                    eliminated = [s for s in corr_matrix.index if s not in selected_strategies]
                    st.session_state.optimization_result = {
                        "selected_strategies": selected_strategies,
                        "eliminated": eliminated,
                        "G": G,
                    }
                
                if "optimization_result" in st.session_state:
                    opt_result = st.session_state.optimization_result
                    selected_strategies = opt_result["selected_strategies"]
                    eliminated = opt_result["eliminated"]
                    G = opt_result["G"]
                    
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
                    
                    fig_network = visualize_network(G, selected_strategies)
                    st.plotly_chart(fig_network, use_container_width=True)
    
if __name__ == "__main__":
    main()
