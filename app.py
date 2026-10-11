from __future__ import annotations
from pathlib import Path
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import prepare_data
from src.analytics import (
    performance_summary, run_clustering, run_classification,
    run_association_rules, run_anomaly_detection,
)
from src.warehouse import DEFAULT_DB, load_warehouse, query_table

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_trades.csv"

st.set_page_config(page_title="Trading Performance & Behavior Mining", page_icon="📊", layout="wide")
st.title("📊 Smart Trading Performance & Behavior Mining System")
st.caption("A Data Mining + Data Warehousing mini-project for analyzing a trade journal. It does not place trades or provide financial advice.")

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload your trade-journal CSV", type=["csv"], help="Keep the required column names shown in the README. You can start with the bundled sample CSV.")
    st.divider()
    page = st.radio("Navigate", ["Overview dashboard", "Behavior mining", "Trade explorer", "Data warehouse", "How to use"], index=0)
    st.divider()
    st.caption("Demo data is synthetic and only illustrates the workflow. Replace it with your own journal export for meaningful personal analysis.")

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes | None, file_name: str | None):
    if file_bytes is None:
        raw = pd.read_csv(SAMPLE_PATH)
        source_name = "Bundled synthetic sample"
    else:
        import io
        raw = pd.read_csv(io.BytesIO(file_bytes))
        source_name = file_name or "Uploaded CSV"
    clean, cleaning_report = prepare_data(raw)
    return clean, cleaning_report, source_name

@st.cache_data(show_spinner=False)
def cached_cluster(df: pd.DataFrame) -> pd.DataFrame:
    return run_clustering(df)

@st.cache_data(show_spinner=False)
def cached_classification(df: pd.DataFrame) -> dict:
    return run_classification(df)

@st.cache_data(show_spinner=False)
def cached_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    return run_anomaly_detection(df)

@st.cache_data(show_spinner=False)
def cached_rules(df: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    return run_association_rules(df, min_support=min_support, min_confidence=min_confidence)

try:
    file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    file_name = uploaded_file.name if uploaded_file is not None else None
    df, cleaning_report, source_name = load_and_clean(file_bytes, file_name)
    warehouse_counts = load_warehouse(df)
except Exception as exc:
    st.error(f"Could not load this dataset: {exc}")
    st.info("Start with data/sample_trades.csv and preserve its column headings when creating your own file.")
    st.stop()

st.sidebar.success(f"Loaded {len(df):,} cleaned trades")

if page == "Overview dashboard":
    st.subheader("Trading performance at a glance")
    st.caption(f"Source: {source_name} · Input rows: {cleaning_report['input_rows']:,} · Removed rows: {cleaning_report['removed_rows']:,}")
    m = performance_summary(df)
    cols = st.columns(4)
    cols[0].metric("Total trades", f"{m['total_trades']:,}")
    cols[1].metric("Win rate", f"{m['win_rate']:.1f}%", help="Wins / (wins + losses); breakeven trades are excluded.")
    cols[2].metric("Net P&L", f"{m['net_pnl']:,.2f}", help="Uses the P&L values provided in your CSV; currency depends on your source file.")
    cols[3].metric("Max drawdown", f"{m['max_drawdown']:,.2f}", help="Largest peak-to-trough decline in cumulative P&L for the imported sequence.")
    cols = st.columns(4)
    cols[0].metric("Average R multiple", f"{m['average_r']:.2f}R")
    cols[1].metric("Average winning trade", f"{m['average_win']:,.2f}")
    cols[2].metric("Plan-following rate", f"{m['plan_follow_rate']:.1f}%")
    cols[3].metric("Trades flagged for review", f"{m['review_flags']:,}")

    left, right = st.columns(2)
    with left:
        daily = df.groupby("trade_date", as_index=False)["pnl"].sum().sort_values("trade_date")
        daily["cumulative_pnl"] = daily["pnl"].cumsum()
        fig = px.line(daily, x="trade_date", y="cumulative_pnl", title="Cumulative P&L by trade date", markers=False)
        fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative P&L (CSV units)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_session = df.groupby("session", as_index=False).agg(net_pnl=("pnl", "sum"), trades=("trade_id", "count"))
        fig = px.bar(by_session, x="session", y="net_pnl", color="session", title="Net P&L by trading session", hover_data=["trades"])
        fig.update_layout(showlegend=False, xaxis_title="Session", yaxis_title="Net P&L (CSV units)")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        outcomes = df["outcome"].value_counts().rename_axis("outcome").reset_index(name="trades")
        fig = px.pie(outcomes, names="outcome", values="trades", title="Win / loss / breakeven distribution", hole=.35)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        monthly = df.groupby("month", as_index=False)["pnl"].sum().sort_values("month")
        fig = px.bar(monthly, x="month", y="pnl", title="Monthly net P&L")
        fig.update_layout(xaxis_title="Month", yaxis_title="Net P&L (CSV units)")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data cleaning report"):
        st.write(cleaning_report)
        st.write("Data has been cleaned and loaded into the local SQLite star-schema warehouse.")

elif page == "Behavior mining":
    st.subheader("Data mining models")
    st.write("The outputs describe patterns in the imported journal; associations and model predictions are not proof of cause and effect.")

    clustered = cached_cluster(df)
    st.markdown("### 1. Clustering — K-Means")
    st.write("Groups trades into Poor, Average, and Good performance clusters using P&L, R multiple, planned risk, risk/reward ratio, and holding time.")
    cluster_summary = clustered.groupby("performance_cluster", as_index=False).agg(trades=("trade_id", "count"), mean_pnl=("pnl", "mean"), mean_r=("r_multiple", "mean"))
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.scatter(clustered, x="r_multiple", y="pnl", color="performance_cluster", hover_data=["trade_id", "session", "strategy", "outcome"], title="Trade clusters by P&L and R multiple")
        fig.update_layout(xaxis_title="R multiple", yaxis_title="P&L (CSV units)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
        st.caption("Cluster names are assigned by average R multiple and are relative to the dataset, not universal labels.")

    st.markdown("### 2. Classification — Random Forest")
    st.write("Uses the earliest 80% of chronological non-breakeven trades for training and the latest 20% for testing. Inputs are journal fields that could be known before entry; P&L and outcome are not used as model inputs.")
    classification = cached_classification(df)
    if not classification["available"]:
        st.warning(classification["reason"])
    else:
        cols = st.columns(4)
        cols[0].metric("Test accuracy", f"{classification['accuracy']:.1%}")
        cols[1].metric("Win precision", f"{classification['precision']:.1%}")
        cols[2].metric("Win recall", f"{classification['recall']:.1%}")
        cols[3].metric("Win F1-score", f"{classification['f1']:.1%}")
        cm = pd.DataFrame(classification["confusion_matrix"], index=["Actual Loss", "Actual Win"], columns=["Predicted Loss", "Predicted Win"])
        st.write(f"Train rows: {classification['train_rows']} · Test rows: {classification['test_rows']}")
        st.dataframe(cm, use_container_width=True)
        st.caption("Synthetic demo results are not evidence of real trading predictability. Performance on a real journal may be low or unstable.")

    st.markdown("### 3. Association Rule Mining — Apriori")
    left, right = st.columns(2)
    with left:
        min_support = st.slider("Minimum support", min_value=0.01, max_value=0.20, value=0.04, step=0.01, help="Minimum fraction of trades that must contain a pattern.")
    with right:
        min_confidence = st.slider("Minimum confidence", min_value=0.30, max_value=0.95, value=0.50, step=0.05, help="Fraction of trades with the antecedent that also have the consequent.")
    try:
        rules = cached_rules(df, float(min_support), float(min_confidence))
        if rules.empty:
            st.info("No rules met these thresholds. Lower minimum support/confidence or load more trades.")
        else:
            rules_display = rules.copy()
            for c in ["support", "confidence"]:
                rules_display[c] = rules_display[c].map(lambda x: f"{x:.1%}")
            rules_display["lift"] = rules_display["lift"].map(lambda x: f"{x:.2f}")
            st.dataframe(rules_display.head(30), use_container_width=True, hide_index=True)
            st.caption("Lift above 1 indicates the consequent occurred more often with the antecedent than in the dataset overall. This does not establish causation.")
    except Exception as exc:
        st.warning(f"Association rules could not be generated: {exc}")

    st.markdown("### 4. Outlier / anomaly detection — Isolation Forest")
    anomaly_df = cached_anomalies(df)
    anomaly_rows = anomaly_df[anomaly_df["anomaly"] == "Potential outlier - review"]
    st.write(f"Potential outliers flagged for review: **{len(anomaly_rows)}**")
    st.caption("An outlier is not automatically a bad or improper trade. It simply differs from typical rows on selected numeric fields.")
    if not anomaly_rows.empty:
        st.dataframe(anomaly_rows[["trade_id", "entry_time", "instrument", "session", "pnl", "r_multiple", "daily_trade_count", "anomaly"]].head(100), use_container_width=True, hide_index=True)

    st.markdown("### Behavioral review checklist")
    flagged = df[df["behavior_review_flags"].astype(str).str.len() > 0]
    st.write(f"Trades with one or more review flags: **{len(flagged)}**")
    if not flagged.empty:
        st.dataframe(flagged[["trade_id", "entry_time", "session", "emotion_before", "followed_plan", "daily_trade_count", "previous_trade_outcome", "pnl", "behavior_review_flags"]].head(150), use_container_width=True, hide_index=True)
    st.caption("Flags are transparent rules for reflection (e.g. 5+ trades/day, plan not followed, or frustrated/FOMO after a loss). They cannot reliably diagnose revenge trading.")

elif page == "Trade explorer":
    st.subheader("Trade explorer")
    left, middle, right = st.columns(3)
    sessions = ["All"] + sorted(df["session"].dropna().astype(str).unique().tolist())
    outcomes = ["All"] + sorted(df["outcome"].unique().tolist())
    instruments = ["All"] + sorted(df["instrument"].dropna().astype(str).unique().tolist())
    chosen_session = left.selectbox("Session", sessions)
    chosen_outcome = middle.selectbox("Outcome", outcomes)
    chosen_instrument = right.selectbox("Instrument", instruments)
    view = df.copy()
    if chosen_session != "All":
        view = view[view["session"] == chosen_session]
    if chosen_outcome != "All":
        view = view[view["outcome"] == chosen_outcome]
    if chosen_instrument != "All":
        view = view[view["instrument"] == chosen_instrument]
    show_cols = ["trade_id", "entry_time", "instrument", "strategy", "direction", "session", "market_condition", "emotion_before", "followed_plan", "risk_reward_ratio", "pnl", "r_multiple", "outcome", "daily_trade_count", "behavior_review_flags"]
    st.write(f"Showing {len(view):,} trades")
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
    st.download_button("Download cleaned trade CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="cleaned_trade_journal.csv", mime="text/csv")

elif page == "Data warehouse":
    st.subheader("Data warehouse — star schema")
    st.write("The app refreshes a local SQLite database (`data/trading_warehouse.db`) from the currently selected CSV. The fact table stores trade measurements and foreign keys; dimensions describe date, session, instrument, strategy, and market condition.")
    st.code("""                  dim_date
                     |
 dim_session — fact_trade — dim_instrument
                     |
                dim_strategy
                     |
           dim_market_condition""", language="text")
    st.markdown("#### Warehouse row counts")
    counts = pd.DataFrame([{"table": table, "rows": count} for table, count in warehouse_counts.items()])
    st.dataframe(counts, use_container_width=True, hide_index=True)
    table_names = ["fact_trade", "dim_date", "dim_session", "dim_instrument", "dim_strategy", "dim_market_condition"]
    selected_table = st.selectbox("Preview a table", table_names)
    try:
        table_df = query_table(selected_table)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Unable to display table: {exc}")
    st.code(f"SQLite database file: {DEFAULT_DB}", language="text")

else:
    st.subheader("How to use this project")
    st.markdown("""
1. **Start with the bundled sample data.** It lets you explore every page without connecting a broker.
2. **Prepare your own CSV.** Keep the column names in `data/sample_trades.csv`. Each row should represent one closed trade.
3. **Upload the CSV** using the sidebar. The app cleans the fields and computes derived measures.
4. **Review the Overview dashboard** for win rate, net P&L, drawdown and session/month patterns.
5. **Open Behavior mining** to inspect clusters, held-out classification metrics, association rules, anomalies and behavior review flags.
6. **Open Data warehouse** to inspect the star-schema tables loaded into SQLite.

**Required CSV columns** (case-insensitive): `trade_id`, `entry_time`, `instrument`, `strategy`, `direction`, `entry_price`, `stop_loss_price`, `take_profit_price`, `planned_risk_amount`, `session`, `market_condition`, `emotion_before`, `followed_plan`, `pnl`.

`exit_time` and `quantity` are optional. `pnl` must be the realized P&L recorded in your journal. The app does not infer P&L from prices because instruments and broker contract sizes differ.

**Safety and honesty:** the sample data is synthetic. Model results depend on data quality and size. This is an educational analytics project, not a buy/sell signal generator, broker integration, or promise of profitable trading.
""")
    st.download_button("Download sample CSV template", data=pd.read_csv(SAMPLE_PATH).head(0).to_csv(index=False).encode("utf-8"), file_name="trade_journal_template.csv", mime="text/csv")

st.divider()
st.caption("Mini-project: Data Mining & Data Warehousing · Algorithms: K-Means, Random Forest, Apriori, Isolation Forest · Warehouse: SQLite star schema")
