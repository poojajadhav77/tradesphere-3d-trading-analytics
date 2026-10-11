"""SQLite star-schema warehouse for the trading-journal project."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "trading_warehouse.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date TEXT UNIQUE NOT NULL,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    weekday TEXT
);
CREATE TABLE IF NOT EXISTS dim_session (
    session_key INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_instrument (
    instrument_key INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_strategy (
    strategy_key INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_market_condition (
    market_condition_key INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_trade (
    trade_key INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    date_key INTEGER NOT NULL,
    session_key INTEGER NOT NULL,
    instrument_key INTEGER NOT NULL,
    strategy_key INTEGER NOT NULL,
    market_condition_key INTEGER NOT NULL,
    direction TEXT,
    entry_time TEXT,
    entry_price REAL,
    stop_loss_price REAL,
    take_profit_price REAL,
    planned_risk_amount REAL,
    risk_reward_ratio REAL,
    pnl REAL,
    r_multiple REAL,
    outcome TEXT,
    emotion_before TEXT,
    followed_plan TEXT,
    FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY(session_key) REFERENCES dim_session(session_key),
    FOREIGN KEY(instrument_key) REFERENCES dim_instrument(instrument_key),
    FOREIGN KEY(strategy_key) REFERENCES dim_strategy(strategy_key),
    FOREIGN KEY(market_condition_key) REFERENCES dim_market_condition(market_condition_key)
);
"""


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def load_warehouse(df: pd.DataFrame, db_path: str | Path = DEFAULT_DB) -> dict:
    """Refresh dimension/fact tables from the cleaned app dataset."""
    conn = connect(db_path)
    try:
        # A full refresh keeps this starter project simple and avoids stale uploaded rows.
        conn.execute("DELETE FROM fact_trade")
        conn.execute("DELETE FROM dim_date")
        conn.execute("DELETE FROM dim_session")
        conn.execute("DELETE FROM dim_instrument")
        conn.execute("DELETE FROM dim_strategy")
        conn.execute("DELETE FROM dim_market_condition")

        dim_dates = df[["entry_time", "trade_date", "weekday"]].drop_duplicates("trade_date")
        for row in dim_dates.itertuples(index=False):
            d = pd.Timestamp(row.entry_time).date()
            key = int(d.strftime("%Y%m%d"))
            conn.execute(
                "INSERT INTO dim_date(date_key, full_date, year, month, day, weekday) VALUES (?, ?, ?, ?, ?, ?)",
                (key, d.isoformat(), d.year, d.month, d.day, str(row.weekday)),
            )
        for table, col in [
            ("dim_session", "session"),
            ("dim_instrument", "instrument"),
            ("dim_strategy", "strategy"),
            ("dim_market_condition", "market_condition"),
        ]:
            key_col = {
                "dim_session": "session_name",
                "dim_instrument": "instrument_name",
                "dim_strategy": "strategy_name",
                "dim_market_condition": "condition_name",
            }[table]
            for value in sorted(df[col].fillna("Unknown").astype(str).unique()):
                conn.execute(f"INSERT INTO {table}({key_col}) VALUES (?)", (value,))

        lookups = {}
        for table, label_col, key_col in [
            ("dim_session", "session_name", "session_key"),
            ("dim_instrument", "instrument_name", "instrument_key"),
            ("dim_strategy", "strategy_name", "strategy_key"),
            ("dim_market_condition", "condition_name", "market_condition_key"),
        ]:
            lookups[table] = dict(conn.execute(f"SELECT {label_col}, {key_col} FROM {table}").fetchall())

        for row in df.itertuples(index=False):
            d = pd.Timestamp(row.entry_time).date()
            date_key = int(d.strftime("%Y%m%d"))
            conn.execute(
                """INSERT INTO fact_trade(
                    trade_id, date_key, session_key, instrument_key, strategy_key,
                    market_condition_key, direction, entry_time, entry_price,
                    stop_loss_price, take_profit_price, planned_risk_amount,
                    risk_reward_ratio, pnl, r_multiple, outcome, emotion_before, followed_plan
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(row.trade_id), date_key,
                    lookups["dim_session"].get(str(row.session), 1),
                    lookups["dim_instrument"].get(str(row.instrument), 1),
                    lookups["dim_strategy"].get(str(row.strategy), 1),
                    lookups["dim_market_condition"].get(str(row.market_condition), 1),
                    str(row.direction), pd.Timestamp(row.entry_time).isoformat(sep=" "),
                    float(row.entry_price), float(row.stop_loss_price), float(row.take_profit_price),
                    float(row.planned_risk_amount), float(row.risk_reward_ratio), float(row.pnl),
                    float(row.r_multiple), str(row.outcome), str(row.emotion_before), str(row.followed_plan),
                ),
            )
        conn.commit()
        counts = {}
        for table in ["fact_trade", "dim_date", "dim_session", "dim_instrument", "dim_strategy", "dim_market_condition"]:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return counts
    finally:
        conn.close()


def query_table(table: str, db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    allowed = {"fact_trade", "dim_date", "dim_session", "dim_instrument", "dim_strategy", "dim_market_condition"}
    if table not in allowed:
        raise ValueError(f"Unsupported warehouse table: {table}")
    conn = connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table} LIMIT 500", conn)
    finally:
        conn.close()
