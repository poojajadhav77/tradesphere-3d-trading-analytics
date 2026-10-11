"""Input validation, cleaning, feature engineering and behavior-review flags."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "trade_id", "entry_time", "instrument", "strategy", "direction",
    "entry_price", "stop_loss_price", "take_profit_price", "planned_risk_amount",
    "session", "market_condition", "emotion_before", "followed_plan", "pnl",
]
CATEGORICAL_COLUMNS = [
    "instrument", "strategy", "direction", "session", "market_condition",
    "emotion_before", "followed_plan",
]
NUMERIC_COLUMNS = [
    "entry_price", "stop_loss_price", "take_profit_price", "planned_risk_amount", "pnl",
]


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean a trade journal and add derived fields used by the dashboard/models.

    Required user CSV columns are listed in REQUIRED_COLUMNS. `exit_time` and
    `quantity` are optional. Monetary P&L is read as supplied; it is not inferred
    from prices because contract sizes and account-currency rules vary by broker.
    """
    if raw is None or raw.empty:
        raise ValueError("The uploaded dataset is empty. Add at least one trade row.")

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing these required columns: " + ", ".join(missing)
            + ". Use data/sample_trades.csv as the template, keeping the column names."
        )

    original_count = len(df)
    # Normalize blank strings and whitespace.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    else:
        df["exit_time"] = pd.NaT

    # Required timestamps and P&L cannot be safely imputed, so discard those rows.
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=["entry_time", "pnl"]).copy()
    if df.empty:
        raise ValueError("No valid rows remain after checking entry_time and pnl. Check your CSV values.")

    # Keep the first instance of a duplicate trade_id. Fill blank IDs for remaining rows.
    blank_ids = df["trade_id"].isna() | (df["trade_id"].astype(str).str.strip() == "")
    df.loc[blank_ids, "trade_id"] = [f"AUTO{i:05d}" for i in range(1, int(blank_ids.sum()) + 1)]
    duplicate_count = int(df["trade_id"].duplicated().sum())
    df = df.drop_duplicates(subset=["trade_id"], keep="first").copy()

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "Unknown"

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(0.0 if pd.isna(median) else float(median))

    if "quantity" not in df.columns:
        df["quantity"] = np.nan
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    quantity_median = df["quantity"].median()
    df["quantity"] = df["quantity"].fillna(0.0 if pd.isna(quantity_median) else float(quantity_median))

    df["planned_risk_amount"] = df["planned_risk_amount"].abs()
    # Avoid division by zero when traders have not entered a risk amount.
    df["risk_reward_ratio"] = (
        (df["take_profit_price"] - df["entry_price"]).abs()
        / (df["entry_price"] - df["stop_loss_price"]).abs().replace(0, np.nan)
    )
    ratio_median = df["risk_reward_ratio"].replace([np.inf, -np.inf], np.nan).median()
    df["risk_reward_ratio"] = df["risk_reward_ratio"].replace([np.inf, -np.inf], np.nan)
    df["risk_reward_ratio"] = df["risk_reward_ratio"].fillna(1.0 if pd.isna(ratio_median) else float(ratio_median))
    df["r_multiple"] = np.where(
        df["planned_risk_amount"] > 0,
        df["pnl"] / df["planned_risk_amount"].replace(0, np.nan),
        np.nan,
    )
    r_median = df["r_multiple"].replace([np.inf, -np.inf], np.nan).median()
    df["r_multiple"] = df["r_multiple"].replace([np.inf, -np.inf], np.nan).fillna(0.0 if pd.isna(r_median) else float(r_median))

    df["outcome"] = np.select([df["pnl"] > 0, df["pnl"] < 0], ["Win", "Loss"], default="Breakeven")
    df["trade_date"] = df["entry_time"].dt.date.astype(str)
    df["weekday"] = df["entry_time"].dt.day_name()
    df["month"] = df["entry_time"].dt.to_period("M").astype(str)
    df = df.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)

    if df["exit_time"].notna().any():
        minutes = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60
        df["holding_minutes"] = minutes.where(minutes >= 0)
    else:
        df["holding_minutes"] = np.nan
    hold_median = df["holding_minutes"].median()
    df["holding_minutes"] = df["holding_minutes"].fillna(0.0 if pd.isna(hold_median) else float(hold_median))

    df["daily_trade_count"] = df.groupby("trade_date")["trade_id"].transform("count")
    df["previous_trade_outcome"] = df["outcome"].shift(1).fillna("First trade")
    df["after_loss"] = df["previous_trade_outcome"].eq("Loss")
    df["high_daily_frequency"] = df["daily_trade_count"] >= 5

    flags = []
    for row in df.itertuples(index=False):
        row_flags = []
        if row.daily_trade_count >= 5:
            row_flags.append("High daily trade count (review)")
        if row.followed_plan.lower() in {"no", "false", "0", "not followed"}:
            row_flags.append("Plan not followed (review)")
        if row.after_loss and row.emotion_before.lower() in {"frustrated", "fomo", "angry", "revenge"}:
            row_flags.append("Post-loss emotion (review)")
        flags.append("; ".join(row_flags) if row_flags else "")
    df["behavior_review_flags"] = flags

    report = {
        "input_rows": int(original_count),
        "clean_rows": int(len(df)),
        "removed_rows": int(original_count - len(df)),
        "duplicate_trade_ids_removed": duplicate_count,
    }
    return df, report
