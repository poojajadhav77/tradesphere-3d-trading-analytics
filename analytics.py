"""Data-mining workflows: performance metrics, clustering, classification, rules, anomalies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASS_CATEGORICAL = ["instrument", "strategy", "direction", "session", "market_condition", "emotion_before", "followed_plan"]
CLASS_NUMERIC = ["risk_reward_ratio"]


def performance_summary(df: pd.DataFrame) -> dict:
    ordered = df.sort_values("entry_time").copy()
    n = len(ordered)
    decisive = ordered[ordered["outcome"].isin(["Win", "Loss"])]
    wins = int((ordered["outcome"] == "Win").sum())
    losses = int((ordered["outcome"] == "Loss").sum())
    positive = ordered.loc[ordered["pnl"] > 0, "pnl"]
    negative = ordered.loc[ordered["pnl"] < 0, "pnl"]
    cumulative = ordered["pnl"].cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    max_drawdown = float(abs(drawdown.min())) if len(drawdown) else 0.0
    win_rate = (wins / len(decisive) * 100.0) if len(decisive) else 0.0
    return {
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "breakeven": int((ordered["outcome"] == "Breakeven").sum()),
        "win_rate": float(win_rate),
        "net_pnl": float(ordered["pnl"].sum()),
        "average_win": float(positive.mean()) if len(positive) else 0.0,
        "average_loss": float(negative.mean()) if len(negative) else 0.0,
        "average_r": float(ordered["r_multiple"].mean()) if n else 0.0,
        "max_drawdown": max_drawdown,
        "plan_follow_rate": float(ordered["followed_plan"].astype(str).str.lower().isin(["yes", "true", "1", "followed"]).mean() * 100),
        "review_flags": int(ordered["behavior_review_flags"].astype(str).str.len().gt(0).sum()),
        "decisive_trades": int(len(decisive)),
    }


def run_clustering(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    cols = ["pnl", "r_multiple", "risk_reward_ratio", "planned_risk_amount", "holding_minutes"]
    X = result[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(result) < 3:
        result["performance_cluster"] = "Not enough data"
        return result
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
    result["_cluster_num"] = labels
    means = result.groupby("_cluster_num")["r_multiple"].mean().sort_values()
    label_map = {int(means.index[0]): "Poor", int(means.index[1]): "Average", int(means.index[2]): "Good"}
    result["performance_cluster"] = result["_cluster_num"].map(label_map)
    return result.drop(columns=["_cluster_num"])


def run_classification(df: pd.DataFrame) -> dict:
    """Train on earlier trades and evaluate on later trades to reduce time leakage.

    Inputs are limited to journal fields potentially known before entry. P&L/outcome,
    post-trade duration, and derived behavior flags are deliberately excluded.
    """
    data = df.sort_values("entry_time").copy()
    data = data[data["outcome"].isin(["Win", "Loss"])].reset_index(drop=True)
    result = {"available": False, "reason": "", "accuracy": None, "precision": None, "recall": None, "f1": None}
    if len(data) < 30:
        result["reason"] = "At least 30 non-breakeven trades are needed for a basic train/test evaluation."
        return result
    y = data["outcome"].astype(str)
    split = int(len(data) * 0.8)
    split = min(max(split, 1), len(data) - 1)
    X = data[CLASS_CATEGORICAL + CLASS_NUMERIC].copy()
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2:
        result["reason"] = "The earlier 80% of rows contains only one outcome class. Add more varied, chronological trades."
        return result
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CLASS_CATEGORICAL),
            ("numeric", StandardScaler(), CLASS_NUMERIC),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("features", transformer),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = ["Loss", "Win"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    result.update({
        "available": True,
        "reason": "",
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, pos_label="Win", zero_division=0)),
        "recall": float(recall_score(y_test, pred, pos_label="Win", zero_division=0)),
        "f1": float(f1_score(y_test, pred, pos_label="Win", zero_division=0)),
        "confusion_matrix": cm,
        "confusion_labels": labels,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "report": classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True),
        "predictions": pd.DataFrame({"entry_time": data.loc[y_test.index, "entry_time"].values,
                                      "actual": y_test.values, "predicted": pred}),
    })
    return result


def run_association_rules(df: pd.DataFrame, min_support: float = 0.04, min_confidence: float = 0.50) -> pd.DataFrame:
    """Mine interpretable combinations that may co-occur with winning/losing trades."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        raise RuntimeError("mlxtend is not installed. Run: pip install -r requirements.txt") from exc
    data = df.copy()
    if len(data) < 10:
        return pd.DataFrame()
    rr_bucket = pd.cut(data["risk_reward_ratio"], bins=[-np.inf, 1.0, 2.0, np.inf], labels=["LowRR", "MediumRR", "HighRR"]).astype(str)
    daily = np.where(data["daily_trade_count"] >= 5, "HighDailyFrequency", "NormalDailyFrequency")
    # Keep item names compact and avoid empty/unknown values creating confusing rules.
    transactions = pd.DataFrame(index=data.index)
    item_columns = {
        "Session": data["session"],
        "Direction": data["direction"],
        "Strategy": data["strategy"],
        "Market": data["market_condition"],
        "Emotion": data["emotion_before"],
        "Plan": data["followed_plan"],
        "RiskReward": rr_bucket,
        "DailyFrequency": daily,
        "Outcome": data["outcome"],
    }
    for prefix, series in item_columns.items():
        item = prefix + "=" + series.astype(str)
        for value in item.unique():
            transactions[value] = item.eq(value)
    freq = apriori(transactions, min_support=min_support, use_colnames=True, max_len=3)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()
    rules = rules[rules["consequents"].apply(lambda s: any(str(x).startswith("Outcome=") for x in s))].copy()
    if rules.empty:
        return pd.DataFrame()
    rules["antecedents_text"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(map(str, s))))
    rules["consequent_text"] = rules["consequents"].apply(lambda s: " + ".join(sorted(map(str, s))))
    rules = rules.rename(columns={"support": "support", "confidence": "confidence", "lift": "lift"})
    return rules.sort_values(["lift", "confidence", "support"], ascending=False)[
        ["antecedents_text", "consequent_text", "support", "confidence", "lift"]
    ].reset_index(drop=True)


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    cols = ["pnl", "r_multiple", "risk_reward_ratio", "planned_risk_amount", "daily_trade_count"]
    X = result[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(result) < 20:
        result["anomaly"] = "Not evaluated (need 20+ trades)"
        return result
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    predictions = model.fit_predict(StandardScaler().fit_transform(X))
    result["anomaly"] = np.where(predictions == -1, "Potential outlier - review", "Typical range")
    return result
