"""Generate clearly synthetic trade-journal rows for demonstrating the project.

These are not real market prices, brokerage records, or trading recommendations.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sample_trades.csv"


def generate_trades(seed: int = 42, days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    instruments = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD"]
    strategies = ["Breakout", "Pullback", "Trend Following", "Range Reversal"]
    sessions = ["Asian", "London", "New York"]
    markets = ["Trending", "Ranging", "High Volatility", "News/Event"]
    emotions = ["Calm", "Confident", "FOMO", "Frustrated"]
    records = []
    trade_n = 1

    for day_num in range(days):
        day = start + pd.Timedelta(days=day_num)
        if day.dayofweek >= 5:
            continue
        # Deliberately vary trade counts so the dashboard can demonstrate overtrading flags.
        n_trades = int(rng.choice([1, 2, 3, 4, 5, 6, 7], p=[.12, .20, .22, .18, .13, .09, .06]))
        prior_loss_today = False
        day_rows = []
        for _ in range(n_trades):
            session = str(rng.choice(sessions, p=[.22, .40, .38]))
            session_hour = {"Asian": int(rng.integers(1, 8)),
                            "London": int(rng.integers(8, 14)),
                            "New York": int(rng.integers(14, 22))}[session]
            minute = int(rng.integers(0, 60))
            entry_time = day + pd.Timedelta(hours=session_hour, minutes=minute)
            instrument = str(rng.choice(instruments, p=[.52, .18, .18, .12]))
            strategy = str(rng.choice(strategies))
            direction = str(rng.choice(["Buy", "Sell"]))
            market = str(rng.choice(markets, p=[.38, .30, .22, .10]))

            # Behavioral variables deliberately influence synthetic outcomes for demo purposes.
            if prior_loss_today and rng.random() < .45:
                emotion = str(rng.choice(["Frustrated", "FOMO", "Calm"], p=[.42, .33, .25]))
            else:
                emotion = str(rng.choice(emotions, p=[.52, .23, .14, .11]))
            followed = "Yes" if rng.random() < (0.89 if emotion in ["Calm", "Confident"] else 0.48) else "No"
            rr = float(np.clip(rng.normal(1.8, .65), .5, 4.0))
            risk_amt = float(rng.choice([100, 150, 200, 250, 300]))

            # Generate a price scale only for display. P&L is a synthetic journal value,
            # not calculated from instrument contract specifications.
            base = {"XAUUSD": 2350, "EURUSD": 1.08, "GBPUSD": 1.27, "BTCUSD": 95000}[instrument]
            entry_price = float(base * (1 + rng.normal(0, .001)))
            stop_distance = float(base * (0.0012 if instrument != "BTCUSD" else .008) * rng.uniform(.7, 1.4))
            take_distance = stop_distance * rr
            if direction == "Buy":
                stop = entry_price - stop_distance
                take = entry_price + take_distance
            else:
                stop = entry_price + stop_distance
                take = entry_price - take_distance

            # Baseline win probability plus modest synthetic behavior/market effects.
            win_prob = .49
            win_prob += .08 if followed == "Yes" else -.10
            win_prob += .03 if emotion == "Calm" else 0
            win_prob -= .10 if emotion == "Frustrated" else 0
            win_prob -= .04 if market == "News/Event" else 0
            win_prob += .025 if strategy == "Trend Following" and market == "Trending" else 0
            is_win = rng.random() < np.clip(win_prob, .12, .82)
            if is_win:
                r_mult = float(min(rr, max(.25, rng.normal(rr * .72, .22))))
                pnl = risk_amt * r_mult
            else:
                r_mult = float(-rng.uniform(.65, 1.18))
                pnl = risk_amt * r_mult
            if abs(pnl) < 0.01:
                pnl = 0.0
            holding_minutes = int(rng.choice([5, 12, 20, 35, 60, 90, 150, 240]))
            exit_time = entry_time + pd.Timedelta(minutes=holding_minutes)
            row = {
                "trade_id": f"TR{trade_n:05d}",
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "instrument": instrument,
                "strategy": strategy,
                "direction": direction,
                "entry_price": round(entry_price, 5 if instrument.endswith("USD") and instrument != "XAUUSD" else 2),
                "stop_loss_price": round(stop, 5 if instrument.endswith("USD") and instrument != "XAUUSD" else 2),
                "take_profit_price": round(take, 5 if instrument.endswith("USD") and instrument != "XAUUSD" else 2),
                "quantity": round(float(rng.choice([0.01, 0.02, 0.05, 0.1, 0.2])), 2),
                "planned_risk_amount": risk_amt,
                "session": session,
                "market_condition": market,
                "emotion_before": emotion,
                "followed_plan": followed,
                "pnl": round(float(pnl), 2),
            }
            day_rows.append(row)
            prior_loss_today = pnl < 0
            trade_n += 1
        records.extend(day_rows)

    df = pd.DataFrame(records).sort_values("entry_time").reset_index(drop=True)
    # Synthetic sample only; trade count/day and outcome relationships are intentionally illustrative.
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate_trades()
    data.to_csv(OUT, index=False)
    print(f"Created {len(data):,} synthetic demo trades at: {OUT}")
    print("These rows are illustrative only, not real market data or financial advice.")
