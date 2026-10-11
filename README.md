# Smart Trading Performance & Behavior Mining System

**Mini-project for Data Mining & Data Warehousing**

This beginner-friendly project analyses a trade journal to discover performance and behaviour patterns. It is an **analytics dashboard**, not a live-trading bot and not a buy/sell signal service.

## What is included?

- Streamlit dashboard with summary metrics, charts, trade explorer, and model results.
- CSV cleaning: missing-value handling, duplicate trade ID handling, numeric conversion, date parsing, and derived features.
- Performance measures: win rate, net P&L, average win/loss, average R multiple, peak-to-trough drawdown, and plan-following rate.
- **K-Means clustering** to group trades into Poor / Average / Good performance clusters (assigned by average R multiple in this dataset).
- **Random Forest classification** to classify Win/Loss from journal fields that could be known before entry. The earliest 80% of chronological trades train the model and the latest 20% are held out for testing.
- **Apriori association-rule mining** to discover combinations associated with Win/Loss, reported with support, confidence, and lift.
- **Isolation Forest anomaly detection** to flag unusual rows for review.
- Behaviour review heuristics for 5+ trades/day, trades where the plan was not followed, and frustrated/FOMO emotion following a loss. These are prompts for self-review, not proof of revenge trading.
- A local **SQLite star-schema data warehouse** with `fact_trade`, `dim_date`, `dim_session`, `dim_instrument`, `dim_strategy`, and `dim_market_condition`.
- A 473-row synthetic sample so the project runs immediately without downloading a dataset.

## Important note about the sample

`data/sample_trades.csv` is synthetic demonstration data. It is not real gold/forex/crypto market data, not broker history, and not evidence that a strategy is profitable. Upload your own closed-trade journal CSV for meaningful personal analysis. The P&L values are read from your CSV as recorded; the app does not infer P&L from instrument prices because contract size and currency conversion depend on the broker and instrument.

---

## 1. Install the tools on Windows

1. Install **Python 3.12** from the official Python website. During installation, tick **Add Python to PATH**. Python 3.11 should also work. (Using a commonly supported release avoids package compatibility issues.)
2. Install **Visual Studio Code**.
3. In VS Code, choose **File → Open Folder** and open the extracted `trading_performance_mining` project folder.
4. Open **Terminal → New Terminal**. The terminal should start inside the project folder.

Check Python:

```powershell
py --version
```

If `py` is not recognized, try `python --version`. If neither works, reinstall Python and tick “Add Python to PATH”.

## 2. Create a virtual environment

Run the following commands in the VS Code PowerShell terminal, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in that terminal and then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the terminal line after activation. Keep using that terminal for this project.

## 3. Generate or refresh the sample data

The sample CSV is already included. You can regenerate it with:

```powershell
python generate_sample_data.py
```

That creates `data/sample_trades.csv`. It is synthetic sample data only.

## 4. Run the dashboard

From the project folder and with `.venv` activated, run:

```powershell
streamlit run app.py
```

Streamlit should print a local address, usually `http://localhost:8501`, and may open it in your browser automatically. If it does not open automatically, copy the **Local URL** from the terminal into Chrome.

To stop the app, return to the terminal and press `Ctrl+C`.

## 5. How the pieces connect

```text
sample_trades.csv OR uploaded CSV
              |
              v
        app.py (Streamlit UI)
              |
              v
   src/data_utils.py (cleaning/features)
              |
       +------+-------------------------+
       |                                |
       v                                v
src/analytics.py                  src/warehouse.py
K-Means / Random Forest           SQLite star schema
Apriori / Isolation Forest         data/trading_warehouse.db
       |                                |
       +---------------+----------------+
                       v
                Dashboard results
```

- **`app.py`** is the main dashboard and connects user input, functions, and display.
- **`src/data_utils.py`** validates the CSV, cleans it, and creates fields such as `outcome`, `r_multiple`, `daily_trade_count`, and behavior review flags.
- **`src/analytics.py`** contains the data mining algorithms.
- **`src/warehouse.py`** creates and loads the star-schema tables in `data/trading_warehouse.db`.
- **`data/sample_trades.csv`** is the default demo dataset.
- **`requirements.txt`** lists the Python packages.

The database is automatically refreshed from whichever CSV is selected in the app. No database username/password is needed because this starter version uses SQLite, which is built into Python. The star-schema design is still shown and can later be migrated to MySQL if your teacher requires MySQL specifically.

## 6. Upload your own trade journal

In the app's left sidebar, use **Upload your trade-journal CSV**. Each row should represent one closed trade.

Required column names (case-insensitive):

```text
trade_id
entry_time
instrument
strategy
direction
entry_price
stop_loss_price
take_profit_price
planned_risk_amount
session
market_condition
emotion_before
followed_plan
pnl
```

Optional fields: `exit_time`, `quantity`.

Use `data/sample_trades.csv` as the template and preserve the column names. Example category values include:

- `direction`: `Buy` or `Sell`
- `session`: `Asian`, `London`, `New York` (or your own labels)
- `followed_plan`: `Yes` or `No`
- `emotion_before`: `Calm`, `Confident`, `FOMO`, `Frustrated` (or your own labels)
- `pnl`: the realized P&L for that trade as it appears in your journal. Do not enter a percent in one row and a currency amount in another.

If your broker exports different column names, rename the headings in a copy of the CSV to match the required names before uploading. Never upload passwords, API keys, account IDs, or other unnecessary personal/financial information.

## 7. What each model does

### A. K-Means clustering

Features used: P&L, R multiple, risk/reward ratio, planned risk amount, and holding time. The model groups similar trade records into three clusters. The app labels the clusters Poor/Average/Good based on their mean R multiple. Cluster names are relative, not absolute ratings.

### B. Random Forest classification

Target: `Win` or `Loss`; breakeven trades are excluded from this evaluation. Inputs: instrument, strategy, direction, session, market condition, pre-trade emotion, plan adherence, and planned risk/reward ratio. It uses a chronological 80/20 split: earlier rows train the model; later rows test it. The app reports accuracy, win precision, win recall, F1-score, and a confusion matrix. It does not use the realized P&L or outcome as a model input.

### C. Apriori association rules

The system turns each trade into items such as `Session=London`, `Emotion=Calm`, `Strategy=Breakout`, `DailyFrequency=NormalDailyFrequency`, and `Outcome=Win`. It searches for patterns that co-occur with Win/Loss and reports:

- **Support**: fraction of all trades containing a rule's items.
- **Confidence**: fraction of trades with the left side that also have the outcome on the right.
- **Lift**: how often the outcome appears with the left side relative to its overall frequency.

A rule is a historical association, not proof that one condition caused a result.

### D. Isolation Forest anomaly detection

Looks for unusual combinations of P&L, R multiple, risk/reward ratio, planned risk, and number of trades that day. An anomaly is only a row worth reviewing; it is not automatically an error or bad trade.

## 8. Data warehouse schema

- `fact_trade`: one row per trade with measures and foreign keys.
- `dim_date`: calendar/date attributes.
- `dim_session`: session names.
- `dim_instrument`: traded instruments.
- `dim_strategy`: trading strategies.
- `dim_market_condition`: market condition labels.

The Data warehouse page lets you preview these tables. The SQLite database file appears after you first run the app: `data/trading_warehouse.db`.

## 9. Troubleshooting

- **`streamlit` is not recognized:** make sure `.venv` is activated and run `python -m streamlit run app.py`.
- **Module not found:** run `pip install -r requirements.txt` with `.venv` active.
- **PowerShell activation error:** use the `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` command above.
- **CSV missing columns:** use the template and keep required headings exactly (upper/lower case does not matter).
- **No association rules:** lower the support/confidence sliders, or add more trades.
- **Not enough classification rows:** classification needs at least 30 non-breakeven rows and more than one class in the training period.
- **Results change after uploading different data:** that is expected; the models analyze the current uploaded dataset and the local warehouse is refreshed from that dataset.

## 10. Suggested mini-project demonstration

1. Run the app on the sample data.
2. Show win rate, net P&L, maximum drawdown, and session/month charts.
3. Explain how cleaning and feature engineering work.
4. Show K-Means clusters and explain that they group similar trade records.
5. Show Random Forest's held-out metrics and confusion matrix.
6. Change Apriori support/confidence and explain one discovered rule.
7. Show anomaly and behavior-review flags, clarifying that flags require human review.
8. Open the Data warehouse page and explain the fact/dimension tables.
9. Upload a journal CSV only if it contains suitable non-sensitive fields.
