# Mini-Project Report Template

## Project Title
**Smart Trading Performance & Behavior Mining System Using Data Mining and Data Warehousing**

> Replace the sample-data observations with outputs produced from the dataset you actually use. The bundled sample is synthetic and should be identified as such in any submission.

## 1. Abstract
This project develops a trade-journal analytics system that applies data preprocessing, clustering, classification, association-rule mining, anomaly detection, and data warehousing to closed-trade records. The system summarizes win rate, net profit/loss, maximum drawdown, average R multiple, and plan-following rate. K-Means groups trades by performance characteristics; Random Forest classifies Win/Loss outcomes using selected journal features; Apriori identifies combinations of trading-context items that co-occur with outcomes; and Isolation Forest flags unusual records for human review. A Streamlit dashboard visualizes findings, while a SQLite star-schema warehouse organizes trade facts and descriptive dimensions. The system supports retrospective analysis only and does not place trades or guarantee financial results.

## 2. Problem Statement
Trade logs contain details about outcomes, instruments, strategies, market conditions, trading sessions, and decision-making behaviour. Reviewing these records manually can make it difficult to find repeated patterns, unusual trades, or differences in performance across contexts. This project uses data mining to summarize a trading journal and discover patterns that can support structured self-review.

## 3. Objectives
1. Clean and preprocess the trade-journal dataset.
2. Calculate key trading-performance metrics.
3. Apply clustering to group similar trades.
4. Apply classification to evaluate Win/Loss prediction using a chronological holdout set.
5. Apply association-rule mining to discover combinations associated with outcomes.
6. Detect unusual records and present transparent behaviour-review flags.
7. Design a star-schema data warehouse for fact and dimension data.
8. Display results in an interactive dashboard.

## 4. Scope and Limitations
- One row represents one closed trade.
- P&L is taken from the journal, not calculated from price moves; contract size, fees, swaps, slippage, and account-currency conversions vary.
- Results depend on the completeness and consistency of the imported data.
- Association rules show co-occurrence, not causation.
- Clusters are relative to the current data.
- Classification metrics on historical records do not guarantee predictive performance on future trades.
- Heuristic flags prompt review but cannot diagnose a trader's intent or prove revenge trading.

## 5. Tools and Technologies
- Python
- Pandas and NumPy for cleaning and calculations
- Scikit-learn for K-Means, Random Forest, scaling, and Isolation Forest
- Mlxtend for Apriori association-rule mining
- Streamlit for the dashboard
- Plotly for interactive visualizations
- SQLite for the local star-schema data warehouse
- Visual Studio Code for development

## 6. Dataset Description
Document the dataset you actually used:

- Source: **[Write the source or state that the records are self-entered]**
- Number of input rows: **[Fill in]**
- Number of cleaned rows: **[Fill in]**
- Time period: **[Fill in]**
- Instruments included: **[Fill in]**
- Important columns: trade ID, entry time, instrument, strategy, direction, entry/stop/target prices, planned risk, session, market condition, emotion, plan adherence, and realized P&L.
- Privacy note: remove account IDs, broker credentials, API keys, and unnecessary identifying information.

## 7. Methodology

### 7.1 Data Preprocessing
1. Standardize headings and remove leading/trailing whitespace.
2. Parse dates and numeric fields.
3. Drop rows with invalid entry timestamps or missing P&L.
4. Fill blank categorical fields with `Unknown` and missing numeric fields with a median/default.
5. Remove duplicate trade IDs.
6. Derive outcome, risk/reward ratio, R multiple, weekday, month, holding time, daily trade count, previous-trade outcome, and behaviour-review flags.
7. Scale model inputs where appropriate.

### 7.2 Performance Analysis
Report total trades, Win/Loss/Breakeven counts, win rate, net P&L, average winning trade, average losing trade, average R multiple, maximum drawdown, plan-following rate, and review-flag count.

### 7.3 Clustering: K-Means
Input features: P&L, R multiple, risk/reward ratio, planned risk amount, and holding duration. The implementation uses three clusters and labels them Poor, Average, and Good according to mean R multiple. Include a screenshot of the cluster scatter plot and the cluster summary table.

### 7.4 Classification: Random Forest
Target: Win or Loss; breakeven trades are omitted. Features: instrument, strategy, direction, session, market condition, emotion before entry, plan adherence, and risk/reward ratio. Rows are sorted by entry time; the first 80% trains the model and the final 20% is used as a holdout test set. Report accuracy, precision, recall, F1-score, and the confusion matrix. Do not report a training score as if it were a test score.

### 7.5 Association-Rule Mining: Apriori
Each trade is transformed into categorical items (for example, Session=London, Emotion=Calm, Strategy=Breakout, and Outcome=Win). Rules are summarized by support, confidence, and lift. Include 3–5 interpretable rules from the actual output and state that they indicate associations only.

### 7.6 Anomaly Detection: Isolation Forest
Selected numeric measures are scaled and analyzed. Potential outliers are provided for manual inspection, not automatically labelled as errors or bad decisions.

### 7.7 Data Warehouse Design
The warehouse follows a star-schema structure:

- **Fact table:** `fact_trade` — one record per trade, with P&L, risk, R multiple, outcome, entry data, and dimension keys.
- **Dimension tables:** `dim_date`, `dim_session`, `dim_instrument`, `dim_strategy`, `dim_market_condition`.

Add a screenshot of the Data Warehouse page and explain how the fact table connects to each dimension through foreign keys.

## 8. System Architecture
CSV input → Streamlit interface → preprocessing/feature engineering → data mining models → SQLite warehouse → charts and result tables.

## 9. Results and Screenshots
Insert actual screenshots from your running project:
1. Overview dashboard.
2. Cumulative P&L and session/month charts.
3. K-Means cluster chart and table.
4. Classification metrics and confusion matrix.
5. Association-rule output.
6. Anomaly/behaviour-review table.
7. Data warehouse tables and row counts.

Write your actual observations here: **[Do not invent results; use outputs from your dataset.]**

## 10. Conclusion
The system demonstrates how a trade journal can be processed and explored using multiple data-mining techniques and a star-schema warehouse. It presents performance summaries, relative trade clusters, held-out Win/Loss classification metrics, association rules, unusual records, and transparent behaviour-review flags in a single dashboard. Its outputs can support retrospective reflection, but they are not causal proof, personalized financial advice, or a guarantee of future performance.

## 11. Future Scope
- Add a validated MySQL database backend if required by the course.
- Add configurable filters for account, instrument, strategy, and date range.
- Import broker exports through user-controlled file uploads.
- Add fees and slippage fields if consistently recorded.
- Improve validation with larger, chronologically separated datasets and drift monitoring.
- Add role-based access and stronger privacy controls before multi-user deployment.
