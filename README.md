# Sales/Demand Forecasting Project

This project builds a **business-ready sales forecasting pipeline** using historical daily sales data.

## What this delivers
- Data cleaning and missing-date handling.
- Time-based feature engineering (day/week/month seasonality + lag features).
- Forecasting model (Linear Regression with autoregressive/time features).
- Evaluation metrics (MAE, RMSE, MAPE) and simple error analysis.
- Visual forecast output (`outputs/sales_forecast.png`) and exported forecast table (`outputs/future_forecast.csv`).
- Business-facing summary insights for planning decisions.

## Project structure
- `forecast_sales.py`: end-to-end pipeline.
- `data/sales_history.csv`: historical input file (auto-generated demo file if missing).
- `outputs/`: model metrics, future forecast CSV, and forecast chart.

## Input data format
Provide a CSV at `data/sales_history.csv` with at least:

- `date` (parseable date)
- `sales` (numeric)

Optional:
- `is_promo` (0/1)

## Run
```bash
python3 forecast_sales.py
```

## Key modeling approach
1. Clean and standardize raw sales data.
2. Fill date gaps to daily frequency.
3. Build features:
   - Calendar: day of week, month, week of year, day of year, weekend flag.
   - Lags: 1, 7, 30 day lags.
   - Rolling means: 7, 30 day windows.
4. Train/test split by time order.
5. Train regression model and evaluate holdout performance.
6. Generate iterative 90-day forward forecast.
7. Export plots and business insights.

## Business interpretation examples
- Average next-30-day expected demand for inventory planning.
- Trend shift across next 90 days for staffing/capacity planning.
- MAPE/MAE for estimating forecast uncertainty.

## Visualization options
This implementation uses **Matplotlib** by default. You can export `outputs/future_forecast.csv` into **Power BI** or **Tableau** for executive dashboards.

## Notes
If no dataset is provided, a realistic synthetic dataset is generated so the full workflow can be demonstrated immediately.
