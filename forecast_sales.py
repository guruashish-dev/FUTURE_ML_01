import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_PATH = DATA_DIR / "sales_history.csv"


def generate_demo_data(path: Path) -> pd.DataFrame:
    """Generate realistic demo daily sales data for ~3 years."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")

    trend = np.linspace(120, 200, len(dates))
    weekly = 12 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 18 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    promo_effect = rng.choice([0, 20], size=len(dates), p=[0.85, 0.15])
    noise = rng.normal(0, 8, len(dates))

    sales = np.maximum(10, trend + weekly + yearly + promo_effect + noise)

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": sales.round(2),
            "is_promo": (promo_effect > 0).astype(int),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_and_clean_data(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = generate_demo_data(path)

    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("Input data must contain 'date' and 'sales' columns.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    if "is_promo" not in df.columns:
        df["is_promo"] = 0

    df = df.dropna(subset=["date", "sales"]).sort_values("date")

    daily = df.groupby("date", as_index=False).agg({"sales": "sum", "is_promo": "max"})
    full_dates = pd.DataFrame({"date": pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")})
    daily = full_dates.merge(daily, on="date", how="left")
    daily["sales"] = daily["sales"].interpolate().bfill().ffill()
    daily["is_promo"] = daily["is_promo"].fillna(0)

    return daily


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["day_of_week"] = feat["date"].dt.dayofweek
    feat["month"] = feat["date"].dt.month
    feat["day_of_year"] = feat["date"].dt.dayofyear
    feat["week_of_year"] = feat["date"].dt.isocalendar().week.astype(int)
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)

    feat["lag_1"] = feat["sales"].shift(1)
    feat["lag_7"] = feat["sales"].shift(7)
    feat["lag_30"] = feat["sales"].shift(30)
    feat["roll_mean_7"] = feat["sales"].shift(1).rolling(7).mean()
    feat["roll_mean_30"] = feat["sales"].shift(1).rolling(30).mean()

    feat = feat.dropna().reset_index(drop=True)
    return feat


def train_forecast_model(feat: pd.DataFrame, horizon: int = 90):
    feature_cols = [
        "is_promo",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_30",
        "roll_mean_7",
        "roll_mean_30",
    ]

    split_idx = int(len(feat) * 0.85)
    train, test = feat.iloc[:split_idx], feat.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["sales"]
    X_test, y_test = test[feature_cols], test["sales"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    mape = np.mean(np.abs((y_test - test_pred) / np.maximum(y_test, 1e-8))) * 100

    history = feat[["date", "sales", "is_promo"]].copy()
    future_preds = []

    for _ in range(horizon):
        next_date = history["date"].max() + pd.Timedelta(days=1)

        lag_1 = history["sales"].iloc[-1]
        lag_7 = history["sales"].iloc[-7]
        lag_30 = history["sales"].iloc[-30]
        roll_mean_7 = history["sales"].iloc[-7:].mean()
        roll_mean_30 = history["sales"].iloc[-30:].mean()

        row = pd.DataFrame(
            {
                "is_promo": [0],
                "day_of_week": [next_date.dayofweek],
                "month": [next_date.month],
                "day_of_year": [next_date.dayofyear],
                "week_of_year": [int(next_date.isocalendar().week)],
                "is_weekend": [1 if next_date.dayofweek >= 5 else 0],
                "lag_1": [lag_1],
                "lag_7": [lag_7],
                "lag_30": [lag_30],
                "roll_mean_7": [roll_mean_7],
                "roll_mean_30": [roll_mean_30],
            }
        )

        pred = float(model.predict(row)[0])
        pred = max(0, pred)
        future_preds.append({"date": next_date, "forecast_sales": pred})

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {"date": [next_date], "sales": [pred], "is_promo": [0]}
                ),
            ],
            ignore_index=True,
        )

    future_df = pd.DataFrame(future_preds)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return model, train, test, test_pred, future_df, metrics


def make_business_plots(train, test, test_pred, future_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train["date"], train["sales"], label="Train (historical)", linewidth=1.2)
    ax.plot(test["date"], test["sales"], label="Actual (holdout)", linewidth=1.5)
    ax.plot(test["date"], test_pred, label="Predicted (holdout)", linewidth=1.5)
    ax.plot(future_df["date"], future_df["forecast_sales"], label="Forecast (next 90 days)", linewidth=2)
    ax.set_title("Sales Forecast: Historical, Validation, and Future")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sales_forecast.png", dpi=180)
    plt.close(fig)


def export_outputs(future_df: pd.DataFrame, metrics: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    future_df.to_csv(OUTPUT_DIR / "future_forecast.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": [round(v, 4) for v in metrics.values()],
        }
    )
    summary.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)


def print_business_summary(metrics: dict, future_df: pd.DataFrame):
    avg_30 = future_df.head(30)["forecast_sales"].mean()
    avg_90 = future_df["forecast_sales"].mean()
    growth = (future_df.tail(30)["forecast_sales"].mean() - future_df.head(30)["forecast_sales"].mean()) / max(avg_30, 1e-8) * 100

    print("\n=== Model Evaluation ===")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    print("\n=== Business Insights ===")
    print(f"Average forecasted daily sales (next 30 days): {avg_30:.2f}")
    print(f"Average forecasted daily sales (next 90 days): {avg_90:.2f}")
    print(f"Expected change from first 30 days to last 30 days: {growth:.2f}%")
    print("Use this forecast for inventory planning, staffing, and campaign timing.")


def main():
    df = load_and_clean_data(DATA_PATH)
    feat = create_features(df)
    model, train, test, test_pred, future_df, metrics = train_forecast_model(feat, horizon=90)
    make_business_plots(train, test, test_pred, future_df)
    export_outputs(future_df, metrics)
    print_business_summary(metrics, future_df)
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
