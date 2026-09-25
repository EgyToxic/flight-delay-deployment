from typing import List, Optional
import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Flight Delay Prediction API",
    description="High-performance API for predicting flight delay probability using 5-Fold LightGBM Ensemble",
    version="1.0.0",
)

# تحميل النموذج والـ Artifacts عند الإقلاع
try:
    models = joblib.load("lightgbm_ensemble_model.pkl")
    artifacts = joblib.load("model_artifacts.pkl")
except Exception as e:
    raise RuntimeError(f"فشل تحميل أصول النموذج: {str(e)}")


class FlightInput(BaseModel):
    month: int = Field(..., ge=1, le=12, example=6)
    day: int = Field(..., ge=1, le=31, example=15)
    day_of_week: int = Field(..., ge=1, le=7, example=4)
    airline: str = Field(..., example="DL")
    origin_airport: str = Field(..., example="ATL")
    destination_airport: str = Field(..., example="LAX")
    scheduled_time: float = Field(..., ge=10, le=1000, example=240.0)
    distance: int = Field(..., ge=50, le=5000, example=1946)
    sched_dep_hour: int = Field(..., ge=0, le=23, example=14)
    sched_arr_hour: int = Field(..., ge=0, le=23, example=18)
    origin_hourly_flights: int = Field(..., ge=0, le=200, example=35)
    aircraft_daily_leg: int = Field(..., ge=1, le=15, example=3)
    aircraft_daily_legs: int = Field(..., ge=1, le=15, example=6)
    turnaround_min: float = Field(..., ge=0, le=1440, example=45.0)
    origin_peak_pressure: float = Field(..., ge=0.0, le=1.0, example=0.45)


def preprocess_input(input_data: FlightInput) -> pd.DataFrame:
    data = input_data.dict()
    route = f"{data['origin_airport']}_{data['destination_airport']}"

    # حساب الميزات الزمانية و الهندسية
    data["ROUTE"] = route
    data["IS_WEEKEND"] = 1 if data["day_of_week"] in [6, 7] else 0
    data["IS_PEAK_HOUR"] = (
        1 if data["sched_dep_hour"] in [7, 8, 9, 16, 17, 18, 19] else 0
    )
    data["IS_RED_EYE"] = 1 if data["sched_dep_hour"] in [23, 0, 1, 2, 3, 4] else 0
    data["IS_OVERNIGHT_ARRIVAL"] = (
        1 if data["sched_arr_hour"] < data["sched_dep_hour"] else 0
    )

    data["HOUR_SIN"] = np.sin(2 * np.pi * data["sched_dep_hour"] / 24.0)
    data["HOUR_COS"] = np.cos(2 * np.pi * data["sched_dep_hour"] / 24.0)
    data["DOW_SIN"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["DOW_COS"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    data["DISTANCE_BAND"] = int(data["distance"] // 500)
    data["IS_LONG_HAUL"] = 1 if data["distance"] > 1500 else 0
    data["SCHED_SPEED_MPH"] = (data["distance"] / (data["scheduled_time"] + 1e-5)) * 60
    data["IS_FIRST_LEG"] = 1 if data["aircraft_daily_leg"] == 1 else 0
    data["IS_TIGHT_TURNAROUND"] = 1 if data["turnaround_min"] < 30 else 0
    data["LEG_PROGRESS"] = data["aircraft_daily_leg"] / (
        data["aircraft_daily_legs"] + 1e-5
    )
    data["ROTATION_RISK"] = (
        data["LEG_PROGRESS"] * (1.0 / (data["turnaround_min"] + 1e-5)) * 100
    )

    # 2. الميزات التفاعلية المرّكبة (Interaction Features)
    data["TURNAROUND_PER_PROGRESS"] = data["turnaround_min"] / (
        data["LEG_PROGRESS"] + 1e-5
    )
    data["TURNAROUND_PER_LEG"] = data["turnaround_min"] / (
        data["aircraft_daily_leg"] + 1e-5
    )
    data["CONGESTION_PRESSURE_PROD"] = (
        data["origin_hourly_flights"] * data["origin_peak_pressure"]
    )
    data["LEG_COMPLETION_RATIO"] = data["aircraft_daily_leg"] / (
        data["aircraft_daily_legs"] + 1e-5
    )
    data["SPEED_DISTANCE_RATIO"] = data["SCHED_SPEED_MPH"] / (
        data["distance"] + 1e-5
    )

    # 3. جلب معدلات التأخير التاريخية (Historical Rates)
    gm = artifacts["global_mean"]
    data["ORIGIN_HIST_DELAY_RATE"] = artifacts["origin_delay_rates"].get(
        data["origin_airport"], gm
    )
    data["AIRLINE_HIST_DELAY_RATE"] = artifacts["airline_delay_rates"].get(
        data["airline"], gm
    )
    data["ROUTE_HIST_DELAY_RATE"] = artifacts["route_delay_rates"].get(route, gm)

    # 4. تطبيق Target Encoding
    for col in ["ROUTE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "AIRLINE"]:
        val = data[col if col != "ROUTE" else "ROUTE"]
        data[f"{col}_TE"] = artifacts["te_maps"][col].get(val, gm)

    df_out = pd.DataFrame([data])

    # مطابقة أسماء وأصناف الأعمدة مع التدريب
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]:
        df_out[col] = df_out[col].astype("category")

    # توحيد الأحرف والأعمدة
    df_out.columns = [c.upper() for c in df_out.columns]
    feature_order = artifacts["features"]

    # إضافة أي أعمدة مفقودة بـ 0
    for col in feature_order:
        if col not in df_out.columns:
            df_out[col] = 0

    return df_out[feature_order]


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "LightGBM 5-Fold CV Ensemble",
        "n_folds": len(models),
    }


@app.post("/predict")
def predict_flight_delay(payload: FlightInput):
    try:
        X_df = preprocess_input(payload)

        # Ensemble Probability Prediction
        fold_probs = [m.predict_proba(X_df)[:, 1][0] for m in models]
        avg_prob = float(np.mean(fold_probs))

        threshold = artifacts["best_threshold"]
        is_delayed = bool(avg_prob >= threshold)

        return {
            "is_delayed": is_delayed,
            "delay_probability": round(avg_prob, 4),
            "risk_level": (
                "HIGH" if avg_prob > 0.65 else ("MEDIUM" if avg_prob >= 0.4 else "LOW")
            ),
            "decision_threshold": round(threshold, 4),
            "fold_predictions": [round(p, 4) for p in fold_probs],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))