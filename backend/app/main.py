from pathlib import Path
from typing import Any, Optional

import json
import math

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# Eco Predict — Day 6 FastAPI Backend
# ============================================================
# Expected project structure:
#
# EcoPredict/
# ├── backend/
# │   └── app/main.py
# ├── data/
# │   ├── raw/lucknow/<station>/air_2024.csv ...
# │   └── processed/
# ├── frontend/
# ├── ml/
# ├── models/
# │   └── multi_station/
# └── outputs/
#     ├── multi_station/
#     └── policy*.csv/json
#
# The backend reads the artifacts produced by the existing
# Day 1–5 ML notebooks. It does not create fabricated ML values.
# ============================================================


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_LUCKNOW_DIR = DATA_DIR / "raw" / "lucknow"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MULTI_MODEL_DIR = MODELS_DIR / "multi_station"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MULTI_OUTPUT_DIR = OUTPUTS_DIR / "multi_station"


STATIONS = {
    "gomti_nagar": {
        "name": "Gomti Nagar",
        "folder": "gomti_nagar",
        # Prototype UI coordinates. Replace with verified official
        # CPCB station coordinates before final deployment.
        "lat": 26.8487,
        "lng": 81.0070,
    },
    "kendriya_vidyalaya": {
        "name": "Kendriya Vidyalaya",
        "folder": "kendriya_vidyalaya",
        "lat": 26.8650,
        "lng": 80.9660,
    },
    "lalbagh": {
        "name": "Lalbagh",
        "folder": "lalbagh",
        "lat": 26.8467,
        "lng": 80.9462,
    },
    "talkatora_dic": {
        "name": "Talkatora DIC",
        "folder": "talkatora_dic",
        "lat": 26.8320,
        "lng": 80.9070,
    },
}

STATION_ALIASES = {}
for station_id, info in STATIONS.items():
    STATION_ALIASES[station_id] = station_id
    STATION_ALIASES[info["name"].lower()] = station_id
    STATION_ALIASES[info["folder"].lower()] = station_id

TARGET = "PM2.5 (µg/m³)"
FORECAST_HORIZON = 24
LAGS = [1, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [6, 12, 24, 72]

POLICIES = [
    {
        "policy_id": "P01",
        "policy": "Reduce heavy vehicle activity",
        "lever": "Traffic / heavy vehicles",
        "assumed_pm25_reduction_pct": 5.0,
        "budget_units": 3,
        "description": "Scenario: reduce heavy-vehicle activity during high-risk periods.",
    },
    {
        "policy_id": "P02",
        "policy": "Improve industrial emission efficiency",
        "lever": "Industrial emissions",
        "assumed_pm25_reduction_pct": 8.0,
        "budget_units": 4,
        "description": "Scenario: improve emission-control efficiency at priority facilities.",
    },
    {
        "policy_id": "P03",
        "policy": "Shift trips to public transport",
        "lever": "Transport demand",
        "assumed_pm25_reduction_pct": 4.0,
        "budget_units": 2,
        "description": "Scenario: shift a portion of private trips to public transport.",
    },
    {
        "policy_id": "P04",
        "policy": "Cleaner energy transition",
        "lever": "Fuel / energy",
        "assumed_pm25_reduction_pct": 6.0,
        "budget_units": 5,
        "description": "Scenario: increase cleaner energy/fuel usage.",
    },
    {
        "policy_id": "P05",
        "policy": "Targeted green intervention",
        "lever": "Local mitigation",
        "assumed_pm25_reduction_pct": 2.5,
        "budget_units": 1,
        "description": "Scenario: targeted local mitigation in priority areas.",
    },
]


app = FastAPI(
    title="Eco Predict API",
    description=(
        "AI-powered environmental governance and decision-support API. "
        "Forecast and risk values come from the Eco Predict ML artifacts."
    ),
    version="1.0.0-day6",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------

def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if pd.isna(value):
        return None
    return value


def resolve_station(station: str) -> str:
    key = station.strip().lower()
    if key in STATION_ALIASES:
        return STATION_ALIASES[key]
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown station '{station}'. Use one of: "
            + ", ".join(STATIONS.keys())
        ),
    )


def station_display_name(station_id: str) -> str:
    return STATIONS[station_id]["name"]


def require_file(path: Path) -> Path:
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Required Eco Predict artifact was not found: {path}. "
                "Run the Day 1–5 ML notebook and make sure the project "
                "folder structure is unchanged."
            ),
        )
    return path


def find_multi_station_file(filename: str) -> Path:
    return MULTI_OUTPUT_DIR / filename


def load_multi_output_csv(filename: str) -> pd.DataFrame:
    path = find_multi_station_file(filename)
    require_file(path)
    return pd.read_csv(path)


def load_processed_multi_station() -> pd.DataFrame:
    path = PROCESSED_DIR / "lucknow_multi_station.csv"
    require_file(path)
    df = pd.read_csv(path)
    if "Timestamp" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="lucknow_multi_station.csv is missing the Timestamp column.",
        )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df.sort_values(["Station", "Timestamp"]).reset_index(drop=True)


def model_file(station_id: str, anomaly: bool = False) -> Path:
    suffix = "pm25_anomaly_detector_" if anomaly else "pm25_forecaster_24h_"
    return MULTI_MODEL_DIR / f"{suffix}{station_id}.joblib"


def build_station_features(station_frame: pd.DataFrame):
    """
    Exact feature logic used by the Day 5 notebook:
    calendar + cyclic features + PM2.5 lags + past-only rolling stats.
    """
    d = station_frame.sort_values("Timestamp").copy()

    d["hour"] = d["Timestamp"].dt.hour
    d["day_of_week"] = d["Timestamp"].dt.dayofweek
    d["month"] = d["Timestamp"].dt.month
    d["day_of_year"] = d["Timestamp"].dt.dayofyear
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    for lag in LAGS:
        d[f"pm25_lag_{lag}"] = d[TARGET].shift(lag)

    for window in ROLLING_WINDOWS:
        d[f"pm25_roll_mean_{window}"] = (
            d[TARGET].shift(1).rolling(window).mean()
        )
        d[f"pm25_roll_std_{window}"] = (
            d[TARGET].shift(1).rolling(window).std()
        )

    feature_cols = [
        "hour", "day_of_week", "month", "day_of_year", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    feature_cols += [f"pm25_lag_{lag}" for lag in LAGS]
    feature_cols += [f"pm25_roll_mean_{w}" for w in ROLLING_WINDOWS]
    feature_cols += [f"pm25_roll_std_{w}" for w in ROLLING_WINDOWS]

    return d, feature_cols


def load_station_model(station_id: str):
    path = model_file(station_id, anomaly=False)
    require_file(path)
    try:
        return joblib.load(path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load forecasting model for {station_id}: {exc}",
        )


def predict_station_24h(station_id: str):
    df = load_processed_multi_station()
    station_name = station_display_name(station_id)
    g = df[df["Station"].astype(str).str.lower() == station_name.lower()].copy()

    if g.empty:
        # Fallback for possible station-id values in the CSV.
        g = df[df["Station"].astype(str).str.lower() == station_id.lower()].copy()

    if g.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No processed data found for {station_name}.",
        )

    g = g.sort_values("Timestamp").copy()
    valid = g.dropna(subset=[TARGET])
    if valid.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No valid PM2.5 observations found for {station_name}.",
        )

    latest_valid = valid.iloc[-1]
    latest_timestamp = latest_valid["Timestamp"]

    feature_frame, features_from_notebook = build_station_features(g)
    usable = feature_frame.dropna(subset=features_from_notebook)

    if usable.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Not enough historical PM2.5 data to build features for {station_name}.",
        )

    latest_features = usable.iloc[[-1]]
    package = load_station_model(station_id)

    model = package["model"]
    features = package.get("features", features_from_notebook)

    missing_features = [c for c in features if c not in latest_features.columns]
    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Model features missing from feature frame: {missing_features}",
        )

    try:
        prediction = float(model.predict(latest_features[features])[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast prediction failed for {station_name}: {exc}",
        )

    recent = valid.tail(24)
    recent_mean = float(recent[TARGET].mean())

    # Return a useful chart-ready history plus the 24h-ahead point.
    history = (
        valid.tail(48)[["Timestamp", TARGET]]
        .rename(columns={"Timestamp": "timestamp", TARGET: "pm25"})
    )

    history_records = []
    for _, row in history.iterrows():
        history_records.append({
            "timestamp": row["timestamp"],
            "pm25": row["pm25"],
            "type": "actual",
        })

    forecast_timestamp = latest_timestamp + pd.Timedelta(hours=24)
    history_records.append({
        "timestamp": forecast_timestamp,
        "pm25": prediction,
        "type": "forecast",
    })

    return {
        "station": station_name,
        "station_id": station_id,
        "latest_timestamp": latest_timestamp,
        "current_pm25": float(latest_valid[TARGET]),
        "recent_24h_mean_pm25": recent_mean,
        "forecast_horizon_hours": 24,
        "forecast_pm25": prediction,
        "forecast_model": package.get("model_name", "Unknown"),
        "series": history_records,
        "note": (
            "This is a 24-hour-ahead PM2.5 forecast produced by the "
            "station-specific Day 5 model."
        ),
    }


def risk_category(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def load_risk_table() -> pd.DataFrame:
    return load_multi_output_csv("tomorrow_station_risk.csv")


def load_priority_table() -> pd.DataFrame:
    return load_multi_output_csv("station_inspection_priority.csv")


def station_row_from_risk(station_id: str) -> dict:
    df = load_risk_table()
    target_name = station_display_name(station_id)

    rows = df[df["Station"].astype(str).str.lower() == target_name.lower()]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No risk result found for {target_name}.",
        )
    return json_safe(rows.iloc[0].to_dict())


def policy_by_id(policy_id: str) -> dict:
    for policy in POLICIES:
        if policy["policy_id"].upper() == policy_id.upper():
            return policy
    raise HTTPException(
        status_code=404,
        detail=f"Unknown policy_id '{policy_id}'.",
    )


# -----------------------------
# Request models
# -----------------------------

class PolicySimulationRequest(BaseModel):
    station: str = Field(default="gomti_nagar")
    policy_id: str
    baseline_pm25: Optional[float] = Field(default=None, gt=0)


class PolicyBundleRequest(BaseModel):
    station: str = Field(default="gomti_nagar")
    policy_ids: list[str]
    baseline_pm25: Optional[float] = Field(default=None, gt=0)


# -----------------------------
# Basic routes
# -----------------------------

@app.get("/")
def root():
    return {
        "name": "Eco Predict API",
        "status": "running",
        "version": "1.0.0-day6",
        "message": "Environmental governance decision-support backend",
    }


@app.get("/health")
def health():
    required_artifacts = {
        "processed_multi_station": (
            PROCESSED_DIR / "lucknow_multi_station.csv"
        ).exists(),
        "station_risk": (
            MULTI_OUTPUT_DIR / "tomorrow_station_risk.csv"
        ).exists(),
        "inspection_priority": (
            MULTI_OUTPUT_DIR / "station_inspection_priority.csv"
        ).exists(),
        "forecast_models": all(
            model_file(sid).exists() for sid in STATIONS
        ),
        "anomaly_models": all(
            model_file(sid, anomaly=True).exists() for sid in STATIONS
        ),
    }

    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "artifacts": required_artifacts,
    }


# -----------------------------
# Station / Dashboard
# -----------------------------

@app.get("/api/stations")
def get_stations():
    return {
        "city": "Lucknow",
        "stations": [
            {
                "id": station_id,
                "name": info["name"],
                "lat": info["lat"],
                "lng": info["lng"],
            }
            for station_id, info in STATIONS.items()
        ],
        "coordinate_note": (
            "Map coordinates are prototype positions and should be replaced "
            "with verified official CPCB station coordinates before final deployment."
        ),
    }


@app.get("/api/dashboard")
def dashboard(
    station: str = Query(default="all"),
):
    risk_df = load_risk_table()

    if station.lower() == "all":
        selected = risk_df.copy()
        station_label = "All Lucknow Stations"
    else:
        station_id = resolve_station(station)
        station_label = station_display_name(station_id)
        selected = risk_df[
            risk_df["Station"].astype(str).str.lower() == station_label.lower()
        ]

    if selected.empty:
        raise HTTPException(status_code=404, detail="No dashboard data found.")

    risk_col = "Risk Score"
    forecast_col = "24h Forecast PM2.5"
    current_col = "Current PM2.5"

    high_count = int((selected[risk_col] >= 50).sum())
    critical_count = int((selected[risk_col] >= 75).sum())

    highest = selected.sort_values(risk_col, ascending=False).iloc[0]

    return json_safe({
        "station": station_label,
        "stations_monitored": int(len(selected)),
        "average_current_pm25": float(selected[current_col].mean()),
        "average_forecast_pm25": float(selected[forecast_col].mean()),
        "highest_risk_station": highest["Station"],
        "highest_risk_score": highest[risk_col],
        "high_or_critical_stations": high_count,
        "critical_stations": critical_count,
        "risk_rows": selected.to_dict(orient="records"),
        "note": (
            "Risk scores are prototype decision-support indicators. "
            "They are not official CPCB AQI classifications or enforcement decisions."
        ),
    })


# -----------------------------
# Analytics
# -----------------------------

@app.get("/api/analytics")
def analytics():
    quality_path = PROCESSED_DIR / "station_quality_report.csv"
    metrics_path = MULTI_MODEL_DIR / "station_forecast_metrics.csv"

    result = {
        "station_quality": [],
        "forecast_metrics": [],
    }

    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        result["station_quality"] = quality.to_dict(orient="records")

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        result["forecast_metrics"] = metrics.to_dict(orient="records")

    if not result["station_quality"] and not result["forecast_metrics"]:
        raise HTTPException(
            status_code=503,
            detail="Analytics artifacts are not available yet.",
        )

    return json_safe(result)


# -----------------------------
# Forecast
# -----------------------------

@app.get("/api/forecast")
def forecast(
    station: str = Query(default="gomti_nagar"),
):
    station_id = resolve_station(station)
    return json_safe(predict_station_24h(station_id))


# -----------------------------
# Risk / Anomaly
# -----------------------------

@app.get("/api/risk")
def risk(
    station: str = Query(default="all"),
):
    df = load_risk_table()

    if station.lower() == "all":
        rows = df.to_dict(orient="records")
    else:
        station_id = resolve_station(station)
        name = station_display_name(station_id)
        rows = df[
            df["Station"].astype(str).str.lower() == name.lower()
        ].to_dict(orient="records")

    if not rows:
        raise HTTPException(status_code=404, detail="No risk data found.")

    return json_safe({
        "stations": rows,
        "signal_label": "Unusual Pattern Detected",
        "signal_explanation": (
            "Recent environmental conditions differ significantly from "
            "historical patterns. This is a screening signal and does not "
            "establish the cause of pollution."
        ),
        "risk_note": (
            "Risk is a prototype prioritization indicator, not an official "
            "regulatory classification."
        ),
    })


# -----------------------------
# Inspection Priority
# -----------------------------

@app.get("/api/inspection-priority")
def inspection_priority():
    df = load_priority_table()
    df = df.sort_values("Inspection Rank")
    return json_safe({
        "rows": df.to_dict(orient="records"),
        "note": (
            "Inspection ranking is decision support for prioritizing limited "
            "attention. It is not an automatic enforcement decision."
        ),
    })


# -----------------------------
# Policy Intelligence
# -----------------------------

@app.get("/api/policy/library")
def policy_library():
    return {
        "policies": POLICIES,
        "estimate_note": (
            "Policy impacts are transparent prototype scenario assumptions, "
            "not causal estimates."
        ),
    }


def get_baseline(station_id: str, provided: Optional[float]) -> float:
    if provided is not None:
        return float(provided)

    forecast_result = predict_station_24h(station_id)
    return float(forecast_result["forecast_pm25"])


@app.post("/api/policy/simulate")
def simulate_policy(request: PolicySimulationRequest):
    station_id = resolve_station(request.station)
    policy = policy_by_id(request.policy_id)
    baseline = get_baseline(station_id, request.baseline_pm25)

    reduction_pct = float(policy["assumed_pm25_reduction_pct"])
    projected = baseline * (1 - reduction_pct / 100)

    return json_safe({
        "station": station_display_name(station_id),
        "policy_id": policy["policy_id"],
        "policy": policy["policy"],
        "baseline_pm25": round(baseline, 2),
        "projected_pm25": round(projected, 2),
        "estimated_reduction": round(baseline - projected, 2),
        "assumed_reduction_pct": reduction_pct,
        "budget_units": int(policy["budget_units"]),
        "estimate_type": "Scenario estimate",
        "note": (
            "The simulator does not retrain the ML model and does not claim "
            "that the policy will definitely cause the estimated reduction."
        ),
    })


@app.post("/api/policy/bundle")
def policy_bundle(request: PolicyBundleRequest):
    station_id = resolve_station(request.station)
    if not request.policy_ids:
        raise HTTPException(
            status_code=400,
            detail="Select at least one policy.",
        )

    selected = [policy_by_id(pid) for pid in request.policy_ids]
    baseline = get_baseline(station_id, request.baseline_pm25)

    total_reduction = min(
        sum(float(p["assumed_pm25_reduction_pct"]) for p in selected),
        80.0,
    )
    projected = baseline * (1 - total_reduction / 100)

    return json_safe({
        "station": station_display_name(station_id),
        "selected_policies": [p["policy"] for p in selected],
        "policy_ids": [p["policy_id"] for p in selected],
        "baseline_pm25": round(baseline, 2),
        "combined_assumed_reduction_pct": round(total_reduction, 2),
        "projected_pm25": round(projected, 2),
        "estimated_reduction": round(baseline - projected, 2),
        "total_budget_units": int(sum(p["budget_units"] for p in selected)),
        "estimate_type": "Scenario estimate",
        "note": (
            "Combined reduction uses the same capped additive scenario "
            "assumption as the existing Day 3 prototype."
        ),
    })


@app.get("/api/policy/recommend")
def policy_recommend(
    station: str = Query(default="gomti_nagar"),
):
    station_id = resolve_station(station)
    baseline = get_baseline(station_id, None)

    rows = []
    for policy in POLICIES:
        reduction_pct = float(policy["assumed_pm25_reduction_pct"])
        projected = baseline * (1 - reduction_pct / 100)
        reduction = baseline - projected
        budget = max(float(policy["budget_units"]), 1.0)

        rows.append({
            "policy_id": policy["policy_id"],
            "policy": policy["policy"],
            "projected_pm25": round(projected, 2),
            "estimated_reduction": round(reduction, 2),
            "assumed_reduction_pct": reduction_pct,
            "budget_units": int(policy["budget_units"]),
            "impact_per_budget_unit": round(reduction / budget, 3),
            "estimate_type": "Scenario estimate",
        })

    rows.sort(
        key=lambda x: (
            x["impact_per_budget_unit"],
            x["estimated_reduction"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(rows, start=1):
        row["recommendation_rank"] = rank

    return json_safe({
        "station": station_display_name(station_id),
        "baseline_pm25": round(baseline, 2),
        "recommendations": rows,
        "note": (
            "Budget units are relative planning units, not rupee costs. "
            "Recommendations are transparent prototype comparisons."
        ),
    })


@app.get("/api/policy/verification")
def policy_verification():
    path = OUTPUTS_DIR / "policy_verification_template.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Policy verification template is not available yet.",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read policy verification template: {exc}",
        )

    return json_safe({
        "verification": data,
        "note": (
            "No fabricated post-policy observation is used. Verified "
            "post-intervention observations are required for effectiveness."
        ),
    })


# -----------------------------
# Run with:
# uvicorn app.main:app --reload
# -----------------------------
