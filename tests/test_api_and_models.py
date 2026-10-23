"""
WattWise AI - Comprehensive Automated Test Suite
Tests:
1. System Health and Student Credentials
2. Dashboard HTML template structure and student headers
3. Hourly Dataset Frequency, Shape, and Absence of NaNs
4. Mathematical Leakage Prevention (Lags strictly >= 24h)
5. Indian Calendar and Festival Engineering Pipeline
6. Machine Learning Model Loading and Inference (Random Forest, LSTM, Naive)
7. Scenario Simulator Projections and Monotonic Growth Logic
8. Profile Clustering Results and Cluster Dimensions
9. Structured Interactive Chart Endpoints (Historical series, heatmap, seasonal, insights)
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.api.endpoints import (
    health_check,
    get_energy_overview_kpis,
    get_historical_demand_series,
    get_heatmap_matrix,
    get_seasonal_curves,
    get_supporting_insights,
    get_model_comparisons,
    get_clustering_profiles,
    get_festival_impact,
    get_feature_importance,
    get_lstm_loss_history,
    get_maharashtra_series,
    predict_demand,
    simulate_expansion,
    PredictRequest,
    ScenarioRequest
)
from src.data.calendar_features import enrich_calendar_features

def test_health_check():
    data = health_check()
    assert data["status"] == "healthy"
    assert "WattWise AI" in data["service"]
    assert len(data["students"]) == 2
    names = [s["name"] for s in data["students"]]
    assert "Pranav Kolhe" in names
    assert "Amber Shukla" in names
    assert "Nagpur" in data["university"]

def test_dashboard_html_contents():
    html_path = "src/api/dashboard.html"
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "WattWise AI" in html
    assert "Pranav Kolhe" in html
    assert "Amber Shukla" in html
    assert "Shri Ramdeobaba University, Nagpur" in html
    assert "Energy Demand. Supply. Intelligence." in html
    assert "Energy Overview" in html
    assert "Demand Forecasting" in html
    assert "Expansion Simulator" in html

def test_hourly_dataset_frequency_and_integrity():
    csv_path = "data/processed/clean_hourly_national.csv"
    assert os.path.exists(csv_path), "Processed hourly CSV must exist"
    df = pd.read_csv(csv_path)
    assert len(df) == 12984, "Must contain exactly 12,984 hourly readings"
    
    # Regular 1-hour step check
    df["datetime"] = pd.to_datetime(df["datetime"])
    diffs = df["datetime"].diff().dropna()
    expected_diff = pd.Timedelta(hours=1)
    assert (diffs == expected_diff).all(), "Every observation interval must be exactly 1 hour"
    
    # Physical boundaries
    assert (df["solar_mw"] >= 0).all(), "Solar generation must be non-negative (sensor drift clamped)"
    assert (df["wind_mw"] >= 0).all(), "Wind generation must be non-negative"
    assert (df["demand_mw"] > 0).all(), "Demand must be strictly positive"

def test_mathematical_leakage_prevention():
    csv_path = "data/processed/clean_hourly_national.csv"
    df = pd.read_csv(csv_path)
    
    sample_idx = 500
    assert df.loc[sample_idx, "demand_lag_24"] == df.loc[sample_idx - 24, "demand_mw"]
    assert df.loc[sample_idx, "demand_lag_168"] == df.loc[sample_idx - 168, "demand_mw"]
    assert df.loc[sample_idx, "demand_lag_336"] == df.loc[sample_idx - 336, "demand_mw"]

def test_calendar_features_pipeline():
    test_dates = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-11-01 12:00:00", "2024-08-15 09:00:00", "2024-03-25 10:00:00"])
    })
    enriched = enrich_calendar_features(test_dates)
    
    # Diwali test
    diwali_row = enriched.iloc[0]
    assert "Diwali" in diwali_row["festival_name"]
    assert diwali_row["is_festival"] == 1
    
    # Independence Day test
    ind_row = enriched.iloc[1]
    assert ind_row["festival_name"] == "Independence Day"
    assert ind_row["is_national_holiday"] == 1
    
    # Holi test
    holi_row = enriched.iloc[2]
    assert holi_row["festival_name"] == "Holi"
    assert holi_row["is_festival"] == 1

def test_predict_random_forest():
    req = PredictRequest(
        model_name="random_forest",
        hour=14,
        dayofweek=2,
        month=5,
        is_festival=0,
        is_public_holiday=0,
        demand_lag_24=195000.0,
        demand_lag_48=192000.0,
        demand_lag_168=190000.0,
        demand_lag_336=188000.0,
        solar_lag_24=25000.0,
        wind_lag_24=8000.0
    )
    res = predict_demand(req)
    assert res["model_used"] == "random_forest"
    assert "predicted_demand_mw" in res
    assert 100000.0 <= res["predicted_demand_mw"] <= 300000.0
    assert res["predicted_demand_gw"] > 0

def test_predict_lstm():
    req = PredictRequest(
        model_name="lstm",
        hour=12,
        dayofweek=1,
        month=4,
        demand_lag_24=190000.0
    )
    res = predict_demand(req)
    assert res["model_used"] == "lstm"
    assert res["predicted_demand_mw"] > 50000.0

def test_predict_seasonal_naive():
    req = PredictRequest(
        model_name="seasonal_naive",
        demand_lag_24=185000.0
    )
    res = predict_demand(req)
    assert res["model_used"] == "Seasonal Naive (Lag-24)"
    assert res["predicted_demand_mw"] == 185000.0

def test_scenario_simulator_logic():
    req = ScenarioRequest(
        annual_demand_growth_pct=6.0,
        annual_solar_growth_pct=18.0,
        annual_wind_growth_pct=12.0,
        horizon_years=5,
        peak_demand_multiplier=1.04
    )
    res = simulate_expansion(req)
    assert res["status"] == "success"
    assert "HYPOTHETICAL" in res["disclaimer"]
    projections = res["projections"]
    assert len(projections) == 6 # Base year (2024) + 5 projected years
    
    # Verify year monotonicity
    assert projections[0]["year"] == 2024
    assert projections[-1]["year"] == 2029
    assert projections[-1]["projected_peak_demand_gw"] > projections[0]["projected_peak_demand_gw"]
    assert projections[-1]["projected_renewable_share_pct"] > projections[0]["projected_renewable_share_pct"]

def test_model_comparisons_and_clustering_endpoints():
    mc = get_model_comparisons()
    assert len(mc["forecasting_benchmark"]) >= 5
    assert len(mc["clustering_benchmark"]) >= 4
    assert mc["best_forecasting_model"]["Model"] == "Random Forest Regressor"
    
    cl = get_clustering_profiles()
    assert len(cl["clusters"]) == 3
    for cluster in cl["clusters"]:
        assert len(cluster["hourly_profile_mw"]) == 24
        assert cluster["mean_demand_mw"] > 0

def test_overview_kpis_and_festivals():
    kpis = get_energy_overview_kpis()
    assert kpis["all_time_peak_demand_gw"] > 200.0
    assert kpis["total_records_analyzed"] == 12984
    
    fest = get_festival_impact()
    assert len(fest["festival_comparisons"]) >= 5

def test_structured_interactive_chart_endpoints():
    # 1. Historical series range
    hist_30d = get_historical_demand_series(range="30D")
    assert hist_30d["range"] == "30D"
    assert len(hist_30d["timestamps"]) > 0
    assert len(hist_30d["demand_gw"]) == len(hist_30d["timestamps"])
    
    # 2. Heatmap matrix
    hm = get_heatmap_matrix()
    assert len(hm["days"]) == 7
    assert len(hm["hours"]) == 24
    assert len(hm["matrix"]) == 7
    assert len(hm["matrix"][0]) == 24
    
    # 3. Seasonal curves
    sc = get_seasonal_curves()
    assert "Summer" in sc["curves"]
    assert len(sc["curves"]["Summer"]) == 24
    
    # 4. Insights
    ins = get_supporting_insights()
    assert len(ins["insights"]) == 3
    
    # 5. Feature importance
    fi = get_feature_importance()
    assert len(fi["features"]) > 0
    
    # 6. LSTM loss
    ll = get_lstm_loss_history()
    assert len(ll["epoch"]) == 6
    
    # 7. Maharashtra series
    mh_s = get_maharashtra_series(sample_step=30)
    assert len(mh_s["dates"]) > 50
    assert len(mh_s["peak_met_gw"]) == len(mh_s["dates"])
