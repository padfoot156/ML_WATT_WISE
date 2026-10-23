"""
WattWise AI - Data Preprocessing & Leakage-Free Feature Engineering Pipeline
Handles Mendeley National Hourly SCADA data and Zenodo State/National Daily datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from src.data.calendar_features import enrich_calendar_features

def preprocess_hourly_data(
    raw_path: str = "data/raw/mendeley/Jan2024_Jun2025.xlsx",
    output_dir: str = "data/processed"
) -> dict:
    """
    Cleans raw hourly data, builds calendar and lagged features strictly avoiding leakage,
    and produces chronological train/val/test splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw hourly data from {raw_path}...")
    df = pd.read_excel(raw_path, sheet_name="Report")
    
    # 1. Clean summary footer rows
    valid_mask = ~df["Timestamp"].astype(str).str.contains(
        "Minimum|Maximum|Average|Sum|Timestamp", case=False, na=False
    )
    df_clean = df[valid_mask].copy()
    
    # 2. Parse Datetime and Sort
    df_clean["datetime"] = pd.to_datetime(df_clean["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df_clean = df_clean.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    
    # 3. Clean numeric columns & clamp sensor zero-drift
    num_cols = ["Demand (MW)", "Wind (MW)", "Solar (MW)", "Total Generation (MW)"]
    for c in num_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    
    # Clamp solar sensor baseline drift at night (<0) to 0.0
    df_clean["Solar (MW)"] = np.maximum(0.0, df_clean["Solar (MW)"])
    df_clean["Wind (MW)"] = np.maximum(0.0, df_clean["Wind (MW)"])
    
    # Standardize column names
    rename_dict = {
        "Demand (MW)": "demand_mw",
        "Wind (MW)": "wind_mw",
        "Solar (MW)": "solar_mw",
        "Total Generation (MW)": "total_generation_mw"
    }
    df_clean = df_clean.rename(columns=rename_dict)
    
    # 4. Verify regular hourly frequency
    full_idx = pd.date_range(start=df_clean["datetime"].min(), end=df_clean["datetime"].max(), freq="1h")
    assert len(full_idx) == len(df_clean), f"Mismatch in hourly timestamps: expected {len(full_idx)}, got {len(df_clean)}"
    
    # 5. Enrich with Indian Calendar & Festival Features
    df_enriched = enrich_calendar_features(df_clean, datetime_col="datetime")
    
    # 6. Leakage-Free Lag and Rolling Features for 24-Hour Horizon Forecasting
    # Any forecast made at issue time t for the day ahead (t+1..t+24) can only access observations <= t.
    # Therefore, all baseline lag features start at shift(24) or shift(1) for direct next-hour.
    # To support both 1-hour and 24-hour forecasting pipelines cleanly:
    
    # Direct 24-hour shift features (guaranteed known at or before t for day-ahead forecast)
    df_enriched["demand_lag_24"] = df_enriched["demand_mw"].shift(24)
    df_enriched["demand_lag_48"] = df_enriched["demand_mw"].shift(48)
    df_enriched["demand_lag_168"] = df_enriched["demand_mw"].shift(168) # 1 week ago
    df_enriched["demand_lag_336"] = df_enriched["demand_mw"].shift(336) # 2 weeks ago
    
    # Rolling stats shifted by 24 hours to prevent target leakage
    rolled_24 = df_enriched["demand_mw"].shift(24).rolling(window=24, min_periods=24)
    df_enriched["demand_roll_mean_24h"] = rolled_24.mean()
    df_enriched["demand_roll_std_24h"] = rolled_24.std()
    df_enriched["demand_roll_min_24h"] = rolled_24.min()
    df_enriched["demand_roll_max_24h"] = rolled_24.max()
    
    rolled_7d = df_enriched["demand_mw"].shift(24).rolling(window=168, min_periods=168)
    df_enriched["demand_roll_mean_7d"] = rolled_7d.mean()
    df_enriched["demand_roll_std_7d"] = rolled_7d.std()
    
    # Generation lag features (shifted by 24h)
    df_enriched["solar_lag_24"] = df_enriched["solar_mw"].shift(24)
    df_enriched["wind_lag_24"] = df_enriched["wind_mw"].shift(24)
    df_enriched["total_gen_lag_24"] = df_enriched["total_generation_mw"].shift(24)
    
    # Short-term 1-hour lag features for direct 1-step ahead evaluation
    df_enriched["demand_lag_1"] = df_enriched["demand_mw"].shift(1)
    df_enriched["demand_lag_2"] = df_enriched["demand_mw"].shift(2)
    df_enriched["solar_lag_1"] = df_enriched["solar_mw"].shift(1)
    df_enriched["wind_lag_1"] = df_enriched["wind_mw"].shift(1)

    # Renewable penetration indicator (historical component)
    df_enriched["renewable_mw"] = df_enriched["solar_mw"] + df_enriched["wind_mw"]
    df_enriched["renewable_share_pct"] = np.where(
        df_enriched["total_generation_mw"] > 0,
        (df_enriched["renewable_mw"] / df_enriched["total_generation_mw"]) * 100.0,
        0.0
    )

    # Save full processed dataset
    full_clean_path = os.path.join(output_dir, "clean_hourly_national.csv")
    df_enriched.to_csv(full_clean_path, index=False)
    print(f"Saved full clean dataset ({len(df_enriched)} rows) to {full_clean_path}")

    # 7. Chronological Partitioning
    # Discard initial warmup period with NaNs from lag_336
    df_model_ready = df_enriched.dropna(subset=["demand_lag_336", "demand_roll_mean_7d"]).copy().reset_index(drop=True)
    
    # Split dates:
    # Train: start of data through 2024-11-30
    # Val: 2024-12-01 through 2025-02-28
    # Test: 2025-03-01 through 2025-06-24
    train_mask = df_model_ready["datetime"] <= "2024-11-30 23:00:00"
    val_mask = (df_model_ready["datetime"] >= "2024-12-01 00:00:00") & (df_model_ready["datetime"] <= "2025-02-28 23:00:00")
    test_mask = df_model_ready["datetime"] >= "2025-03-01 00:00:00"
    
    train_df = df_model_ready[train_mask].reset_index(drop=True)
    val_df = df_model_ready[val_mask].reset_index(drop=True)
    test_df = df_model_ready[test_mask].reset_index(drop=True)
    
    train_path = os.path.join(output_dir, "hourly_train.csv")
    val_path = os.path.join(output_dir, "hourly_val.csv")
    test_path = os.path.join(output_dir, "hourly_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Splits saved:")
    print(f" - Train: {train_path} ({len(train_df)} rows, {train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f" - Val:   {val_path} ({len(val_df)} rows, {val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f" - Test:  {test_path} ({len(test_df)} rows, {test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    summary = {
        "total_records": len(df_enriched),
        "model_ready_records": len(df_model_ready),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "train_period": f"{train_df['datetime'].min()} to {train_df['datetime'].max()}",
        "val_period": f"{val_df['datetime'].min()} to {val_df['datetime'].max()}",
        "test_period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "demand_mean": float(df_enriched["demand_mw"].mean()),
        "demand_min": float(df_enriched["demand_mw"].min()),
        "demand_max": float(df_enriched["demand_mw"].max()),
        "solar_max": float(df_enriched["solar_mw"].max()),
        "wind_max": float(df_enriched["wind_mw"].max())
    }
    return summary

def preprocess_daily_state_data(
    zenodo_dir: str = "data/raw/zenodo",
    output_dir: str = "data/processed"
) -> dict:
    """
    Cleans and merges Zenodo state-level electricity and weather data for Maharashtra and India total.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Processing Zenodo state-level data...")
    
    # 1. Load files
    df_energy = pd.read_csv(os.path.join(zenodo_dir, "daily_energy_met_MU.csv"))
    df_peak = pd.read_csv(os.path.join(zenodo_dir, "peak_met_MW.csv"))
    df_unmet = pd.read_csv(os.path.join(zenodo_dir, "peak_unmet_MW.csv"))
    df_weather_mh = pd.read_csv(os.path.join(zenodo_dir, "Maharashtra.csv"))
    
    # Convert dates
    df_energy["Date"] = pd.to_datetime(df_energy["Date"])
    df_peak["Date"] = pd.to_datetime(df_peak["Date"])
    df_unmet["Date"] = pd.to_datetime(df_unmet["Date"])
    df_weather_mh["date"] = pd.to_datetime(df_weather_mh["date"])
    
    # Extract Maharashtra and National Total
    df_mh = pd.DataFrame({
        "date": df_energy["Date"],
        "energy_met_mu": df_energy["Maharashtra"],
        "peak_met_mw": df_peak["Maharashtra"],
        "peak_unmet_mw": df_unmet["Maharashtra"]
    })
    
    df_national = pd.DataFrame({
        "date": df_energy["Date"],
        "energy_met_mu": df_energy["Total"],
        "peak_met_mw": df_peak["Total"],
        "peak_unmet_mw": df_unmet["Total"]
    })
    
    # Merge Maharashtra weather features
    weather_cols = [
        "date",
        "2m_temperature_mean", "2m_temperature_max", "2m_temperature_min",
        "surface_solar_radiation_downwards_mean",
        "total_cloud_cover_mean",
        "utci_mean"
    ]
    df_weather_sub = df_weather_mh[weather_cols].copy()
    
    # Convert temperatures from Kelvin to Celsius
    df_weather_sub["temp_c_mean"] = df_weather_sub["2m_temperature_mean"] - 273.15
    df_weather_sub["temp_c_max"] = df_weather_sub["2m_temperature_max"] - 273.15
    df_weather_sub["temp_c_min"] = df_weather_sub["2m_temperature_min"] - 273.15
    df_weather_sub["utci_c_mean"] = df_weather_sub["utci_mean"] - 273.15
    df_weather_sub = df_weather_sub.drop(columns=["2m_temperature_mean", "2m_temperature_max", "2m_temperature_min", "utci_mean"])
    
    df_mh_merged = pd.merge(df_mh, df_weather_sub, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df_national = df_national.sort_values("date").reset_index(drop=True)
    
    # Save processed daily datasets
    mh_path = os.path.join(output_dir, "clean_daily_maharashtra.csv")
    nat_path = os.path.join(output_dir, "clean_daily_national.csv")
    
    df_mh_merged.to_csv(mh_path, index=False)
    df_national.to_csv(nat_path, index=False)
    
    print(f"Saved Maharashtra daily data ({len(df_mh_merged)} days) to {mh_path}")
    print(f"Saved National daily data ({len(df_national)} days) to {nat_path}")
    
    return {
        "maharashtra_records": len(df_mh_merged),
        "national_records": len(df_national),
        "period": f"{df_mh_merged['date'].min().strftime('%Y-%m-%d')} to {df_mh_merged['date'].max().strftime('%Y-%m-%d')}"
    }

if __name__ == "__main__":
    h_summary = preprocess_hourly_data()
    d_summary = preprocess_daily_state_data()
    print("Hourly summary:", json.dumps(h_summary, indent=2))
    print("Daily summary:", json.dumps(d_summary, indent=2))
