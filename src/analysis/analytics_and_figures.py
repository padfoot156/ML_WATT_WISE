"""
WattWise AI - Empirical Analytics and Publication-Grade Data Visualizations
Generates all analytical charts and statistical summaries for the report and dashboard:
- Time series & heatmaps
- Festival impact comparison
- Model comparison and LSTM loss curves
- Renewable generation analysis
- Long-term state expansion & unmet demand trends
- Hypothetical expansion scenarios
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set matplotlib backend and styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_all_analytics_and_figures():
    print("Generating comprehensive energy analytics and report figures...")
    
    # 1. Load Datasets
    hourly_df = pd.read_csv("data/processed/clean_hourly_national.csv")
    hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])
    hourly_df["date"] = hourly_df["datetime"].dt.date
    
    mh_daily_df = pd.read_csv("data/processed/clean_daily_maharashtra.csv")
    mh_daily_df["date"] = pd.to_datetime(mh_daily_df["date"])
    
    preds_df = pd.read_csv("results/forecast_predictions.csv")
    preds_df["datetime"] = pd.to_datetime(preds_df["datetime"])
    
    with open("results/lstm_loss_history.json") as f:
        lstm_history = json.load(f)
        
    models_comp_df = pd.read_csv("report_artifacts/forecast_model_comparison.csv")

    # -------------------------------------------------------------
    # Figure 1: Historical Demand Time Series with Seasonal Envelope
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    daily_avg_demand = hourly_df.groupby("date")["demand_mw"].agg(["mean", "min", "max"])
    dates = pd.to_datetime(daily_avg_demand.index)
    
    plt.plot(dates, daily_avg_demand["mean"] / 1000.0, label="Daily Mean Demand (GW)", color="#0284c7", lw=1.8)
    plt.fill_between(dates, daily_avg_demand["min"] / 1000.0, daily_avg_demand["max"] / 1000.0, 
                     color="#38bdf8", alpha=0.25, label="Diurnal Min-Max Envelope")
    plt.title("WattWise AI — Historical All-India Electricity Demand (2024–2025)", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Demand (GW)", fontsize=11)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "historical_demand_timeseries.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print("Saved:", f1_path)

    # -------------------------------------------------------------
    # Figure 2: Hourly and Weekly Demand Heatmap Matrix
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    pivot_week_hour = hourly_df.groupby(["dayofweek", "hour"])["demand_mw"].mean().unstack() / 1000.0
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    im = plt.imshow(pivot_week_hour.values, aspect="auto", cmap="YlGnBu", origin="lower")
    plt.colorbar(im, label="Average Load (GW)")
    plt.yticks(ticks=np.arange(7), labels=day_labels, fontsize=10)
    plt.xticks(ticks=np.arange(0, 24, 2), labels=[f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=10)
    plt.title("WattWise AI — Hourly & Weekly Grid Demand Heatmap Matrix", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (IST)", fontsize=11)
    plt.ylabel("Day of Week", fontsize=11)
    plt.tight_layout()
    f2_path = os.path.join(FIGURES_DIR, "hourly_weekly_demand_heatmap.png")
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print("Saved:", f2_path)

    # -------------------------------------------------------------
    # Figure 3: Seasonal Demand Profiles (IMD Definitions)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    season_colors = {"Winter": "#0284c7", "Summer": "#ef4444", "Monsoon": "#10b981", "Post-Monsoon": "#f59e0b"}
    
    for s_name, color in season_colors.items():
        s_data = hourly_df[hourly_df["season"] == s_name]
        if len(s_data) > 0:
            hourly_season = s_data.groupby("hour")["demand_mw"].mean() / 1000.0
            plt.plot(hourly_season.index, hourly_season.values, label=f"{s_name} (n={len(s_data)} hrs)", 
                     color=color, lw=2.4)
            
    plt.title("WattWise AI — Diurnal Demand Curves by IMD Meteorological Seasons", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (IST)", fontsize=11)
    plt.ylabel("Average Demand (GW)", fontsize=11)
    plt.xticks(np.arange(0, 24))
    plt.legend(frameon=True)
    plt.tight_layout()
    f3_path = os.path.join(FIGURES_DIR, "seasonal_demand_profiles.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print("Saved:", f3_path)

    # -------------------------------------------------------------
    # Figure 4: Festival vs Baseline Demand Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 5.5))
    fest_summary = []
    major_fests = [
        "Diwali (Deepavali / Lakshmi Puja)", "Holi", "Ganesh Chaturthi",
        "Dussehra (Vijayadashami)", "Eid al-Fitr", "Independence Day", "Republic Day"
    ]
    
    comp_results = []
    for fest in major_fests:
        fest_days = hourly_df[hourly_df["festival_name"] == fest]
        if len(fest_days) > 0:
            fest_demand = fest_days["demand_mw"].mean() / 1000.0
            fest_month = fest_days["month"].iloc[0]
            # Baseline: same day of week in same month excluding festival window
            baseline_days = hourly_df[
                (hourly_df["month"] == fest_month) & 
                (hourly_df["is_festival"] == 0) & 
                (hourly_df["dayofweek"].isin(fest_days["dayofweek"].unique()))
            ]
            base_demand = baseline_days["demand_mw"].mean() / 1000.0 if len(baseline_days) > 0 else hourly_df["demand_mw"].mean() / 1000.0
            diff_pct = ((fest_demand - base_demand) / base_demand) * 100.0
            
            comp_results.append({
                "festival": fest.split(" (")[0],
                "festival_demand_gw": round(fest_demand, 2),
                "baseline_demand_gw": round(base_demand, 2),
                "diff_pct": round(diff_pct, 2)
            })

    comp_df = pd.DataFrame(comp_results)
    x = np.arange(len(comp_df))
    width = 0.35
    
    plt.bar(x - width/2, comp_df["festival_demand_gw"], width, label="Festival Day Demand", color="#f97316")
    plt.bar(x + width/2, comp_df["baseline_demand_gw"], width, label="Comparable Non-Festival Baseline", color="#64748b")
    
    for i in range(len(comp_df)):
        diff = comp_df["diff_pct"].iloc[i]
        sign = "+" if diff > 0 else ""
        plt.text(i, max(comp_df["festival_demand_gw"].iloc[i], comp_df["baseline_demand_gw"].iloc[i]) + 2,
                 f"{sign}{diff:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#0f172a")

    plt.xticks(x, comp_df["festival"], rotation=15, ha="right", fontsize=10)
    plt.ylabel("Mean Demand (GW)", fontsize=11)
    plt.ylim(0, max(comp_df["festival_demand_gw"].max(), comp_df["baseline_demand_gw"].max()) * 1.15)
    plt.title("WattWise AI — Electricity Demand: Festivals vs Comparable Baseline Periods", fontsize=13, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()
    f4_path = os.path.join(FIGURES_DIR, "festival_demand_comparison.png")
    plt.savefig(f4_path, dpi=200)
    plt.close()
    print("Saved:", f4_path)
    comp_df.to_csv("report_artifacts/festival_impact_empirical_summary.csv", index=False)

    # -------------------------------------------------------------
    # Figure 5: Actual vs Predicted Demand (Held-Out Test Sample)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5.5))
    # Pick a 14-day window in held-out test set (e.g. May 2024 peak heatwave)
    test_slice = preds_df.dropna().iloc[1000:1000 + 336] # 14 days = 336 hours
    
    plt.plot(test_slice["datetime"], test_slice["actual_demand_mw"] / 1000.0, label="Actual Observed Demand (GW)", color="#0f172a", lw=2)
    plt.plot(test_slice["datetime"], test_slice["random_forest_pred"] / 1000.0, label="Random Forest (Best Model)", color="#0284c7", lw=1.8, linestyle="--")
    plt.plot(test_slice["datetime"], test_slice["lstm_pred"] / 1000.0, label="PyTorch LSTM Forecast", color="#10b981", lw=1.5, alpha=0.85)
    plt.plot(test_slice["datetime"], test_slice["naive_pred"] / 1000.0, label="Seasonal-Naive Baseline (Lag-24)", color="#94a3b8", lw=1.2, linestyle=":")
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.title("WattWise AI — 24-Hour Ahead Forecast vs Actual Demand (Held-Out Test Window)", fontsize=13, fontweight="bold")
    plt.xlabel("Timestamp (IST)", fontsize=11)
    plt.ylabel("Electricity Demand (GW)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f5_path = os.path.join(FIGURES_DIR, "actual_vs_predicted_demand.png")
    plt.savefig(f5_path, dpi=200)
    plt.close()
    print("Saved:", f5_path)

    # -------------------------------------------------------------
    # Figure 6: Forecast Residuals Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    residuals_rf = (test_slice["actual_demand_mw"] - test_slice["random_forest_pred"]) / 1000.0
    residuals_lstm = (test_slice["actual_demand_mw"] - test_slice["lstm_pred"]) / 1000.0
    
    plt.hist(residuals_rf, bins=30, alpha=0.6, color="#0284c7", label=f"Random Forest (Mean={residuals_rf.mean():.2f} GW, Std={residuals_rf.std():.2f} GW)")
    plt.hist(residuals_lstm, bins=30, alpha=0.5, color="#10b981", label=f"LSTM (Mean={residuals_lstm.mean():.2f} GW, Std={residuals_lstm.std():.2f} GW)")
    plt.axvline(0, color="#ef4444", linestyle="--", lw=1.5, label="Zero Error Line")
    
    plt.title("WattWise AI — Forecast Residuals Distribution (GW)", fontsize=13, fontweight="bold")
    plt.xlabel("Residual Error: Actual - Forecast (GW)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    f6_path = os.path.join(FIGURES_DIR, "forecast_residuals.png")
    plt.savefig(f6_path, dpi=200)
    plt.close()
    print("Saved:", f6_path)

    # -------------------------------------------------------------
    # Figure 7: Model Comparison Metrics (MAE & RMSE)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    models_sorted = models_comp_df.sort_values("RMSE (MW)", ascending=False)
    y_pos = np.arange(len(models_sorted))
    bar_height = 0.35
    
    plt.barh(y_pos + bar_height/2, models_sorted["RMSE (MW)"] / 1000.0, bar_height, label="RMSE (GW)", color="#ef4444")
    plt.barh(y_pos - bar_height/2, models_sorted["MAE (MW)"] / 1000.0, bar_height, label="MAE (GW)", color="#0284c7")
    
    for i, v in enumerate(models_sorted["RMSE (MW)"] / 1000.0):
        plt.text(v + 0.1, i + bar_height/2, f"{v:.2f} GW", va="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(models_sorted["MAE (MW)"] / 1000.0):
        plt.text(v + 0.1, i - bar_height/2, f"{v:.2f} GW", va="center", fontsize=9)

    plt.yticks(y_pos, models_sorted["Model"], fontsize=10)
    plt.xlabel("Error Magnitude on Held-Out Test Set (GW)", fontsize=11)
    plt.xlim(0, max(models_sorted["RMSE (MW)"] / 1000.0) * 1.2)
    plt.title("WattWise AI — Forecasting Model Benchmark: MAE & RMSE", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    f7_path = os.path.join(FIGURES_DIR, "forecast_model_metrics_comparison.png")
    plt.savefig(f7_path, dpi=200)
    plt.close()
    print("Saved:", f7_path)

    # -------------------------------------------------------------
    # Figure 8: PyTorch LSTM Training & Validation Loss Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    eps = lstm_history["epoch"]
    plt.plot(eps, lstm_history["train_loss"], marker="o", lw=2, color="#0284c7", label="Training Loss (MSE)")
    plt.plot(eps, lstm_history["val_loss"], marker="s", lw=2, color="#ef4444", label="Validation Loss (MSE)")
    
    best_ep = int(np.argmin(lstm_history["val_loss"])) + 1
    best_loss = min(lstm_history["val_loss"])
    plt.scatter([best_ep], [best_loss], s=120, color="#f59e0b", zorder=5, label=f"Best Model (Epoch {best_ep}: {best_loss:.4f})")
    
    plt.title("WattWise AI — PyTorch LSTM Sequence Model: Learning Curves", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Mean Squared Error (Standardized)", fontsize=11)
    plt.xticks(eps)
    plt.legend(frameon=True)
    plt.tight_layout()
    f8_path = os.path.join(FIGURES_DIR, "lstm_training_validation_loss.png")
    plt.savefig(f8_path, dpi=200)
    plt.close()
    print("Saved:", f8_path)

    # -------------------------------------------------------------
    # Figure 9: Renewable Generation Trends & Solar/Wind Diurnal Output
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 9A: Average Diurnal Solar vs Wind Profile
    diurnal_re = hourly_df.groupby("hour")[["solar_mw", "wind_mw"]].mean() / 1000.0
    ax1.plot(diurnal_re.index, diurnal_re["solar_mw"], label="Solar Generation (GW)", color="#f59e0b", lw=2.5)
    ax1.plot(diurnal_re.index, diurnal_re["wind_mw"], label="Wind Generation (GW)", color="#06b6d4", lw=2.5)
    ax1.set_title("Diurnal Complementarity (Solar vs Wind)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Hour of Day (IST)", fontsize=10)
    ax1.set_ylabel("Average Output (GW)", fontsize=10)
    ax1.set_xticks(np.arange(0, 24, 2))
    ax1.legend(frameon=True)
    
    # 9B: Monthly Renewable Share Percentage
    hourly_df["year_month"] = hourly_df["datetime"].dt.to_period("M")
    monthly_re = hourly_df.groupby("year_month")["renewable_share_pct"].mean()
    m_labels = [str(p) for p in monthly_re.index]
    
    ax2.bar(m_labels, monthly_re.values, color="#10b981", alpha=0.85)
    ax2.set_title("Monthly Observed Renewable Share (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year-Month", fontsize=10)
    ax2.set_ylabel("Renewable Generation / Total (%)", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    
    for i, v in enumerate(monthly_re.values):
        ax2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8.5)

    plt.suptitle("WattWise AI — Observed Solar & Wind Renewable Generation Intelligence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f9_path = os.path.join(FIGURES_DIR, "renewable_generation_trends.png")
    plt.savefig(f9_path, dpi=200)
    plt.close()
    print("Saved:", f9_path)

    # -------------------------------------------------------------
    # Figure 10: Long-Term Maharashtra Demand Growth (2013-2024)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # Peak Met vs Unmet
    ax1.plot(mh_daily_df["date"], mh_daily_df["peak_met_mw"] / 1000.0, label="Peak Power Supplied (GW)", color="#0284c7", lw=1.5)
    ax1.set_ylabel("Peak Supplied (GW)", fontsize=10)
    ax1.set_title("Maharashtra Electricity Grid Expansion: Peak Met vs Peak Unmet (2013–2024)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    
    ax2.fill_between(mh_daily_df["date"], 0, mh_daily_df["peak_unmet_mw"], color="#ef4444", alpha=0.7, label="Peak Deficit / Unmet Demand (MW)")
    ax2.set_ylabel("Unmet Deficit (MW)", fontsize=10)
    ax2.set_xlabel("Year", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    f10_path = os.path.join(FIGURES_DIR, "maharashtra_expansion_trends.png")
    plt.savefig(f10_path, dpi=200)
    plt.close()
    print("Saved:", f10_path)

    # -------------------------------------------------------------
    # Figure 11: Hypothetical Expansion Scenario Simulator Projections
    # -------------------------------------------------------------
    plt.figure(figsize=(11, 5.5))
    
    # Base Year 2024 metrics:
    base_year = 2024
    base_annual_demand_twh = (hourly_df[hourly_df["datetime"].dt.year == 2024]["demand_mw"].sum() * 1.0) / 1e6
    base_peak_gw = hourly_df[hourly_df["datetime"].dt.year == 2024]["demand_mw"].max() / 1000.0
    base_re_twh = (hourly_df[hourly_df["datetime"].dt.year == 2024]["renewable_mw"].sum() * 1.0) / 1e6
    
    future_years = np.arange(2024, 2031)
    
    # 3 Scenarios:
    # A: Business As Usual (BAU): Demand growth 5.5%/yr, RE growth 12%/yr
    # B: High Electrification & Accelerated RE: Demand growth 7.5%/yr, RE growth 22%/yr
    # C: Stagnant Renewables & Climate Stress: Demand growth 8.5%/yr, RE growth 5%/yr
    
    scenarios = {
        "BAU (5.5% Demand, 12% RE)": {"d_rate": 0.055, "re_rate": 0.12, "color": "#0284c7", "style": "-"},
        "Accelerated Green Transition (7.5% Demand, 22% RE)": {"d_rate": 0.075, "re_rate": 0.22, "color": "#10b981", "style": "--"},
        "High Stress / Stagnant RE (8.5% Demand, 5% RE)": {"d_rate": 0.085, "re_rate": 0.05, "color": "#ef4444", "style": ":"}
    }
    
    scenario_records = []
    for s_name, params in scenarios.items():
        demands = [base_peak_gw * ((1 + params["d_rate"]) ** (yr - 2024)) for yr in future_years]
        plt.plot(future_years, demands, label=s_name, color=params["color"], linestyle=params["style"], lw=2.4)
        for yr, dem in zip(future_years, demands):
            scenario_records.append({
                "scenario": s_name,
                "year": int(yr),
                "peak_demand_gw": round(float(dem), 2),
                "assumed_demand_growth_pct": round(params["d_rate"] * 100, 1),
                "assumed_re_growth_pct": round(params["re_rate"] * 100, 1)
            })

    plt.axvline(2024, color="#64748b", linestyle="--", label="Historical Baseline (2024)")
    plt.scatter([2024], [base_peak_gw], color="#0f172a", s=60, zorder=5)
    plt.text(2024.1, base_peak_gw - 5, f"Observed Peak 2024\n{base_peak_gw:.1f} GW", fontsize=9, fontweight="bold")
    
    plt.title("WattWise AI — Hypothetical Peak Demand Expansion Scenarios (2024–2030)", fontsize=13, fontweight="bold")
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Projected Peak Demand (GW)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    f11_path = os.path.join(FIGURES_DIR, "expansion_scenario_simulation.png")
    plt.savefig(f11_path, dpi=200)
    plt.close()
    print("Saved:", f11_path)
    
    with open("report_artifacts/scenario_assumptions.json", "w") as f:
        json.dump(scenario_records, f, indent=2)

    print("All 11 analytical figures and data summaries generated successfully!")

if __name__ == "__main__":
    generate_all_analytics_and_figures()
