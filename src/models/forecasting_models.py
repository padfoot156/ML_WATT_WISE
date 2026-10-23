"""
WattWise AI - Machine Learning Demand Forecasting Pipeline
Implements:
1. Seasonal-Naive Baseline
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. PyTorch Deep LSTM Sequence Model

Chronological train/val/test splits, zero leakage, scaler fit strictly on training data.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Feature definition
FEATURE_COLS = [
    # Calendar & Cyclical
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos",
    "is_weekend", "season_code",
    "is_festival", "is_public_holiday",
    "is_pre_festival_window", "is_post_festival_window",
    "days_to_nearest_festival",
    # Historical Lagged Demand (shifted >= 24h)
    "demand_lag_24", "demand_lag_48", "demand_lag_168", "demand_lag_336",
    # Rolling Historical Demand Statistics
    "demand_roll_mean_24h", "demand_roll_std_24h",
    "demand_roll_min_24h", "demand_roll_max_24h",
    "demand_roll_mean_7d", "demand_roll_std_7d",
    # Historical Generation Lags
    "solar_lag_24", "wind_lag_24", "total_gen_lag_24"
]

TARGET_COL = "demand_mw"

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0

# ----------------- PyTorch LSTM Architecture -----------------
class DemandLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(DemandLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :] # Take last time step
        x = self.relu(self.fc1(last_hidden))
        out = self.fc2(x)
        return out.squeeze(-1)

class TimeSeriesSeqDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X_data, y_data, seq_len=24):
    """Creates sequence windows of length seq_len for LSTM."""
    Xs, ys = [], []
    for i in range(len(X_data) - seq_len):
        Xs.append(X_data[i:(i + seq_len)])
        ys.append(y_data[i + seq_len])
    return np.array(Xs), np.array(ys)

# ----------------- Training Pipeline -----------------
def run_forecasting_experiments(data_dir: str = "data/processed", models_dir: str = "models", results_dir: str = "results"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("report_artifacts", exist_ok=True)

    print("Loading preprocessed splits...")
    train_df = pd.read_csv(os.path.join(data_dir, "hourly_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "hourly_val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "hourly_test.csv"))

    # Feature matrices
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].values

    # Scaler fit STRICTLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature list
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    results_table = []
    test_predictions = {
        "datetime": test_df["datetime"].tolist(),
        "actual_demand_mw": y_test.tolist()
    }

    # 1. Seasonal Naive Baseline (24-hour seasonal lag)
    print("\n--- Evaluating Seasonal-Naive Baseline (Lag-24) ---")
    t0 = time.time()
    y_pred_naive = test_df["demand_lag_24"].values
    t_naive = time.time() - t0
    
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    r2_naive = r2_score(y_test, y_pred_naive)
    mape_naive = calculate_mape(y_test, y_pred_naive)
    
    results_table.append({
        "Model": "Seasonal Naive (Lag-24)",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_naive, 2),
        "RMSE (MW)": round(rmse_naive, 2),
        "R2": round(r2_naive, 4),
        "MAPE (%)": round(mape_naive, 2),
        "Training Time (s)": round(t_naive, 3),
        "Test Period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "Model Configuration": "Persistence of lag_24"
    })
    test_predictions["naive_pred"] = y_pred_naive.tolist()

    # 2. Ridge Regression
    print("\n--- Training Ridge Regression ---")
    t0 = time.time()
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train_scaled, y_train)
    t_ridge = time.time() - t0
    
    y_pred_ridge = ridge.predict(X_test_scaled)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mape_ridge = calculate_mape(y_test, y_pred_ridge)
    
    results_table.append({
        "Model": "Ridge Regression",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_ridge, 2),
        "RMSE (MW)": round(rmse_ridge, 2),
        "R2": round(r2_ridge, 4),
        "MAPE (%)": round(mape_ridge, 2),
        "Training Time (s)": round(t_ridge, 3),
        "Test Period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "Model Configuration": "alpha=100.0, L2 regularization"
    })
    joblib.dump(ridge, os.path.join(models_dir, "ridge_model.joblib"))
    test_predictions["ridge_pred"] = y_pred_ridge.tolist()

    # 3. Decision Tree Regressor
    print("\n--- Training Decision Tree Regressor ---")
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=12, min_samples_split=20, min_samples_leaf=10, random_state=42)
    dt.fit(X_train, y_train)
    t_dt = time.time() - t0
    
    y_pred_dt = dt.predict(X_test)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    r2_dt = r2_score(y_test, y_pred_dt)
    mape_dt = calculate_mape(y_test, y_pred_dt)
    
    results_table.append({
        "Model": "Decision Tree Regressor",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_dt, 2),
        "RMSE (MW)": round(rmse_dt, 2),
        "R2": round(r2_dt, 4),
        "MAPE (%)": round(mape_dt, 2),
        "Training Time (s)": round(t_dt, 3),
        "Test Period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "Model Configuration": "max_depth=12, min_samples_split=20, min_samples_leaf=10"
    })
    joblib.dump(dt, os.path.join(models_dir, "decision_tree_model.joblib"))
    test_predictions["decision_tree_pred"] = y_pred_dt.tolist()

    # 4. Random Forest Regressor
    print("\n--- Training Random Forest Regressor ---")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    t_rf = time.time() - t0
    
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    mape_rf = calculate_mape(y_test, y_pred_rf)
    
    results_table.append({
        "Model": "Random Forest Regressor",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_rf, 2),
        "RMSE (MW)": round(rmse_rf, 2),
        "R2": round(r2_rf, 4),
        "MAPE (%)": round(mape_rf, 2),
        "Training Time (s)": round(t_rf, 3),
        "Test Period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "Model Configuration": "n_estimators=100, max_depth=16, min_samples_leaf=5"
    })
    joblib.dump(rf, os.path.join(models_dir, "random_forest_model.joblib"))
    test_predictions["random_forest_pred"] = y_pred_rf.tolist()

    # Save feature importances from Random Forest
    importances_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importances_df.to_csv("report_artifacts/feature_importance.csv", index=False)
    print("Saved feature importance rankings to report_artifacts/feature_importance.csv")

    # 5. Gradient Boosting Regressor
    print("\n--- Training Gradient Boosting Regressor ---")
    t0 = time.time()
    gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42)
    gbr.fit(X_train, y_train)
    t_gbr = time.time() - t0
    
    y_pred_gbr = gbr.predict(X_test)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
    r2_gbr = r2_score(y_test, y_pred_gbr)
    mape_gbr = calculate_mape(y_test, y_pred_gbr)
    
    results_table.append({
        "Model": "Gradient Boosting Regressor",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_gbr, 2),
        "RMSE (MW)": round(rmse_gbr, 2),
        "R2": round(r2_gbr, 4),
        "MAPE (%)": round(mape_gbr, 2),
        "Training Time (s)": round(t_gbr, 3),
        "Test Period": f"{test_df['datetime'].min()} to {test_df['datetime'].max()}",
        "Model Configuration": "n_estimators=120, lr=0.08, max_depth=6"
    })
    joblib.dump(gbr, os.path.join(models_dir, "gradient_boosting_model.joblib"))
    test_predictions["gradient_boosting_pred"] = y_pred_gbr.tolist()

    # 6. PyTorch LSTM Deep Sequence Model
    print("\n--- Training PyTorch LSTM Sequence Model ---")
    seq_len = 24
    
    # Scale target for LSTM stability
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    joblib.dump(y_scaler, os.path.join(models_dir, "y_scaler.joblib"))

    # Create sequence datasets
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len=seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len=seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_len=seq_len)

    train_loader = DataLoader(TimeSeriesSeqDataset(X_train_seq, y_train_seq), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesSeqDataset(X_val_seq, y_val_seq), batch_size=64, shuffle=False)
    test_loader = DataLoader(TimeSeriesSeqDataset(X_test_seq, y_test_seq), batch_size=64, shuffle=False)

    torch.manual_seed(42)
    device = torch.device("cpu")
    lstm_model = DemandLSTM(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.002, weight_decay=1e-5)

    epochs = 20
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    training_history = {"epoch": [], "train_loss": [], "val_loss": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        lstm_model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = lstm_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        lstm_model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = lstm_model(bx)
                loss = criterion(out, by)
                val_losses.append(loss.item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        training_history["epoch"].append(ep)
        training_history["train_loss"].append(mean_train_loss)
        training_history["val_loss"].append(mean_val_loss)

        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            patience_counter = 0
            torch.save(lstm_model.state_dict(), os.path.join(models_dir, "best_lstm.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep}")
                break

    t_lstm = time.time() - t0

    # Save LSTM loss history
    with open(os.path.join(results_dir, "lstm_loss_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    # Load best LSTM weights for test evaluation
    lstm_model.load_state_dict(torch.load(os.path.join(models_dir, "best_lstm.pt")))
    lstm_model.eval()
    lstm_preds_scaled = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = lstm_model(bx)
            lstm_preds_scaled.extend(out.cpu().numpy().tolist())

    y_pred_lstm = y_scaler.inverse_transform(np.array(lstm_preds_scaled).reshape(-1, 1)).flatten()
    # Align test targets with sequence length offset
    y_test_lstm_actual = y_test[seq_len:]
    test_df_lstm = test_df.iloc[seq_len:].reset_index(drop=True)

    mae_lstm = mean_absolute_error(y_test_lstm_actual, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_actual, y_pred_lstm))
    r2_lstm = r2_score(y_test_lstm_actual, y_pred_lstm)
    mape_lstm = calculate_mape(y_test_lstm_actual, y_pred_lstm)

    results_table.append({
        "Model": "LSTM Neural Network",
        "Forecast Horizon": "24-Hour Day-Ahead",
        "MAE (MW)": round(mae_lstm, 2),
        "RMSE (MW)": round(rmse_lstm, 2),
        "R2": round(r2_lstm, 4),
        "MAPE (%)": round(mape_lstm, 2),
        "Training Time (s)": round(t_lstm, 3),
        "Test Period": f"{test_df_lstm['datetime'].min()} to {test_df_lstm['datetime'].max()}",
        "Model Configuration": f"2-layer LSTM(hidden=64, dropout=0.2), seq_len={seq_len}, Adam(lr=0.002)"
    })

    # Pad or store LSTM predictions
    test_predictions["lstm_pred"] = [None] * seq_len + y_pred_lstm.tolist()

    # ----------------- Save Deliverables -----------------
    results_df = pd.DataFrame(results_table)
    print("\n================== DEMAND FORECASTING COMPARISON TABLE ==================")
    print(results_df.to_string(index=False))

    results_df.to_csv("report_artifacts/forecast_model_comparison.csv", index=False)
    results_df.to_json("report_artifacts/forecast_model_comparison.json", orient="records", indent=2)

    # Save test predictions
    pred_df = pd.DataFrame(test_predictions)
    pred_df.to_csv("report_artifacts/forecast_predictions.csv", index=False)
    pred_df.to_csv(os.path.join(results_dir, "forecast_predictions.csv"), index=False)

    # Identify and save best model
    best_row = results_df.sort_values("RMSE (MW)").iloc[0].to_dict()
    with open("report_artifacts/best_model_metrics.json", "w") as f:
        json.dump(best_row, f, indent=2)
    print(f"\nBest Model by RMSE: {best_row['Model']} (RMSE: {best_row['RMSE (MW)']} MW, MAE: {best_row['MAE (MW)']} MW, R2: {best_row['R2']})")

    return results_df

if __name__ == "__main__":
    run_forecasting_experiments()
