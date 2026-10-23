# WattWise AI — System Architecture & Implementation Design
**Academic Course:** Machine Learning Laboratory | Shri Ramdeobaba University, Nagpur
**Students:** Pranav Kolhe (Class: A8, Batch: B4, Roll: 54) & Amber Shukla (Class: A8, Batch: B3, Roll: 38)

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Quality Validation"]
        A1["Grid-India NLDC SCADA (Mendeley Data 10.17632/y58jknpgs8.2)"] --> B1["Clean Hourly National Dataset (12,984 hrs)"]
        A2["Zenodo State Electricity & Weather (10.5281/zenodo.14983362)"] --> B2["Clean Daily Maharashtra Dataset (3,954 days)"]
    end

    subgraph FeatureEngineering ["2. Leakage-Free Feature Engineering"]
        B1 --> C1["Calendar & Festival Pipeline (IMD Seasons, Exact Movable Dates)"]
        C1 --> C2["Strict Shifted Lags (t-24, t-48, t-168, t-336)"]
        C2 --> C3["Shifted Rolling Statistics (24h/7d Mean, Std, Min, Max)"]
        C3 --> D["Chronological Partition (Train: Jan-Nov '24 | Val: Dec '24-Feb '25 | Test: Mar-Jun '25)"]
    end

    subgraph Modeling ["3. Model Benchmark & Analytics Engine"]
        D --> M1["Seasonal-Naive Baseline"]
        D --> M2["Ridge Regression"]
        D --> M3["Decision Tree Regressor"]
        D --> M4["Random Forest Regressor (Best Model)"]
        D --> M5["Gradient Boosting Regressor"]
        D --> M6["PyTorch Deep LSTM Sequence Model"]
        B1 --> CL1["K-Means Profile Clustering (k=3)"]
        B1 --> CL2["Hierarchical Agglomerative Clustering (Ward Linkage)"]
    end

    subgraph Serving ["4. Application & Interface Layer"]
        M4 & M6 & CL1 & CL2 --> API["FastAPI Analytical Server (RESTful Endpoints)"]
        API --> UI["WattWise AI Premium Dashboard (7 Interactive Modules)"]
        API --> SIM["Hypothetical Scenario Simulator Engine"]
    end
```

## Architectural Highlights
1. **Strict Decoupling**: Data engineering, model training, analytical evaluation, and RESTful API endpoints reside in isolated Python packages (`src.data`, `src.models`, `src.analysis`, `src.api`).
2. **Deterministic Reproducibility**: Fixed random seeds (`seed=42`), persistent scaler artifacts (`scaler.joblib`, `y_scaler.joblib`), and serialized weights (`best_lstm.pt`).
3. **Multi-Scale Temporal Analytics**:
   - High-frequency 1-hour resolution for operational load dispatch & day-ahead forecasting.
   - Low-frequency multi-year daily resolution for state infrastructure expansion and peak unmet deficit tracking.
