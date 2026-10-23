# WattWise AI — Forecasting Methodology and Leakage Prevention
**Machine Learning Laboratory Mini-Project Methodology Report**

## 1. Time-Series Partitioning vs Random Splitting
Standard cross-validation (e.g., K-Fold) assumes independent and identically distributed (i.i.d.) data. Applying random shuffling to electricity load time-series leads to catastrophic data leakage: the model interpolates between adjacent hours rather than learning true forecasting dynamics.

In WattWise AI:
- **Chronological Training Period**: 2024-01-15 to 2024-11-30 (7,704 hourly records). Covers full seasonal variation and major Indian festivals (Makar Sankranti, Maha Shivratri, Holi, Eid, Independence Day, Ganesh Chaturthi, Navratri, Dussehra, Diwali).
- **Chronological Validation Period**: 2024-12-01 to 2025-02-28 (2,160 hourly records). Used strictly for LSTM early stopping (patience=5) and hyperparameter selection.
- **Chronological Held-Out Test Period**: 2025-03-01 to 2025-06-24 (2,784 hourly records). Preserved untouched until final evaluation; evaluates performance during late spring and summer heatwaves where peak grid load reaches national records (>248 GW).

## 2. Mathematical Leakage Prevention Proof
Let $y_t$ represent electricity demand at timestamp $t$. In a 24-hour day-ahead forecasting formulation:
$$\hat{y}_{t+h} = f(\mathcal{X}_t), \quad orall h \in [1, 24]$$
where information set $\mathcal{X}_t$ must strictly satisfy:
$$\mathcal{X}_t \subseteq \{ y_	au, \mathbf{g}_	au \mid 	au \le t \} \cup \{ \mathbf{c}_{t+h} \}$$
Here:
- $\mathbf{c}_{t+h}$ represents deterministic calendar features (hour, weekday, month, festival tag) which are known in advance without uncertainty.
- $\mathbf{g}_	au$ represents generation indicators observed up to issue time $t$.
- All historical demand features are constructed via $	ext{shift}(k)$ where $k \ge 24$. Specifically:
  $$	ext{demand\_lag\_24} = y_{t - 24 + h - 24} = y_t \quad (	ext{for } h=24)$$
  $$	ext{demand\_roll\_mean\_24h} = rac{1}{24} \sum_{i=0}^{23} y_{t - i}$$
No target observation $y_{t+h}$ or future window statistic enters the feature vector.

## 3. Scaling Transformation Isolation
Scalers (`StandardScaler`) are fit strictly on training partitions:
$$\mu_{	ext{train}} = rac{1}{N_{	ext{train}}} \sum_{i \in 	ext{train}} x_i, \quad \sigma_{	ext{train}} = \sqrt{rac{1}{N_{	ext{train}}} \sum_{i \in 	ext{train}} (x_i - \mu_{	ext{train}})^2}$$
Validation and test sets are transformed using $\mu_{	ext{train}}$ and $\sigma_{	ext{train}}$, ensuring zero statistical leakage from the evaluation period.
