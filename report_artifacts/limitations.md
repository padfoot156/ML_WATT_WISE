# WattWise AI — Analytical Limitations and Academic Boundaries
**Course Evaluation Integrity Disclosure**

In accordance with rigorous engineering and scientific ethics, the following limitations are explicitly documented:

1. **Analytical Forecasting vs Operational Dispatch**:
   - WattWise AI is an educational time-series forecasting and scenario planning intelligence tool. It does not model generator unit commitment (UC), ramp-rate physical limits, reactive power balance, or transmission line thermal capacity constraints. It is not an official electricity-dispatch engine.
2. **Renewable Generation vs Total Supply**:
   - The primary dataset includes national grid SCADA measurements of solar generation, wind generation, and total aggregate generation. Solar and wind constitute components of clean generation and must never be conflated with total available grid capacity or spinning reserve.
3. **Hypothetical Expansion Scenarios**:
   - Scenarios evaluated in the Expansion Planner (e.g., annual demand growth of 5.5% to 8.5%, renewable capacity growth of 12% to 22%) are user-steered exploratory simulations designed to test grid sensitivity. They do not represent official Government of India (CEA/Ministry of Power) policy commitments or guaranteed future outcomes.
4. **Spatial Aggregation Resolution**:
   - National SCADA hourly data aggregates consumption across all five regional grids (Northern, Western, Southern, Eastern, North-Eastern). While highly effective for macro-grid load forecasting, it averages out localized transmission bottlenecks and intra-state distribution network losses.
5. **Weather-Load Coupling**:
   - High-resolution ERA5 reanalysis data is integrated at the daily state level for Maharashtra (2013-2024). For operational real-time hourly forecasting, numerical weather prediction (NWP) forecasts are required rather than retrospective reanalysis observations.
