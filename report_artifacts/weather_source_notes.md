# Meteorological Data Sources and Thermal Comfort Index Documentation
**WattWise AI — Energy Demand, Supply & Expansion Intelligence**

## 1. ERA5 Atmospheric Reanalysis Source
State-level meteorological parameters are acquired from the Zenodo benchmark repository:
- **DOI**: 10.5281/zenodo.14983362
- **Origin**: European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 atmospheric reanalysis.
- **Spatial Scope**: Land-weighted spatial aggregations for the State of Maharashtra (covering Vidarbha/Nagpur, Marathwada, Western Maharashtra, and Konkan).
- **Temporal Scope**: Daily frequency synchronized with state electricity grid dispatch (2013-2024).

## 2. Parameter Definitions and Conversions
1. **2m Air Temperature**:
   - `2m_temperature_mean`, `2m_temperature_max`, `2m_temperature_min`
   - Converted from Kelvin to Celsius: $T_{°C} = T_K - 273.15$.
2. **Universal Thermal Climate Index (UTCI)**:
   - Evaluates physiological thermal stress on the human body combining ambient temperature, mean radiant temperature, wind speed, and relative humidity.
   - High UTCI in central India (>35°C in April-June) corresponds to severe heat stress and triggers unprecedented cooling demand (air conditioners, agricultural irrigation pumps).
3. **Surface Solar Radiation Downwards (SSRD)**:
   - Measured in Joules per square meter ($J/m^2$); tracks daily global horizontal irradiance influencing solar PV generation capacity.
4. **Total Cloud Cover (TCC)**:
   - Dimensionless fraction $[0, 1]$ indicating atmospheric cloud obstruction.

## 3. Spatial Aggregation Boundaries
When modeling national-level SCADA demand, local weather variations across diverse climatic zones (Himalayan, Indo-Gangetic, Coastal, Deccan Plateau) exhibit localized counter-balancing effects. For national forecasting, calendar features and lagged load statistics serve as primary drivers, while state-level models directly utilize regional ERA5 meteorological parameters.
