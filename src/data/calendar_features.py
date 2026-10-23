"""
WattWise AI - Indian Calendar & Festival Feature Engineering
Contains verified Indian National Holidays, Gazetted Holidays, and Major Festivals (2024-2025)
Standardized seasonal definitions per India Meteorological Department (IMD).
"""

import pandas as pd
import numpy as np
from datetime import datetime, date

# Verified Indian Festivals and National Holidays for 2024 and 2025
INDIAN_EVENTS = {
    # 2024 Events
    date(2024, 1, 1): {"name": "New Year's Day", "category": "Observance"},
    date(2024, 1, 15): {"name": "Makar Sankranti / Pongal", "category": "Major Festival"},
    date(2024, 1, 26): {"name": "Republic Day", "category": "National Holiday"},
    date(2024, 3, 8): {"name": "Maha Shivratri", "category": "Major Festival"},
    date(2024, 3, 25): {"name": "Holi", "category": "Major Festival"},
    date(2024, 3, 29): {"name": "Good Friday", "category": "Gazetted Holiday"},
    date(2024, 4, 9): {"name": "Gudi Padwa / Ugadi", "category": "Major Festival"},
    date(2024, 4, 11): {"name": "Eid al-Fitr", "category": "Major Festival"},
    date(2024, 4, 17): {"name": "Ram Navami", "category": "Major Festival"},
    date(2024, 4, 21): {"name": "Mahavir Jayanti", "category": "Gazetted Holiday"},
    date(2024, 5, 23): {"name": "Buddha Purnima", "category": "Gazetted Holiday"},
    date(2024, 6, 17): {"name": "Eid al-Adha (Bakrid)", "category": "Major Festival"},
    date(2024, 7, 17): {"name": "Muharram", "category": "Gazetted Holiday"},
    date(2024, 8, 15): {"name": "Independence Day", "category": "National Holiday"},
    date(2024, 8, 19): {"name": "Raksha Bandhan", "category": "Major Festival"},
    date(2024, 8, 26): {"name": "Krishna Janmashtami", "category": "Major Festival"},
    date(2024, 9, 7): {"name": "Ganesh Chaturthi", "category": "Major Festival"},
    date(2024, 10, 2): {"name": "Gandhi Jayanti", "category": "National Holiday"},
    date(2024, 10, 3): {"name": "Navratri Ghatasthapana", "category": "Major Festival"},
    date(2024, 10, 11): {"name": "Maha Navami", "category": "Major Festival"},
    date(2024, 10, 12): {"name": "Dussehra (Vijayadashami)", "category": "Major Festival"},
    date(2024, 10, 29): {"name": "Dhanteras", "category": "Major Festival"},
    date(2024, 11, 1): {"name": "Diwali (Deepavali / Lakshmi Puja)", "category": "Major Festival"},
    date(2024, 11, 2): {"name": "Govardhan Puja", "category": "Major Festival"},
    date(2024, 11, 3): {"name": "Bhai Dooj", "category": "Major Festival"},
    date(2024, 11, 7): {"name": "Chhath Puja", "category": "Major Festival"},
    date(2024, 11, 15): {"name": "Guru Nanak Jayanti", "category": "Gazetted Holiday"},
    date(2024, 12, 25): {"name": "Christmas", "category": "Gazetted Holiday"},

    # 2025 Events
    date(2025, 1, 1): {"name": "New Year's Day", "category": "Observance"},
    date(2025, 1, 14): {"name": "Makar Sankranti / Pongal", "category": "Major Festival"},
    date(2025, 1, 26): {"name": "Republic Day", "category": "National Holiday"},
    date(2025, 2, 26): {"name": "Maha Shivratri", "category": "Major Festival"},
    date(2025, 3, 14): {"name": "Holi", "category": "Major Festival"},
    date(2025, 3, 30): {"name": "Gudi Padwa / Ugadi", "category": "Major Festival"},
    date(2025, 3, 31): {"name": "Eid al-Fitr", "category": "Major Festival"},
    date(2025, 4, 6): {"name": "Ram Navami", "category": "Major Festival"},
    date(2025, 4, 10): {"name": "Mahavir Jayanti", "category": "Gazetted Holiday"},
    date(2025, 4, 18): {"name": "Good Friday", "category": "Gazetted Holiday"},
    date(2025, 5, 12): {"name": "Buddha Purnima", "category": "Gazetted Holiday"},
    date(2025, 6, 7): {"name": "Eid al-Adha (Bakrid)", "category": "Major Festival"},
}

def get_imd_season(month: int) -> str:
    """India Meteorological Department (IMD) four seasonal definitions."""
    if month in [1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

def enrich_calendar_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Enriches a DataFrame with rigorous time-series calendar and festival features.
    Guarantees no future leakage: each feature is purely deterministic from the calendar timestamp.
    """
    df = df.copy()
    dt = df[datetime_col]
    
    # 1. Standard calendar divisions
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek # 0=Monday, 6=Sunday
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["season"] = df["month"].apply(get_imd_season)
    
    # 2. Season dummy / ordinal
    season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post-Monsoon": 3}
    df["season_code"] = df["season"].map(season_map)
    
    # 3. Cyclical sine/cosine encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365.25)

    # 4. Festival & Holiday lookup
    dates_series = dt.dt.date
    df["festival_name"] = dates_series.apply(lambda d: INDIAN_EVENTS.get(d, {}).get("name", "None"))
    df["festival_category"] = dates_series.apply(lambda d: INDIAN_EVENTS.get(d, {}).get("category", "None"))
    
    df["is_festival"] = (df["festival_name"] != "None").astype(int)
    df["is_national_holiday"] = (df["festival_category"] == "National Holiday").astype(int)
    df["is_public_holiday"] = df["festival_category"].isin(["National Holiday", "Gazetted Holiday", "Major Festival"]).astype(int)

    # 5. Pre-festival and Post-festival Proximity Windows
    # Collect all festival dates
    fest_dates = sorted(list(INDIAN_EVENTS.keys()))
    
    # Vectorized computation of days to nearest festival
    def compute_distance(current_d):
        diffs = [(current_d - fd).days for fd in fest_dates]
        abs_diffs = [abs(d) for d in diffs]
        min_idx = int(np.argmin(abs_diffs))
        return diffs[min_idx] # positive if after festival, negative if before festival

    diff_days = [compute_distance(d) for d in dates_series]
    df["days_from_nearest_festival"] = diff_days
    df["days_to_nearest_festival"] = [-d for d in diff_days]
    df["is_pre_festival_window"] = ((df["days_from_nearest_festival"] >= -2) & (df["days_from_nearest_festival"] < 0)).astype(int)
    df["is_post_festival_window"] = ((df["days_from_nearest_festival"] > 0) & (df["days_from_nearest_festival"] <= 2)).astype(int)
    
    return df
