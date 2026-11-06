import pandas as pd
import numpy as np

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    """Clean raw shipments dataframe and return (clean_df, medians)."""
    df = raw.copy()
    
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. service_type: strip and title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. payment_mode: replace missing with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. declared_value: remove rupee sign and commas, convert to numeric
    if df["declared_value"].dtype == object:
        val_cleaned = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
        df["declared_value_inr"] = pd.to_numeric(val_cleaned, errors="coerce")
    else:
        df["declared_value_inr"] = df["declared_value"]
    df = df.drop(columns=["declared_value"])
    
    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    
    # 6. parcel_weight_kg: values above 100 are scanner-fault sentinels -> NaN
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. Compute or apply medians for filling gaps
    fill_cols = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {col: df[col].median() for col in fill_cols}
    
    for col in fill_cols:
        df[col] = df[col].fillna(medians[col])
        
    return df, medians

def build_features(clean, hubs):
    """Build feature table by merging hubs and engineering required features."""
    df = clean.merge(hubs, on="hub_id", how="left")
    
    # promised_hours based on service_type
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    
    # is_cod: 1 if payment_mode is COD, else 0
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    
    # is_weekend: 1 if Saturday (5) or Sunday (6)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    
    # dispatch_hour: integer hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    
    # log_declared_value: natural log of declared_value_inr
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df
