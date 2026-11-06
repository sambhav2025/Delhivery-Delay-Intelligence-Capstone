import os
import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Load raw data
    raw_path = "data/raw_shipments.csv"
    hubs_path = "data/hubs.csv"
    
    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)
    
    # Track metrics for checks file
    raw_rows = len(raw)
    raw_dedup = raw.drop_duplicates()
    duplicate_rows_removed = raw_rows - len(raw_dedup)
    rows_after_dedup = len(raw_dedup)
    
    missing_distance_km_before = raw_dedup["distance_km"].isna().sum()
    missing_parcel_weight_before = raw_dedup[(raw_dedup["parcel_weight_kg"].isna()) | (raw_dedup["parcel_weight_kg"] <= 100)]["parcel_weight_kg"].isna().sum() # standard missing before sentinel check
    # Let's count sentinels specifically
    sentinel_weight_rows = (raw_dedup["parcel_weight_kg"] == 999.0).sum()
    missing_parcel_weight_before_pure = raw_dedup["parcel_weight_kg"].isna().sum()
    
    # service type spellings before/after
    svc_raw = raw_dedup["service_type"].astype(str)
    service_type_spellings_before = svc_raw.nunique()
    
    cleaned_df, medians = clean_shipments(raw)
    service_type_spellings_after = cleaned_df["service_type"].nunique()
    
    missing_declared_value_before = raw_dedup["declared_value"].isna().sum() if "declared_value" in raw_dedup else 0
    missing_payment_mode_before = raw_dedup["payment_mode"].isna().sum()
    
    median_distance_km = round(raw_dedup["distance_km"].median(), 2)
    median_parcel_weight_kg = round(raw_dedup[raw_dedup["parcel_weight_kg"] <= 100]["parcel_weight_kg"].median(), 2)
    
    # Build features & Modelling table
    modelling_df = build_features(cleaned_df, hubs)
    rows_after_hub_merge = len(modelling_df)
    
    # Target: is_delayed
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)
    
    # Save modelling table
    modelling_df.to_csv("data/modelling_table.csv", index=False)
    
    median_declared_value_inr = round(modelling_df["declared_value_inr"].median(), 2)
    declared_value_mean_inr = round(modelling_df["declared_value_inr"].mean(), 2)
    log_declared_value_mean = round(modelling_df["log_declared_value"].mean(), 4)
    log_declared_value_median = round(modelling_df["log_declared_value"].median(), 4)
    
    nulls_in_features_and_target = modelling_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    
    delayed_count = int(modelling_df["is_delayed"].sum())
    delay_rate = round(delayed_count / len(modelling_df), 4)
    
    # Quartiles & Outlier bounds for delivery_hours
    q1 = modelling_df["delivery_hours"].quantile(0.25)
    q3 = modelling_df["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((modelling_df["delivery_hours"] < lower_bound) | (modelling_df["delivery_hours"] > upper_bound)).sum())
    
    checks_data = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before_pure),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", median_distance_km),
        ("median_parcel_weight_kg", median_parcel_weight_kg),
        ("median_declared_value_inr", median_declared_value_inr),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round(q1, 2)),
        ("hours_q3", round(q3, 2)),
        ("hours_lower_bound", round(lower_bound, 2)),
        ("hours_upper_bound", round(upper_bound, 2)),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("declared_value_median_inr", median_declared_value_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("log_declared_value_median", log_declared_value_median),
    ]
    pd.DataFrame(checks_data, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay by group table
    group_rows = []
    
    # 1. service_type
    for stype, grp in modelling_df.groupby("service_type"):
        group_rows.append({
            "dimension": "service_type",
            "group": stype,
            "shipments": len(grp),
            "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(grp["is_delayed"].mean(), 4),
            "mean_hours": round(grp["delivery_hours"].mean(), 2)
        })
        
    # 2. hub_tier
    for tier, grp in modelling_df.groupby("hub_tier"):
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(tier),
            "shipments": len(grp),
            "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(grp["is_delayed"].mean(), 4),
            "mean_hours": round(grp["delivery_hours"].mean(), 2)
        })
        
    # 3. distance_band
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_df["distance_band"] = pd.cut(modelling_df["distance_km"], bins=bins, labels=labels, include_lowest=True, right=True)
    for band, grp in modelling_df.groupby("distance_band", observed=False):
        group_rows.append({
            "dimension": "distance_band",
            "group": str(band),
            "shipments": len(grp),
            "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(grp["is_delayed"].mean(), 4),
            "mean_hours": round(grp["delivery_hours"].mean(), 2)
        })
        
    delay_by_group_df = pd.DataFrame(group_rows)
    delay_by_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # Correlations
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_matrix = modelling_df[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True)
    
    print("Part A execution complete. Check outputs/ for results.")

if __name__ == "__main__":
    run_part_a()
