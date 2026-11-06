import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    
    # Load modelling table
    df = pd.read_csv("data/modelling_table.csv")
    
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 1. Split: 70% train, 15% val, 15% test (840 / 180 / 180)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)
    
    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())
    
    checks_b = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
        ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
    ]
    pd.DataFrame(checks_b, columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)
    
    # 3. Baseline (mean prediction)
    val_mean_pred = np.full_like(y_val, y_train.mean(), dtype=float)
    base_mae = mean_absolute_error(y_val, val_mean_pred)
    base_mse = mean_squared_error(y_val, val_mean_pred)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, val_mean_pred)
    
    metrics_rows = [{
        "model": "mean_baseline", "split": "val",
        "MAE": round(base_mae, 4), "MSE": round(base_mse, 4),
        "RMSE": round(base_rmse, 4), "R2": round(base_r2, 4)
    }]
    
    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    tree_results = {}
    
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
        va_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        label_d = str(d) if d is not None else "unlimited"
        depth_rows.append({"max_depth": label_d, "train_rmse": round(tr_rmse, 4), "val_rmse": round(va_rmse, 4)})
        tree_results[d] = va_rmse
        
    pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)
    
    # Best depth by lowest val_rmse
    best_depth = min(depths, key=lambda d: tree_results[d])
    
    # 5. Four candidate models
    # Linear Regression (with pipeline/scaler)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_val_scaled)
    
    # Decision Tree (optimal depth)
    dt_best = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
    dt_best.fit(X_train, y_train) # trees scale invariant
    dt_pred = dt_best.predict(X_val)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    
    candidates = [
        ("linear_regression", lr_pred),
        ("decision_tree", dt_pred),
        ("random_forest", rf_pred),
        ("gradient_boosting", gb_pred)
    ]
    
    val_model_scores = {}
    for name, pred in candidates:
        mae = mean_absolute_error(y_val, pred)
        mse = mean_squared_error(y_val, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, pred)
        val_model_scores[name] = rmse
        metrics_rows.append({
            "model": name, "split": "val",
            "MAE": round(mae, 4), "MSE": round(mse, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)
        })
        
    # 6. Learning-rate experiment for Gradient Boosting
    lr_rows = []
    for lrate in [0.01, 0.1, 1.0]:
        gb_exp = GradientBoostingRegressor(n_estimators=100, learning_rate=lrate, random_state=42)
        gb_exp.fit(X_train, y_train)
        pred_exp = gb_exp.predict(X_val)
        lr_rows.append({
            "learning_rate": lrate,
            "val_rmse": round(np.sqrt(mean_squared_error(y_val, pred_exp)), 4),
            "val_r2": round(r2_score(y_val, pred_exp), 4)
        })
    pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    # 7. Cross-validation for top 2 models
    sorted_candidates = sorted(val_model_scores.items(), key=lambda x: x[1])
    top_two = [sorted_candidates[0][0], sorted_candidates[1][0]]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    
    X_train_val_df = pd.concat([X_train, X_val])
    y_train_val_ser = pd.concat([y_train, y_val])
    X_train_val_scaled = scaler.fit_transform(X_train_val_df)
    
    for mname in top_two:
        if mname == "linear_regression":
            model_obj = LinearRegression()
            X_data = X_train_val_scaled
        elif mname == "decision_tree":
            model_obj = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
            X_data = X_train_val_df
        elif mname == "random_forest":
            model_obj = RandomForestRegressor(n_estimators=100, random_state=42)
            X_data = X_train_val_df
        else:
            model_obj = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            X_data = X_train_val_df
            
        scores = cross_val_score(model_obj, X_data, y_train_val_ser, cv=kf, scoring="r2")
        cv_rows.append({
            "model": mname,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4)
        })
        
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)
    
    # Final model: winning candidate evaluated on test set exactly once
    winner_name = sorted_candidates[0][0]
    if winner_name == "linear_regression":
        final_pred = lr.predict(X_test_scaled)
    elif winner_name == "decision_tree":
        final_pred = dt_best.predict(X_test)
    elif winner_name == "random_forest":
        final_pred = rf.predict(X_test)
    else:
        final_pred = gb.predict(X_test)
        
    metrics_rows.append({
        "model": winner_name, "split": "test",
        "MAE": round(mean_absolute_error(y_test, final_pred), 4),
        "MSE": round(mean_squared_error(y_test, final_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, final_pred)), 4),
        "R2": round(r2_score(y_test, final_pred), 4)
    })
    
    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print("Part B execution complete. Check outputs/ for results.")

if __name__ == "__main__":
    run_part_b()
