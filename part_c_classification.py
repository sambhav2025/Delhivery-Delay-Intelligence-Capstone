import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from src.shipment_prep import FEATURES, NOISE_COLUMNS

def get_metrics_dict(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": model_name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4)
    }

def run_part_c():
    os.makedirs("outputs", data_dir := "data", exist_ok=True)
    
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 1. Stratified split (840 / 180 / 180)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, stratify=y_train_val, random_state=42)
    
    models_rows = []
    
    # 2. Blind baseline ("not delayed" for all)
    val_blind_pred = np.zeros_like(y_val)
    models_rows.append(get_metrics_dict("blind_all_on_time", y_val, val_blind_pred))
    
    # Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))]) if 'Pipeline' in globals() else None
    # Let's import Pipeline explicitly
    from sklearn.pipeline import Pipeline
    
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    models_rows.append(get_metrics_dict("logistic_regression", y_val, lr_pipe.predict(X_val)))
    
    # Decision Tree tuning over depth in [2, 3, 4, 5, 6, 8, 10]
    best_dt_f1 = -1
    best_dt_depth = 2
    for d in [2, 3, 4, 5, 6, 8, 10]:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        f1 = f1_score(y_val, dt.predict(X_val), zero_division=0)
        if f1 > best_dt_f1:
            best_dt_f1 = f1
            best_dt_depth = d
            
    dt_best = DecisionTreeClassifier(criterion="gini", max_depth=best_dt_depth, random_state=42)
    dt_best.fit(X_train, y_train)
    models_rows.append(get_metrics_dict(f"decision_tree_depth_{best_dt_depth}", y_val, dt_best.predict(X_val)))
    
    # Default Random Forest
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_default.fit(X_train, y_train)
    models_rows.append(get_metrics_dict("random_forest_default", y_val, rf_default.predict(X_val)))
    
    pd.DataFrame(models_rows).to_csv("outputs/part_c_models.csv", index=False)
    
    # 3. Feature signals & SelectKBest
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_full.fit(X_train, y_train)
    rf_imp = rf_full.feature_importances_
    
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_
    
    feat_score_rows = []
    for feat, score, imp in zip(FEATURES, f_scores, rf_imp):
        feat_score_rows.append({
            "feature": feat,
            "f_score": round(float(score), 4),
            "rf_importance": round(float(imp), 4),
            "is_noise": feat in NOISE_COLUMNS
        })
    feat_df = pd.DataFrame(feat_score_rows)
    feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
    feat_df = feat_df.sort_values("f_rank")
    feat_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].to_csv("outputs/part_c_feature_scores.csv", index=False)
    
    # SelectKBest evaluation for k in [5, 7, 10, 12, 17]
    selectk_rows = []
    for k_val in [5, 7, 10, 12, 17]:
        pipe_k = Pipeline([
            ("selector", SelectKBest(f_classif, k=k_val)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe_k.fit(X_train, y_train)
        val_f1 = f1_score(y_val, pipe_k.predict(X_val), zero_division=0)
        selectk_rows.append({"k": k_val, "val_f1": round(val_f1, 4)})
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)
    
    # Fix k = 7 for remaining tasks
    # 4. Tuning with k = 7
        # Transform train/val with SelectKBest(k=7)
    skb = SelectKBest(f_classif, k=7)
    X_train_k7 = skb.fit_transform(X_train, y_train)
    X_val_k7 = skb.transform(X_val)
    
    # Grid Search
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }
    rf_base = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_base, param_grid, cv=5)
    grid_search.fit(X_train_k7, y_train)
    grid_best_pred = grid_search.predict(X_val_k7)
    grid_val_f1 = f1_score(y_val, grid_best_pred, zero_division=0)
    
    # Randomized Search
    rand_search = RandomizedSearchCV(rf_base, param_grid, n_iter=9, cv=5, random_state=42)
    rand_search.fit(X_train_k7, y_train)
    rand_best_pred = rand_search.predict(X_val_k7)
    rand_val_f1 = f1_score(y_val, rand_best_pred, zero_division=0)
    
    # Default forest on same 7 features
    rf_def_k7 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def_k7.fit(X_train_k7, y_train)
    def_val_f1 = f1_score(y_val, rf_def_k7.predict(X_val_k7), zero_division=0)
    
    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid_search.best_params_),
            "best_cv_accuracy": round(grid_search.best_score_, 4),
            "val_f1": round(grid_val_f1, 4)
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rand_search.best_params_),
            "best_cv_accuracy": round(rand_search.best_score_, 4),
            "val_f1": round(rand_val_f1, 4)
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(def_val_f1, 4)
        }
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    
    # 5. Choose final configuration based on highest val_f1
    # Priority tie-break: default, then grid, then randomized
    best_row_idx = max(range(len(tuning_rows)), key=lambda i: (tuning_rows[i]["val_f1"], 2 if i==2 else (1 if i==0 else 0)))
    chosen_name = tuning_rows[best_row_idx]["search"]
    
    if chosen_name == "grid_search":
        best_params = grid_search.best_params_
        final_clf = RandomForestClassifier(random_state=42, **best_params)
    elif chosen_name == "randomized_search":
        best_params = rand_search.best_params_
        final_clf = RandomForestClassifier(random_state=42, **best_params)
    else:
        best_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
        final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", final_clf)
    ])
    final_pipeline.fit(X_train, y_train)
    
    # 6. Alert thresholds
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_rows = []
    
    valid_thresholds_for_recall = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        thresh_rows.append({
            "threshold": t, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)
        })
        if rec >= 0.80:
            valid_thresholds_for_recall.append(t)
            
    pd.DataFrame(thresh_rows).to_csv("outputs/part_c_thresholds.csv", index=False)
    
    chosen_threshold = max(valid_thresholds_for_recall) if valid_thresholds_for_recall else 0.1
    
    # Save final config
    max_d_str = str(best_params.get("max_depth", "unlimited"))
    if max_d_str == "None": max_d_str = "unlimited"
    
    config_row = [{
        "selector_k": 7,
        "n_estimators": best_params.get("n_estimators", 100),
        "max_depth": max_d_str,
        "min_samples_leaf": best_params.get("min_samples_leaf", 1),
        "threshold": chosen_threshold,
        "chosen_from": chosen_name
    }]
    pd.DataFrame(config_row).to_csv("outputs/part_c_final_config.csv", index=False)
    
    # 7. Test evaluation with chosen threshold
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    test_metrics = get_metrics_dict("final_model_test", y_test, test_preds)
    test_metrics["threshold"] = chosen_threshold
    # reorder columns to put threshold first
    cols = ["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]
    pd.DataFrame([test_metrics])[cols].to_csv("outputs/part_c_final_test.csv", index=False)
    
    print("Part C execution complete. Check outputs/ for results.")

if __name__ == "__main__":
    run_part_c()
