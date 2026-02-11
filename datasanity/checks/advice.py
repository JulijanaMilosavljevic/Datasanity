def generate_modeling_advice(results: dict) -> dict:
    """
    Generate practical modeling advice based on earlier checks.
    Expects the full results dict from core (with keys: imbalance, missing, id_columns, leakage, duplicates).
    """
    advice = []
    risks = []

    imbalance = results.get("imbalance", {})
    missing = results.get("missing", {})
    ids = results.get("id_columns", {})
    leakage = results.get("leakage", {})
    duplicates = results.get("duplicates", {})

    task = imbalance.get("task_hint", "classification")
    n_unique = imbalance.get("n_unique")

    # --- General hygiene advice ---
    if ids.get("id_like_columns"):
        risks.append("ID-like columns can cause memorization / poor generalization.")
        advice.append("Drop ID-like columns (e.g., customer_id) or use them only for grouping/splitting, not as features.")

    if leakage.get("suspicious_features"):
        risks.append("Possible leakage can inflate offline metrics and fail in production.")
        advice.append("Audit suspicious features and ensure they’re available at prediction time (no future info).")

    if duplicates.get("num_duplicates", 0) > 0:
        risks.append("Duplicate rows can bias training and evaluation.")
        advice.append("Remove duplicates; if time-series/user data, deduplicate per entity/time window.")

    if missing.get("high_missing_columns"):
        advice.append("Handle missingness: impute (median/most_frequent), add missing indicators, or drop high-missing columns.")

    # --- Task-specific advice ---
    if task == "classification":
        if imbalance.get("warning"):
            advice.append("Use stratified split; consider class weights or resampling (SMOTE/undersampling).")
            advice.append("Prefer macro-F1 / balanced accuracy for multi-class; for imbalanced binary use PR-AUC, recall/precision.")
        if isinstance(n_unique, int) and n_unique > 15:
            advice.append("If too many classes: consider label grouping or binning to reduce class cardinality.")
        advice.append("Start with strong baselines: Logistic Regression / Linear SVM / LightGBM/XGBoost.")
    else:
        # regression
        advice.append("Use train/validation split appropriate for data (time-based if temporal).")
        advice.append("Use MAE/RMSE; check residuals and outliers; consider log-transform if target is heavy-tailed.")
        advice.append("Start with baselines: Linear/Ridge, RandomForestRegressor, LightGBM/XGBoost.")

    # Nice compact output
    return {
        "task_hint": task,
        "top_risks": risks[:5],
        "recommended_actions": advice[:10],
    }
