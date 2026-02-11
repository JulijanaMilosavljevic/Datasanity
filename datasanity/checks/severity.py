def compute_dataset_severity(results: dict) -> dict:
    """
    Compute a simple 0–100 risk score based on detected issues.
    Higher = riskier dataset for modeling.
    """
    score = 0
    reasons = []

    imbalance = results.get("imbalance", {})
    missing = results.get("missing", {})
    ids = results.get("id_columns", {})
    leakage = results.get("leakage", {})
    duplicates = results.get("duplicates", {})

    # --- Class imbalance ---
    if imbalance.get("warning"):
        score += 30
        reasons.append("Class imbalance detected")

    # --- ID-like columns ---
    if ids.get("id_like_columns"):
        score += 20
        reasons.append("ID-like columns present")

    # --- High missingness ---
    if missing.get("high_missing_columns"):
        score += 30
        reasons.append("Columns with high missing values")

    # --- Leakage ---
    if leakage.get("suspicious_features"):
        score += 40
        reasons.append("Potential target leakage")

    # --- Duplicates ---
    if duplicates.get("num_duplicates", 0) > 0:
        score += 10
        reasons.append("Duplicate rows detected")

    # Cap score at 100
    score = min(score, 100)

    if score < 30:
        level = "Low risk"
        color = "green"
    elif score < 70:
        level = "Moderate risk"
        color = "yellow"
    else:
        level = "High risk"
        color = "red"

    return {
        "score": score,
        "risk_level": level,
        "color": color,
        "reasons": reasons
    }

