import pandas as pd

def check_class_imbalance(df, target):
    if target not in df.columns:
        return {"error": "Target column not found."}

    y = df[target]
    n = len(y)
    nunique = int(y.nunique(dropna=False))

    # --- Heuristics: is this more like regression? ---
    # If many unique values (especially numeric), it's likely regression or should be binned.
    is_numeric = pd.api.types.is_numeric_dtype(y)

    # "Many classes" thresholds (tunable)
    many_unique_absolute = nunique > 15
    many_unique_relative = (nunique / max(n, 1)) > 0.05  # e.g., >5% unique of rows
    likely_regression = is_numeric and (many_unique_absolute or many_unique_relative)

    # Distribution (still useful even for numeric target, but could be huge)
    counts = y.value_counts(normalize=True, dropna=False)

    warning = None
    recommendation = None
    task_hint = "classification"

    if likely_regression:
        task_hint = "regression"
        recommendation = (
            "Target looks continuous / high-cardinality. Consider regression, "
            "or bin the target into fewer groups before classification."
        )
        # Not a "class imbalance" issue per se, but it's still a modeling warning
        warning = "Target likely better treated as regression (or binned classification)."
    else:
        # Standard imbalance warning for classification-like targets
        if nunique > 50:
            warning = "High number of classes. Consider binning/label grouping."
            recommendation = "Reduce class cardinality (binning) or revisit target definition."
        elif counts.min() < 0.1:
            warning = "Severe class imbalance detected."
            recommendation = (
                "Consider stratified split, class weights, resampling, and metrics like macro-F1."
            )
        else:
            recommendation = "No severe imbalance detected."

    return {
        "n_rows": n,
        "n_unique": nunique,
        "is_numeric": bool(is_numeric),
        "task_hint": task_hint,              # "classification" or "regression"
        "distribution": counts.to_dict(),    # may be large; ok for now
        "warning": warning,
        "recommendation": recommendation,
    }
