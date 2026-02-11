from __future__ import annotations
import pandas as pd


def _count_feature_types(df: pd.DataFrame, target: str | None = None) -> dict:
    X = df.drop(columns=[target], errors="ignore") if target else df
    n_cols = X.shape[1]

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    # bool tretiramo kao numeric-ish
    return {
        "n_features": n_cols,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "cat_ratio": (len(cat_cols) / n_cols) if n_cols else 0.0,
    }


def suggest_models(df: pd.DataFrame, target: str, results: dict) -> dict:
    """
    Returns a ranked list of model suggestions and a baseline recipe.
    Uses dataset shape + feature types + earlier checks.
    """
    n_rows = df.shape[0]
    feat = _count_feature_types(df, target)

    imb = results.get("imbalance", {}) or {}
    task = imb.get("task_hint", "classification")
    n_unique = imb.get("n_unique")

    has_cat = feat["n_categorical"] > 0
    cat_ratio = feat["cat_ratio"]

    imbalance_warning = bool(imb.get("warning"))

    suggestions = []

    def add(name: str, why: list[str], when: list[str], notes: list[str] | None = None):
        suggestions.append({
            "model": name,
            "why": why,
            "when_to_use": when,
            "notes": notes or []
        })

    # --- Common guidance ---
    if task == "classification":
        is_multiclass = isinstance(n_unique, int) and n_unique > 2
        if has_cat:
            # Tree-based models handle mixed types well (after encoding)
           add(
                "LightGBM / XGBoost (Gradient Boosted Trees)",
                why=[
                    "Strong performance on tabular data with nonlinear relationships.",
                    "Captures feature interactions with minimal feature engineering."
                ],
                when=[
                    "Mixed numerical and categorical features.",
                    "Medium to large datasets (e.g., >2k rows)."
                ],
                notes=[
                    "Use OneHotEncoder or target encoding for categorical features.",
                    "For class imbalance: try class weights or scale_pos_weight."
                ]
        )

        add(
            "Logistic Regression (Baseline)",
            why=[
                "Fast and interpretable baseline model.",
                "Great first reference before using more complex models."
            ],
            when=[
                "Small to medium datasets.",
                "After encoding categorical variables."
            ],
            notes=[
                "Tune regularization strength (C).",
                "Use multinomial + saga solver for multi-class."
            ]
        )

        add(
            "Linear SVM",
            why=[
                "Often outperforms logistic regression in high-dimensional feature spaces.",
                "Works well with sparse one-hot encoded data."
            ],
            when=[
                "Large number of categorical features.",
                "Not extremely large datasets."
            ],
            notes=[
                "Use probability calibration if probabilities are required."
            ]
        )


        if imbalance_warning:
            # Add explicit suggestion for imbalance-friendly algorithms/metrics
            add(
                "Class-weighted Trees / Balanced Random Forest",
                why=[
                    "More robust when the target distribution is highly imbalanced.",
                    "Reduces bias toward the majority class."
                ],
                when=[
                    "Severe class imbalance detected."
                ],
                notes=[
                    "Evaluate with macro-F1 or PR-AUC instead of accuracy."
                ]
            )


        baseline = {
            "split": "Stratified train/val split",
            "metrics": "macro-F1 (multiclass) / PR-AUC + recall (binary imbalanced)",
            "pipeline": [
                "Drop ID-like columns",
                "Handle missing (simple impute + indicators)",
                "Encode categoricals (OneHot or target encoding)",
                "Train baselines: LogReg, then XGBoost/LightGBM",
            ],
        }

    else:
        # regression
        add(
            "LightGBM / XGBoost Regressor",
            why=[
                "Top-performing models for tabular regression problems.",
                "Handles nonlinearities and feature interactions well."
            ],
            when=[
                "Mixed feature types or nonlinear relationships.",
                "Medium to large datasets."
            ],
            notes=[
                "Evaluate with MAE and RMSE.",
                "Consider log-transform if target has heavy tails."
            ]
        )

        add(
            "Ridge Regression (Baseline)",
            why=[
                "Fast and stable linear baseline.",
                "Good reference before moving to nonlinear models."
            ],
            when=[
                "Smaller datasets or mostly linear relationships."
            ],
            notes=[
                "Standardize numerical features before training."
            ]
        )

        add(
            "RandomForestRegressor",
            why=[
                "A good baseline when you want nonlinearity without tuning.",
                "Less sensitive to feature scales."
            ],
            when=[
                "Small/medium sized dataset.",
                "You want a fast robust check."
            ],
            notes=[
                "Can underfit/overfit — uses CV."
            ]
        )

        baseline = {
            "split": "Train/val split (time-based ako ima vreme)",
            "metrics": "MAE + RMSE",
            "pipeline": [
                "Drop ID-like columns",
                "Handle missing (median/most_frequent + indicators)",
                "Encode categoricals (OneHot/target encoding)",
                "Train baselines: Ridge, RF, then XGBoost/LightGBM",
            ],
        }

    # Simple ranking heuristic: prefer trees when many categoricals or more rows
    def score_suggestion(s):
        m = s["model"].lower()
        sc = 0
        if "xgboost" in m or "lightgbm" in m:
            sc += 3
            if n_rows > 2000:
                sc += 2
            if cat_ratio > 0.25:
                sc += 1
        if "logistic" in m or "ridge" in m:
            sc += 2
        if "svm" in m:
            sc += 1
            if cat_ratio > 0.35:
                sc += 1
        if "randomforest" in m:
            sc += 1
        return sc

    suggestions = sorted(suggestions, key=score_suggestion, reverse=True)[:5]

    return {
        "task_hint": task,
        "n_rows": n_rows,
        "n_features": feat["n_features"],
        "feature_mix": {
            "n_numeric": feat["n_numeric"],
            "n_categorical": feat["n_categorical"],
            "cat_ratio": round(feat["cat_ratio"], 3),
        },
        "top_models": suggestions[:3],
        "baseline_plan": baseline,
    }
