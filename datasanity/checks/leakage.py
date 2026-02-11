import numpy as np

def check_target_leakage(df, target):
    numeric_df = df.select_dtypes(include=np.number)

    if target not in numeric_df.columns:
        return {"suspicious_features": []}

    corrs = numeric_df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
    suspicious = [col for col in corrs.index if col != target and corrs[col] > 0.95]

    return {
        "suspicious_features": suspicious,
        "warning": "Possible target leakage detected." if suspicious else None
    }
